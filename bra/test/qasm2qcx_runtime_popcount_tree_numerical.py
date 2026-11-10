#!/usr/bin/env python3
"""Audit nested runtime-range popcount expression trees (#511).

Run with --bra /path/to/bra, including a UBSan build. References combine
individual little-endian bit vectors and apply each operation in forward order.
"""

import argparse
import cmath
import itertools
import math
import operator
import pathlib
import random
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_runtime_popcount_operand_chain_numerical as operands


def combine(left, right, operation):
    assert len(left[1]) == len(right[1]) and left[1]
    function = {'&': operator.and_, '|': operator.or_, '^': operator.xor}[operation]
    return f'({left[0]}) {operation} ({right[0]})', [function(a, b) for a, b in zip(left[1], right[1])]


def wrap(node, chain, tokens=None):
    return (operands.chain_expression(node[0], chain if tokens is None else tokens),
            operands.transform(node[1], chain))


def complemented(node, depth):
    return '~' * depth + f'({node[0]})', operands.binary.complemented(node[1], depth)


def build(shape, leaves, operations):
    a, b, c, mask = leaves
    first, second = operations
    if shape == 'left':
        return combine(combine(a, b, first), c, second)
    if shape == 'right':
        return combine(a, combine(b, c, second), first)
    return combine(combine(a, b, first), combine(c, mask, second), first)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    comparisons = 0
    setUp = operands.BackendTests.setUp
    execute = operands.BackendTests.execute
    program = operands.BackendTests.program
    check_exact = operands.BackendTests.check_exact

    def check_nodes(self, source, nodes, *, preservation=(), batch_size=64):
        for first in range(0, len(nodes), batch_size):
            batch = nodes[first:first + batch_size]
            statements = [source]
            outputs, expected = [], []
            for index, (expression, bits) in enumerate(batch):
                name = f'out{index}'
                statements.append(f'uint {name}=popcount({expression}); ')
                outputs.append((name, None))
                expected.append(sum(bits))
            for index, (expression, expected_value) in enumerate(preservation):
                name = f'preserved{index}'
                statements.append(f'uint {name}=uint({expression}); ')
                outputs.append((name, None))
                expected.append(expected_value)
            self.check_exact(''.join(statements), outputs, expected)
            type(self).comparisons += len(batch)

    def check_case(self, sources, constants, depths, operations):
        bounds_names = (('first', 'stride', 'last'), ('start', 'step', 'stop'), ('begin', 'delta', 'finish'))
        source, leaves, same = [], [], []
        for name, (width, pattern, bounds), constant, names in zip(('a', 'b', 'c'), sources, constants, bounds_names):
            source.append(('const ' if constant else '') + f'bit[{width}] {name}="{pattern:0{width}b}"; ')
            for variable, value in zip(names, bounds):
                source.append(f'int {variable}={value}; ')
            leaves.append((f'{name}[{names[0]}:{names[1]}:{names[2]}]', operands.binary.selected(width, pattern, bounds)))
            same.append(f'{name}=="{pattern:0{width}b}"')
        width = len(leaves[0][1])
        self.assertTrue(all(len(bits) == width for _, bits in leaves))
        mask = [index % 2 for index in range(width)]
        pattern = sum(bit << index for index, bit in enumerate(mask))
        source.append(f'const bit[{width}] MASK="{pattern:0{width}b}"; ')
        nodes = []
        for groups, operations_pair, shape in itertools.product(depths, operations, ('left', 'right', 'balanced')):
            a, b, c = [complemented(node, depth) for node, depth in zip(leaves, groups[:3])]
            nodes.append(complemented(build(shape, (a, b, c, ('MASK', mask)), operations_pair), groups[3]))
        self.check_nodes(''.join(source), nodes,
                         preservation=((' && '.join(same), 1), (f'popcount({leaves[0][0]})', sum(leaves[0][1]))))

    def test_exhaustive_small_patterns_operators_tree_shapes_and_complements(self):
        for width in (1, 2):
            depths = tuple(itertools.product((0, 1), repeat=4)) if width == 1 else ((0, 0, 0, 0), (1, 1, 1, 1))
            for a, b, c in itertools.product(range(1 << width), repeat=3):
                self.check_case(((width, a, (0, 1, width - 1)), (width, b, (width - 1, -1, 0)),
                                 (width, c, (0, 1, width - 1))),
                                (bool(a % 2), bool(b % 2), bool(c % 2)), depths,
                                tuple(itertools.product(('&', '|', '^'), repeat=2)))

    def test_all_native_widths_wide_sources_aliases_and_source_kinds(self):
        rng = random.Random(511)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        operations = (('&', '^'), ('|', '&'), ('^', '|'))
        for width in (*range(1, native + 2), 65, 129):
            self.check_case(((width, rng.getrandbits(width), (0, 1, width - 1)),
                             (width + 1, rng.getrandbits(width + 1), (width, -1, 1)),
                             (2 * width, rng.getrandbits(2 * width), (2 * width - 1, -2, 1))),
                            (bool(width % 2), bool(width // 2 % 2), bool(width // 4 % 2)),
                            ((2, 3, 2, 3),), operations)
        selections = (((4, (-4, 1, 3)), (8, (7, -1, 0)), (16, (15, -2, 1))),
                      ((8, (-1, -2, -7)), (129, (119, 3, 128)), (8, (0, 2, 6))),
                      ((1, (-1, 1, 0)), (129, (128, -128, 0)), (8, (7, -1, 6))),
                      ((129, (2, 4, 5)), (1, (0, 1, 0)), (4, (3, -1, 3))))
        for selection in selections:
            sources = tuple((width, rng.getrandbits(width), bounds) for width, bounds in selection)
            for constants in itertools.product((False, True), repeat=3):
                self.check_case(sources, constants, ((1, 2, 3, 2),), operations)

    def test_operations_at_intermediate_nodes_distances_and_nested_calls(self):
        for width in (1, 4, 33, 65, 129):
            patterns = tuple(int((bits * width)[:width], 2) for bits in ('1011', '1101', '0011'))
            source = 'int first=0; int last=' + str(width - 1) + '; '
            for name, pattern in zip(('a', 'b', 'c'), patterns):
                source += f'bit[{width}] {name}="{pattern:0{width}b}"; '
            source += f'const bit[{width}] MASK="{patterns[1]:0{width}b}"; bool yes=true; uint huge=uint(-1); '
            nodes = []
            for index, (name, distance) in enumerate(itertools.product(('rotl', 'rotr'),
                    sorted({-2147483648, -width - 1, -1, 0, 1, width, width + 1, 2147483647}))):
                token = f'distance{index}'
                source += f'int {token}={distance}; '
                leaves = tuple((f'{name}[first:last]', [(pattern >> bit) & 1 for bit in range(width)])
                               for name, pattern in zip(('a', 'b', 'c'), patterns))
                for operation in ('&', '|', '^'):
                    middle = combine(leaves[0], leaves[1], operation)
                    middle = wrap(middle, ((name, distance, 1), ('<<', 1, 2)), ((name, token, 1), ('<<', '1', 2)))
                    right = wrap(leaves[2], (('rotr', 1, 1),), (('rotr', 'yes', 1),))
                    nodes.append(wrap(combine(middle, right, '^'), (('>>', 1, 1),)))
            a, b, c = leaves
            for distance, token in ((width + 1, str(width + 1)), (4294967295, 'huge'), (10**40, str(10**40))):
                nodes.append(combine(wrap(combine(a, b, '&'), (('<<', distance, 1),), (('<<', token, 1),)), c, '^'))
            nested = combine(combine(a, b, '&'), c, '^')
            nested_count = sum(nested[1])
            nodes.append(wrap(combine(combine(a, c, '|'), b, '^'),
                              (('rotl', nested_count, 1), ('>>', 1, 0)),
                              (('rotl', f'popcount({nested[0]})', 1), ('>>', '1', 0))))
            self.check_nodes(source, nodes, preservation=((f'popcount(a[first:last])', sum(a[1])),))

    def test_seeded_deeper_trees_with_fresh_source_reads(self):
        rng = random.Random(511)
        for batch in range(16):
            source = ['int first=0; ']
            nodes, preservation = [], []
            for offset in range(8):
                case = batch * 8 + offset
                width = rng.choice((1, 4, 8, 33, 65, 129))
                patterns = [rng.getrandbits(width) for _ in range(4)]
                # All declarations precede the expressions; each expression
                # uses its own stop variable to retain the intended width.
                source.append(f'int last{offset}={width - 1}; ')
                serial = [0]
                def changed(node):
                    for _ in range(rng.randrange(4)):
                        serial[0] += 1
                        operation = rng.choice(('rotl', 'rotr', '<<', '>>'))
                        distance = rng.randrange(-2, 3) if operation in ('rotl', 'rotr') else rng.randrange(width + 2)
                        token = f'd{offset}_{serial[0]}'
                        source.append(f'int {token}={distance}; ')
                        depth = rng.randrange(4)
                        node = wrap(node, ((operation, distance, depth),), ((operation, token, depth),))
                    return node
                leaves = []
                for index, pattern in enumerate(patterns):
                    name = f'a{offset}_{index}'
                    source.append(('const ' if (case >> index) % 2 else '') + f'bit[{width}] {name}="{pattern:0{width}b}"; ')
                    leaves.append(changed((f'{name}[first:last{offset}]', [(pattern >> bit) & 1 for bit in range(width)])))
                    preservation.append((f'{name}=="{pattern:0{width}b}"', 1))
                def merged(left, right):
                    return changed(combine(left, right, rng.choice(('&', '|', '^'))))
                a, b, c, d = leaves
                node = (merged(merged(merged(a, b), c), d) if case % 3 == 0 else
                        merged(a, merged(b, merged(c, d))) if case % 3 == 1 else merged(merged(a, b), merged(c, d)))
                nodes.append(node)
                preservation.append((f'popcount(a{offset}_0[first:last{offset}])', patterns[0].bit_count()))
            self.check_nodes(''.join(source), nodes, preservation=preservation)

    def test_nested_consumers_self_assignment_and_loop_reentry(self):
        self.check_exact('bit[4] a="1011"; const bit[4] B="1101"; bit[4] c="0011"; int first=0; int last=3; '
            'uint result=1; result=popcount(rotl((a[first:last] & B[first:last]) ^ c[first:last],result)); '
            'array[uint,4] values={0,0,0,0}; values[result]=result; uint stored=values[2]; '
            'uint[1] narrow=popcount((a[first:last] & B[first:last]) ^ c[first:last]); '
            'bit[2] packed=bit[2](uint[2](popcount((a[first:last] & B[first:last]) ^ c[first:last]))); '
            'uint nested=popcount(uint[4](popcount((a[first:last] & B[first:last]) ^ c[first:last])));',
            (('result', None), ('stored', None), ('narrow', None), ('packed', 2), ('nested', None)), (2, 2, 0, 2, 1))
        for constants in itertools.product((False, True), repeat=3):
            source = ''.join(('const ' if constant else '') + f'bit[4] {name}="{pattern:04b}"; '
                             for name, pattern, constant in zip(('a', 'b', 'c'), (11, 13, 3), constants))
            counts = []
            for stop in (3, 2, 1, 0):
                a = operands.binary.selected(4, 11, (0, 1, stop))
                b = operands.binary.selected(4, 13, (stop, -1, 0))
                c = operands.binary.selected(4, 3, (0, 1, stop))
                counts.append(sum(combine(combine(('a', a), ('b', b), '^'), ('c', c), '&')[1]))
            self.check_exact(source + 'int first=0; int stop=3; uint total=0; uint count=0; '
                'while (stop>=0) { count=popcount((a[first:stop] ^ b[stop:-1:first]) & c[first:stop]); '
                'stop-=1; if (stop==1) { continue; } total+=count; if (stop==-1) { break; } }',
                (('total', None), ('count', None)), (sum(counts[i] for i in (0, 2, 3)), counts[3]))
        self.check_exact('bit[4] a="1111"; bit[4] b="1111"; bit[4] c="0000"; int first=0; int last=3; int visits=0; '
            'while (popcount((a[first:last] & b[first:last]) ^ c[first:last])>uint(0)) { a>>=1; visits+=1; }',
            (('visits', None), ('a', 4)), (4, 0))

    def test_reached_errors_subtree_completion_width_checks_and_destination_write_suppression(self):
        conversion = 'UINT to INT conversion out of range'
        cases = (('(a[first:0:last] & mask) ^ rotl(B[start/zero:stop],1/zero)', 'assertion failed', (), 1000000),
                 ('(rotl(a[first:last],huge) & B[start:1]) ^ mask', conversion, (), 1000000),
                 ('(a[first:last] & mask) ^ rotl(B[start:1],1/zero)', 'division by zero', ('22222',), 1000000),
                 ('(a[first:last] & mask) ^ rotl(B[start:1],1)', 'assertion failed', ('22222', '33333'), 1000000),
                 ('(a[first:last] & B[start:1]) ^ rotl(mask,1/zero)', 'assertion failed', (), 1000000),
                 ('(a[first:last] & mask) ^ (B[start:1] | rotl(mask,1/zero))', 'division by zero', ('22222',), 1000000),
                 ('(a[first:last] ^ a[first:last]) & rotl(mask,huge)', conversion, ('22222',), 1000000),
                 ('((a[first:last] & mask) << huge) ^ rotl(B[start:stop],huge)', conversion, ('22222',), 1000000),
                 ('rotr((a[first:last] & mask) ^ B[start:stop],huge)', conversion, ('22222',), 1000000),
                 ('rotl((a[first:last] & mask) ^ B[start:stop],popcount((a[first:last] & mask) ^ rotr(B[start:stop],huge)))',
                  conversion, ('22222',), 1000000),
                 ('(a[first:last] & mask) ^ B[start:0:stop]', 'assertion failed', ('22222',), 1000000),
                 ('(rotl(mask,huge) ^ a[first/zero:last]) & B[start:stop]', conversion, (), 1000000),
                 ('(a[first:last] & B[start:stop]) ^ c[first:last]', 'assertion failed', (), 2))
        cls = self.module.QASM2QCXConverter
        for value, error, markers, capacity in cases:
            ast = self.module.openqasm3.parser.parse('OPENQASM 3; bit[4] a="1011"; const bit[8] B="11011101"; '
                'bit[4] mask="1101"; bit[4] c="0011"; int first=0; int last=3; int start=0; int stop=3; int zero=0; '
                f'uint huge=uint(-1); uint result=6; result=popcount({value});')
            with patch.object(cls, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', capacity):
                converter = cls(ast); converter.visit(ast)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
            destination = converter._QASM2QCXConverter__capitalize_variable_name('result')
            lines = []
            for line in converter:
                if line.startswith(f'LET {destination} := QASM2QCX_'):
                    lines.append('PRINTLN 11111')
                lines.append(line)
                if line == '@QASM2QCX_POPCOUNT_0_TREE_CHILD_0_DONE':
                    lines.append('PRINTLN 22222')
                if line == '@QASM2QCX_POPCOUNT_0_TREE_CHILD_1_DONE':
                    lines.append('PRINTLN 33333')
            result = self.execute(lines)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual(result.stdout.split(), list(markers))
            type(self).checked += 1

    def test_skipped_domains_and_short_circuiting(self):
        base = ('bit[4] a="1011"; const bit[4] B="1101"; int first=0; int last=3; int zero=0; '
                'uint huge=uint(-1); uint result=7; int loop_first=2; int loop_last=1; ')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        for value in ('(a[first:0:999] & B) ^ rotl(B[first:last],1/zero)',
                      '(a[first:last] ^ B) & rotr(B[first:last],huge)',
                      'rotr((a[first:last] & B) ^ B[first:last],huge) >> -1'):
            for wrapper in wrappers:
                self.check_exact(base + wrapper.format(f'result=popcount({value});'), (('result', None),), (7,))
            self.check_exact(base + f'bool x=false && bool(popcount({value})); bool y=true || bool(popcount({value}));',
                             (('x', None), ('y', None)), (0, 1))

    def test_measurement_feedback_and_normal_fused_gate_parameters(self):
        for preparation in ('', 'x q[0]; ', 'h q[0]; ', 'h q[0]; x q[0]; '):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; ' + preparation +
                'bit[2] measured=measure q; const bit[2] MASK="10"; bit[2] zero="00"; int first=0; int last=1; '
                'uint count=popcount((rotl(measured[first:last],1) & MASK[first:last]) ^ zero[first:last]); '
                'bool valid=count==uint(measured[0]); '
                'if (count==uint(1)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0];',
                (('valid', None), ('feedback', None)), (1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        leaves = (('a[first:last]', [1, 1, 0, 1]), ('b[first:last]', [1, 0, 1, 1]),
                  ('c[first:last]', [1, 1, 0, 0]), ('mask', [1, 0, 1, 0]))
        for operations, shape, fused in itertools.product(itertools.product(('&', '|', '^'), repeat=2),
                                                          ('left', 'right', 'balanced'), (False, True)):
            node = wrap(build(shape, leaves, operations), (('rotl', -1, 1), ('>>', 1, 0)))
            phase = cmath.exp(sum(node[1]) * 1j * math.pi / 8)
            for gate in (f'p(float(popcount({node[0]}))*pi/8) q;', f'pow(popcount({node[0]})) @ p(pi/8) q;'):
                source = ('include "stdgates.inc"; qubit[2] q; bit[4] a="1011"; bit[4] b="1101"; '
                          'bit[4] c="0011"; bit[4] mask="0101"; int first=0; int last=3; h q; ' + gate)
                BackendAuditTests.check(self, self.program(source, fused=fused),
                                        (0.5, phase/2, phase/2, phase*phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime-range popcount tree programs checked: {BackendTests.checked}; bit-vector comparisons: {BackendTests.comparisons}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
