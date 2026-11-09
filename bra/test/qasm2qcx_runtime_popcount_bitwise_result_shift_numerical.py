#!/usr/bin/env python3
"""Audit shifted runtime-range bitwise population counts (#505).

Run with --bra /path/to/bra, including a UBSan build. The reference applies
each operation forward to individual little-endian bits, including zero fill.
"""

import argparse
import cmath
import itertools
import math
import pathlib
import random
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_runtime_popcount_bitwise_complement_numerical as binary


def shifted_expression(left, right, operation, depths, chain):
    value = binary.expression(left, right, operation, depths)
    for direction, distance, depth in chain:
        value = '~' * depth + f'(({value}) {direction} ({distance}))'
    return value


def reference(left, right, operation, depths, chain):
    a, b, outer = depths
    combine = {'&': int.__and__, '|': int.__or__, '^': int.__xor__}[operation]
    bits = binary.complemented([combine(x, y) for x, y in zip(
        binary.complemented(left, a), binary.complemented(right, b))], outer)
    for direction, distance, depth in chain:
        take = min(distance, len(bits))
        bits = ([0] * take + bits[:len(bits) - take] if direction == '<<'
                else bits[take:] + [0] * take)
        bits = binary.complemented(bits, depth)
    return sum(bits)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    comparisons = 0
    setUp = binary.BackendTests.setUp
    execute = binary.BackendTests.execute
    program = binary.BackendTests.program
    check_exact = binary.BackendTests.check_exact

    def test_documented_shifted_bitwise_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-popcount-bitwise-result-shift-example -->'
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        source = source.replace('OPENQASM 3.0;', '', 1)
        self.check_exact(source, tuple((name, None) for name in
            ('left', 'right', 'outer', 'before', 'mixed', 'operands', 'fixed', 'reversed',
             'united', 'filled', 'literal', 'repeated', 'wide', 'skipped', 'fresh')),
            (1, 1, 3, 2, 1, 2, 1, 1, 3, 3, 0, 3, 64, 0, 1))

    def check_case(self, left, right, constants, depths, chains):
        # Keep independent cases below the converter's per-program output limit,
        # including the fixed-width path's unrolled operations on 129-bit inputs.
        if len(chains) > 6:
            for first in range(0, len(chains), 6):
                self.check_case(left, right, constants, depths, chains[first:first + 6])
            return
        sa, pa, ba = left
        sb, pb, bb = right
        av, bv = binary.selected(*left), binary.selected(*right)
        self.assertEqual(len(av), len(bv))
        width = len(av)
        mask = sum(bit << i for i, bit in enumerate(bv))
        source = [('const ' if constants[0] else '') + f'bit[{sa}] a="{pa:0{sa}b}"; ',
                  ('const ' if constants[1] else '') + f'bit[{sb}] b="{pb:0{sb}b}"; ',
                  ('const ' if constants[1] else '') + f'bit[{width}] mask="{mask:0{width}b}"; ']
        for name, value in zip(('first', 'stride', 'last', 'start', 'step', 'stop'), (*ba, *bb)):
            source.append(f'int {name}={value}; ')
        forms = (('a[first:stride:last]', 'b[start:step:stop]', av, bv),
                 ('a[first:stride:last]', 'mask', av, bv), ('mask', 'a[first:stride:last]', bv, av))
        outputs, expected = [], []
        for chain_index, chain in enumerate(chains):
            tokens = []
            for index, (direction, distance, depth) in enumerate(chain):
                name = f'distance{chain_index}_{index}'
                kind = 'uint' if distance > 2147483647 or index % 2 else 'int'
                if distance > 4294967295:
                    name = str(distance)
                else:
                    source.append(f'{kind} {name}={distance}; ')
                tokens.append((direction, name, depth))
            for operation, groups, (lhs, rhs, lv, rv) in itertools.product(('&', '|', '^'), depths, forms):
                name = f'out{len(outputs)}'
                value = shifted_expression(lhs, rhs, operation, groups, tokens)
                source.append(f'uint {name}=popcount({value}); ')
                outputs.append((name, None))
                expected.append(reference(lv, rv, operation, groups, chain))
        source.append(f'bool same=a=="{pa:0{sa}b}" && b=="{pb:0{sb}b}" && mask=="{mask:0{width}b}"; '
                      'uint fresh=popcount(a[first:stride:last]);')
        self.check_exact(''.join(source), (*outputs, ('same', None), ('fresh', None)), (*expected, 1, sum(av)))
        type(self).comparisons += len(expected)

    def test_exhaustive_small_patterns_and_complement_parities(self):
        for size in (1, 2):
            chains = tuple((('<<', d, outer), ('>>', 1, 1))
                           for d, outer in itertools.product(range(size + 2), (0, 1)))
            for a, b in itertools.product(range(1 << size), repeat=2):
                self.check_case((size, a, (0, 1, size - 1)), (size, b, (size - 1, -1, 0)),
                                (bool(a % 2), bool(b % 2)), tuple(itertools.product((0, 1), repeat=3)), chains)

    def test_width_boundaries_huge_distances_and_long_mixed_chains(self):
        rng = random.Random(505)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (*range(1, native + 2), 65, 129):
            chains = [((direction, distance, outer),)
                      for direction, distance, outer in itertools.product(
                          ('<<', '>>'), sorted({0, 1, width - 1, width, width + 1,
                                              2147483647, 4294967295, 10**40}), (0, 1))]
            chains += [tuple((rng.choice(('<<', '>>')), rng.randrange(width + 2), rng.randrange(4))
                             for _ in range(6)) for _ in range(3)]
            self.check_case((width, rng.getrandbits(width), (0, 1, width - 1)),
                            (width, rng.getrandbits(width), (width - 1, -1, 0)),
                            (bool(width % 2), bool(width // 2 % 2)), ((2, 3, 2), (3, 2, 3)), chains)

    def test_negative_aliases_strides_and_different_containing_widths(self):
        for sa, ba, sb, bb in ((8, (-1, -2, -7), 129, (119, 3, 128)),
                               (4, (-4, 1, 3), 8, (7, -1, 0)),
                               (1, (-1, 1, 0), 129, (128, -128, 0)),
                               (129, (128, -3, 119), 8, (0, 2, 6)),
                               (129, (2, 4, 5), 1, (0, 1, 0))):
            for constants in itertools.product((False, True), repeat=2):
                self.check_case((sa, int(('11010011' * sa)[:sa], 2), ba),
                                (sb, int(('10110110' * sb)[:sb], 2), bb), constants,
                                ((0, 1, 2), (1, 0, 3)),
                                ((('<<', 1, 1), ('>>', 2, 2), ('<<', 0, 3)),
                                 (('>>', 999, 1), ('<<', 1, 0))))

    def test_computed_fixed_operands_nested_counts_and_consumers(self):
        av = [1, 1, 0, 1]
        operands = (('rotl(~mask,-1)', [1, 0, 0, 0]), ('(~mask)<<1', [0, 0, 1, 0]),
                    ('mask ^ "1010"', [1, 1, 1, 0]), ('mask[{index,0,1,2}]', [1, 1, 0, 1]),
                    ('bit[4](uint[4](mask))', [1, 0, 1, 1]))
        source = ['bit[4] a="1011"; bit[4] mask="1101"; int first=0; int last=3; int index=3; ']
        outputs, expected = [], []
        distance = 'popcount(a[first:last] & mask[first:last])'
        for operation, (fixed, fv), reverse in itertools.product(('&', '|', '^'), operands, (False, True)):
            lhs, rhs, lv, rv = 'a[first:last]', fixed, av, fv
            if reverse:
                lhs, rhs, lv, rv = rhs, lhs, rv, lv
            name = f'out{len(outputs)}'
            expr = shifted_expression(lhs, rhs, operation, (1, 2, 3), (('<<', distance, 1), ('>>', '1', 2)))
            source.append(f'uint {name}=popcount({expr}); ')
            outputs.append((name, None))
            expected.append(reference(lv, rv, operation, (1, 2, 3), (('<<', 2, 1), ('>>', 1, 2))))
        self.check_exact(''.join(source), outputs, expected)
        type(self).comparisons += len(expected)
        self.check_exact('bit[4] a="1011"; const bit[4] B="1101"; int first=0; int last=3; '
            'uint result=1; result=popcount(~((a[first:last] & B) << result)); '
            'array[uint,4] values={0,0,0,0}; values[result]=result; uint stored=values[3]; '
            'uint[1] narrow=popcount(~((a[first:last] & B) << 1)); '
            'bit[2] packed=bit[2](uint[2](popcount(~((a[first:last] & B) << 1)))); '
            'uint nested=popcount(uint[4](popcount(~((a[first:last] & B) << 1))));',
            (('result', None), ('stored', None), ('narrow', None), ('packed', 2), ('nested', None)),
            (3, 3, 1, 3, 2))

    def test_loop_reentry_shrinking_selections_and_transfers(self):
        forms = (('a[first:last]', 'b[start:-1:stop]'), ('a[first:last]', 'mask'), ('mask', 'a[first:last]'))
        for left, right in forms:
            expr = shifted_expression(left, right, '^', (1, 1, 1), (('<<', 'first', 1), ('>>', '1', 0)))
            counts = [reference(binary.selected(8, 0xB6, (i, 1, i + 3)),
                                binary.selected(8, 0xD3, (7 - i, -1, 4 - i)) if right != 'mask' and left != 'mask'
                                else [1, 0, 1, 1], '^', (1, 1, 1), (('<<', i, 1), ('>>', 1, 0)))
                      for i in range(4)]
            self.check_exact('bit[8] a="10110110"; const bit[8] b="11010011"; bit[4] mask="1101"; '
                'int first=0; int last=3; int start=7; int stop=4; uint total=0; uint count=0; '
                f'while (first<5) {{ last=first+3; start=7-first; stop=4-first; count=popcount({expr}); '
                'first+=1; if (first==2) { continue; } total+=count; if (first==4) { break; } }',
                (('total', None), ('count', None), ('first', None)),
                (sum(counts[i] for i in (0, 2, 3)), counts[3], 4))
        expected = sum(reference(binary.selected(4, 11, (0, 1, stop)), binary.selected(4, 13, (0, 1, stop)),
                                 '|', (0, 1, 2), (('>>', 1, 1),)) for stop in (3, 2, 1, 0))
        self.check_exact('bit[4] a="1011"; bit[4] b="1101"; int first=0; int stop=3; uint total=0; '
            'while (stop>=0) { total+=popcount(~((a[first:stop] | ~b[first:stop]) >> 1)); stop-=1; }',
            (('total', None),), (expected,))

    def test_skipped_paths_and_short_circuiting(self):
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        base = ('bit[4] a="1011"; const bit[4] B="1101"; int first=0; int last=3; '
                'int zero=0; uint huge=uint(-1); uint result=7; ')
        for expr in ('(a[first:0:999] & B) << (1/zero)',
                     '~(((a[first:last] ^ B[first:last]) >> huge) << -1)',
                     '(a[first:last] | B) >> (1/zero)'):
            for wrapper in wrappers:
                self.check_exact(base + wrapper.format(f'result=popcount({expr});'), (('result', None),), (7,))
            self.check_exact(base + f'bool x=false && bool(popcount({expr})); bool y=true || bool(popcount({expr}));',
                             (('x', None), ('y', None)), (0, 1))

    def test_reached_errors_preserve_order_and_suppress_destination_write(self):
        cases = (('(a[first:0:last] & B[start:stop]) << (1/zero)', 'assertion failed', False, 1000000),
                 ('(a[first:last] ^ B[start:1]) << (1/zero)', 'assertion failed', True, 1000000),
                 ('((a[first:last] | B[start:stop]) << huge) >> (1/zero)', 'division by zero', True, 1000000),
                 ('((a[first:last] | B[start:stop]) << huge) >> -1', 'assertion failed', True, 1000000),
                 ('(a[first:last] & mask) >> -1', 'assertion failed', False, 1000000),
                 ('(a[first:last] & ("11" << (1/zero))) >> -1', 'assertion failed', False, 1000000),
                 ('(("11" << (1/zero)) & a[first:last]) >> -1', 'division by zero', False, 1000000),
                 ('(a[first:last] & B[start/zero:stop]) >> -1', 'division by zero', True, 1000000),
                 ('(a[first:2] & B[start/zero:stop]) << -1', 'assertion failed', False, 2))
        cls = self.module.QASM2QCXConverter
        for expr, error, left_done, capacity in cases:
            source = ('bit[4] a="1011"; const bit[8] B="11011101"; bit[4] mask="1101"; '
                      'int first=0; int last=3; int start=0; int stop=3; int zero=0; '
                      f'uint huge=uint(-1); uint result=6; result=popcount({expr});')
            ast = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source)
            with patch.object(cls, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', capacity):
                converter = cls(ast); converter.visit(ast)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
            destination = converter._QASM2QCXConverter__capitalize_variable_name('result')
            lines = []
            for line in converter:
                if line.startswith(f'LET {destination} := QASM2QCX_'):
                    lines.append('PRINTLN 11111')
                lines.append(line)
                if line == '@QASM2QCX_POPCOUNT_0_LEFT_SNAPSHOT_DONE':
                    lines.append('PRINTLN 22222')
            result = self.execute(lines)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual(result.stdout.split(), ['22222'] if left_done else [])
            type(self).checked += 1

    def test_measurement_feedback_and_normal_fused_gate_parameters(self):
        for preparation in ('', 'x q[0]; ', 'h q[0]; ', 'h q[0]; x q[0]; '):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; ' + preparation +
                'bit[2] measured=measure q; const bit[2] MASK="01"; int first=0; int last=1; '
                'uint count=popcount(~((measured[first:last] & MASK) << 1)); '
                'bool valid=count==uint(2)-uint(measured[0]); '
                'if (count==uint(1)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0];',
                (('valid', None), ('feedback', None)), (1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        forms = (('a[first:last]', 'b[first:last]'), ('a[first:last]', 'b'), ('b', 'a[first:last]'))
        for operation, (left, right), fused in itertools.product(('&', '|', '^'), forms, (False, True)):
            av, bv = [1, 1, 0, 1], [1, 0, 1, 1]
            if left == 'b':
                av, bv = bv, av
            chain = (('<<', 1, 1), ('>>', 1, 0))
            count = reference(av, bv, operation, (1, 0, 1), chain)
            phase = cmath.exp(count * 1j * math.pi / 8)
            expr = shifted_expression(left, right, operation, (1, 0, 1), chain)
            for gate in (f'p(float(popcount({expr}))*pi/8) q;', f'pow(popcount({expr})) @ p(pi/8) q;'):
                source = ('include "stdgates.inc"; qubit[2] q; bit[4] a="1011"; bit[4] b="1101"; '
                          'int first=0; int last=3; h q; ' + gate)
                BackendAuditTests.check(self, self.program(source, fused=fused),
                                        (0.5, phase/2, phase/2, phase*phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'shifted runtime-range bitwise popcount programs checked: {BackendTests.checked}; '
          f'bit-vector comparisons: {BackendTests.comparisons}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
