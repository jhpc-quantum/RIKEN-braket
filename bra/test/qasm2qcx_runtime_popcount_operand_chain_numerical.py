#!/usr/bin/env python3
"""Audit independent runtime-range popcount operand chains (#509).

Run with --bra /path/to/bra, including a UBSan build. The reference applies
each operand's operations forward to individual bits, combines the vectors,
then applies the outer chain. Independent cases use small program batches.
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


def chain_expression(value, chain):
    for operation, distance, depth in chain:
        value = (f'{operation}({value},{distance})' if operation in ('rotl', 'rotr')
                 else f'({value}) {operation} ({distance})')
        value = '~' * depth + f'({value})'
    return value


def transform(bits, chain):
    value = list(bits)
    for operation, distance, depth in chain:
        if operation in ('rotl', 'rotr'):
            count = (distance if operation == 'rotl' else -distance) % len(value)
            value = value[len(value) - count:] + value[:len(value) - count]
        else:
            count = min(distance, len(value))
            value = ([0] * count + value[:len(value) - count] if operation == '<<'
                     else value[count:] + [0] * count)
        value = binary.complemented(value, depth)
    return value


def expression(left, right, operation, depths, chains):
    a, b, outer = depths
    left = chain_expression('~' * a + f'({left})', chains[0])
    right = chain_expression('~' * b + f'({right})', chains[1])
    return chain_expression('~' * outer + f'(({left}) {operation} ({right}))', chains[2])


def reference(left, right, operation, depths, chains):
    assert len(left) == len(right) and left
    a, b, outer = depths
    left = transform(binary.complemented(left, a), chains[0])
    right = transform(binary.complemented(right, b), chains[1])
    combine = {'&': int.__and__, '|': int.__or__, '^': int.__xor__}[operation]
    bits = binary.complemented([combine(x, y) for x, y in zip(left, right)], outer)
    return sum(transform(bits, chains[2]))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    comparisons = 0
    setUp = binary.BackendTests.setUp
    execute = binary.BackendTests.execute
    program = binary.BackendTests.program
    check_exact = binary.BackendTests.check_exact

    def check_case(self, left, right, constants, depths, scenarios):
        # Bound generated programs, including fixed-width unrolling and wide
        # constant lookups, without reducing the combinations being checked.
        if len(scenarios) > 2:
            for first in range(0, len(scenarios), 2):
                self.check_case(left, right, constants, depths, scenarios[first:first + 2])
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
        outputs, expected = [], []
        for scenario_index, chains in enumerate(scenarios):
            tokens = []
            for side, chain in enumerate(chains):
                entries = []
                for index, (operation, distance, depth) in enumerate(chain):
                    token = f'distance{scenario_index}_{side}_{index}'
                    if distance > 4294967295:
                        token = str(distance)
                    else:
                        kind = 'uint' if (distance >= 0 and index % 2) or distance > 2147483647 else 'int'
                        source.append(f'{kind} {token}={distance}; ')
                    entries.append((operation, token, depth))
                tokens.append(tuple(entries))
            forms = (('a[first:stride:last]', 'b[start:step:stop]', av, bv, False),
                     ('a[first:stride:last]', 'mask', av, bv, False),
                     ('mask', 'a[first:stride:last]', bv, av, True))
            for operation, groups, (lhs, rhs, lv, rv, reverse) in itertools.product(('&', '|', '^'), depths, forms):
                active = (chains[1], chains[0], chains[2]) if reverse else chains
                arguments = (tokens[1], tokens[0], tokens[2]) if reverse else tokens
                name = f'out{len(outputs)}'
                source.append(f'uint {name}=popcount({expression(lhs, rhs, operation, groups, arguments)}); ')
                outputs.append((name, None))
                expected.append(reference(lv, rv, operation, groups, active))
        source.append(f'bool same=a=="{pa:0{sa}b}" && b=="{pb:0{sb}b}" && mask=="{mask:0{width}b}"; '
                      'uint fresh=popcount(a[first:stride:last]);')
        self.check_exact(''.join(source), (*outputs, ('same', None), ('fresh', None)), (*expected, 1, sum(av)))
        type(self).comparisons += len(expected)

    def test_exhaustive_small_patterns_operators_and_complement_parities(self):
        scenarios = (((('<<', 1, 0),), (), ()),
                     ((('rotl', 1, 0),), (('>>', 1, 1),), ()),
                     ((('rotr', -1, 1), ('<<', 1, 2)), (('rotl', 1, 1),), (('>>', 1, 1),)),
                     ((('<<', 2, 1), ('rotr', -2147483648, 1)), (('>>', 2, 1),), (('rotl', -1, 1),)))
        for size in (1, 2):
            for a, b in itertools.product(range(1 << size), repeat=2):
                self.check_case((size, a, (0, 1, size - 1)), (size, b, (size - 1, -1, 0)),
                                (bool(a % 2), bool(b % 2)), tuple(itertools.product((0, 1), repeat=3)), scenarios)

    def test_widths_distance_boundaries_and_long_independent_chains(self):
        rng = random.Random(509)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (*range(1, native + 2), 65, 129):
            scenarios = [((('rotl', -1, 1), ('<<', 1, 0)), (('rotr', -2147483648, 1),), (('>>', 1, 1),)),
                         ((('>>', width, 1),), (('<<', width, 1),), (('rotl', -1, 0),))]
            if width in (1, 2, native - 1, native, native + 1, 65, 129):
                scenarios += [(((name, distance, 2), ('<<', 1, 1)), (('rotr', -1, 3),), (('>>', 1, 0),))
                              for name, distance in itertools.product(('rotl', 'rotr'),
                                  sorted({-2147483648, -width - 1, -1, 0, 1, width, width + 1, 2147483647}))]
                scenarios += [((('<<', distance, 1),), (('>>', distance, 1),), (('rotr', -1, 1),))
                              for distance in (width + 1, 4294967295, 10**40)]
                for _ in range(3):
                    chains = []
                    for _ in range(3):
                        chain = []
                        for _ in range(5):
                            name = rng.choice(('rotl', 'rotr', '<<', '>>'))
                            distance = rng.randrange(-2, 3) if name in ('rotl', 'rotr') else rng.randrange(3)
                            chain.append((name, distance, rng.randrange(4)))
                        chains.append(tuple(chain))
                    scenarios.append(tuple(chains))
            self.check_case((width, rng.getrandbits(width), (0, 1, width - 1)),
                            (width, rng.getrandbits(width), (width - 1, -1, 0)),
                            (bool(width % 2), bool(width // 2 % 2)), ((2, 3, 2), (3, 2, 3)), scenarios)

    def test_aliases_strides_singletons_source_kinds_and_different_widths(self):
        for sa, ba, sb, bb in ((8, (-1, -2, -7), 129, (119, 3, 128)),
                               (4, (-4, 1, 3), 8, (7, -1, 0)),
                               (1, (-1, 1, 0), 129, (128, -128, 0)),
                               (129, (128, -3, 119), 8, (0, 2, 6)),
                               (129, (2, 4, 5), 1, (0, 1, 0))):
            for constants in itertools.product((False, True), repeat=2):
                self.check_case((sa, int(('11010011' * sa)[:sa], 2), ba),
                                (sb, int(('10110110' * sb)[:sb], 2), bb), constants, ((0, 1, 2), (1, 0, 3)),
                                (((('rotl', -1, 1), ('<<', 1, 2)), (('>>', 1, 1),), (('rotr', 1, 3),)),
                                 ((('>>', 999, 1),), (('<<', 1, 1),), ())))

    def test_computed_fixed_operands_nested_distances_and_bool_promotions(self):
        av = [1, 1, 0, 1]
        operands = (('rotl(~mask,-1)', [1, 0, 0, 0]), ('(~mask)<<1', [0, 0, 1, 0]),
                    ('mask ^ "1010"', [1, 1, 1, 0]), ('mask[{index,0,1,2}]', [1, 1, 0, 1]),
                    ('bit[4](uint[4](mask))', [1, 0, 1, 1]), ('bit[4](angle[4](mask))', [1, 0, 1, 1]))
        source = ['bit[4] a="1011"; bit[4] mask="1101"; int first=0; int last=3; int index=3; bool distance=true; ']
        outputs, expected = [], []
        nested = 'popcount((mask[first:last] << 1) & a[first:last])'
        nested_count = reference([1, 0, 1, 1], av, '&', (0, 0, 0), ((( '<<', 1, 0),), (), ()))
        runtime_tokens = (('rotl', nested, 1), ('>>', '1', 2), ('rotr', 'distance', 1))
        runtime_chain = (('rotl', nested_count, 1), ('>>', 1, 2), ('rotr', 1, 1))
        for operation, (fixed, fv), reverse in itertools.product(('&', '|', '^'), operands, (False, True)):
            lhs, rhs, lv, rv = 'a[first:last]', fixed, av, fv
            tokens, chains = (runtime_tokens, (), (('rotl', '1', 1),)), (runtime_chain, (), (('rotl', 1, 1),))
            if reverse:
                lhs, rhs, lv, rv = rhs, lhs, rv, lv
                tokens, chains = (tokens[1], tokens[0], tokens[2]), (chains[1], chains[0], chains[2])
            name = f'out{len(outputs)}'
            source.append(f'uint {name}=popcount({expression(lhs, rhs, operation, (1, 2, 3), tokens)}); ')
            outputs.append((name, None))
            expected.append(reference(lv, rv, operation, (1, 2, 3), chains))
        self.check_exact(''.join(source), outputs, expected)
        type(self).comparisons += len(expected)
        self.check_exact('bit[4] a="1011"; bit[4] b="1101"; int first=0; int last=3; bool distance=false; '
            'uint out=popcount(rotl(a[first:last],distance) ^ rotr(b[first:last],distance));',
            (('out', None),), (2,))

    def test_nested_consumers_self_assignment_and_loop_reentry(self):
        self.check_exact('bit[4] a="1011"; const bit[4] B="1101"; int first=0; int last=3; '
            'uint result=1; result=popcount(rotl(a[first:last],result) & (B >> 1)); '
            'array[uint,4] values={0,0,0,0}; values[result]=result; uint stored=values[2]; '
            'uint[1] narrow=popcount(rotl(a[first:last],1) & (B >> 1)); '
            'bit[2] packed=bit[2](uint[2](popcount(rotl(a[first:last],1) & (B >> 1)))); '
            'uint nested=popcount(uint[4](popcount(rotl(a[first:last],1) & (B >> 1))));',
            (('result', None), ('stored', None), ('narrow', None), ('packed', 2), ('nested', None)), (2, 2, 0, 2, 1))
        for ca, cb in itertools.product((False, True), repeat=2):
            prefix = ('const ' if ca else '') + 'bit[4] a="1011"; '
            prefix += ('const ' if cb else '') + 'bit[4] b="1101"; '
            counts = [reference(binary.selected(4, 11, (0, 1, stop)), binary.selected(4, 13, (stop, -1, 0)),
                                '^', (1, 0, 1), ((('rotl', stop, 1), ('<<', 1, 1)),
                                                (('rotr', -1, 1),), (('>>', 1, 0),)))
                      for stop in (3, 2, 1, 0)]
            self.check_exact(prefix + 'int first=0; int stop=3; uint total=0; uint count=0; '
                'while (stop>=0) { count=popcount(~((~((~rotl(~a[first:stop],stop)) << 1)) ^ '
                '(~rotr(b[stop:-1:first],-1))) >> 1); '
                'stop-=1; if (stop==1) { continue; } total+=count; if (stop==-1) { break; } }',
                (('total', None), ('count', None)), (sum(counts[i] for i in (0, 2, 3)), counts[3]))
        self.check_exact('bit[4] a="1111"; bit[4] b="1111"; int first=0; int last=3; int visits=0; '
            'while (popcount((a[first:last] >> 1) & rotl(b[first:last],1))>uint(0)) { a>>=1; visits+=1; }',
            (('visits', None), ('a', 4)), (3, 1))

    def test_skipped_domains_and_short_circuiting(self):
        base = ('bit[4] a="1011"; const bit[4] B="1101"; int first=0; int last=3; int zero=0; '
                'uint huge=uint(-1); uint result=7; int loop_first=2; int loop_last=1; ')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        for value in ('rotl(a[first:0:999],1/zero) & B',
                      '(a[first:last] << huge) ^ rotr(B[first:last],huge)',
                      'rotr(rotl(a[first:last],huge) | (B[first:last] >> -1),1/zero)'):
            for wrapper in wrappers:
                self.check_exact(base + wrapper.format(f'result=popcount({value});'), (('result', None),), (7,))
            self.check_exact(base + f'bool x=false && bool(popcount({value})); bool y=true || bool(popcount({value}));',
                             (('x', None), ('y', None)), (0, 1))

    def test_reached_errors_operand_completion_and_destination_write_suppression(self):
        conversion = 'UINT to INT conversion out of range'
        cases = (('rotl(a[first:0:last],1/zero) & B[start/zero:stop]', 'assertion failed', (), 1000000),
                 ('(a[first:last] << -1) & B[start/zero:stop]', 'assertion failed', (), 1000000),
                 ('rotl(a[first:last],huge) & B[start/zero:stop]', conversion, (), 1000000),
                 ('rotl(a[first:last] << huge,1/zero) & B[start/zero:stop]', 'division by zero', (), 1000000),
                 ('rotl(a[first:last],1) ^ rotl(B[start:1],1/zero)', 'assertion failed', ('22222',), 1000000),
                 ('rotl(a[first:2],1/zero) & mask', 'assertion failed', (), 1000000),
                 ('rotl(mask,huge) & rotl(a[first/zero:last],1)', conversion, (), 1000000),
                 ('rotl(a[first:last],1) & rotl(mask,huge)', conversion, ('22222',), 1000000),
                 ('rotr(rotl(a[first:last],1) & rotl(B[start:stop],1),1/zero)', 'division by zero', ('22222', '33333'), 1000000),
                 ('rotl(a[first:last],1) ^ (B[start:stop] >> -1)', 'assertion failed', ('22222',), 1000000),
                 ('mask ^ rotr(a[first:last] << huge,huge)', conversion, (), 1000000),
                 ('rotl(a[first:last],popcount((B[start:stop] << huge) & mask)) & (B[start:stop] << (1/zero))',
                  'division by zero', ('22222',), 1000000),
                 ('rotl(a[first:last],1/zero) ^ (B[start/zero:stop] << -1)', 'assertion failed', (), 2))
        cls = self.module.QASM2QCXConverter
        for value, error, markers, capacity in cases:
            ast = self.module.openqasm3.parser.parse('OPENQASM 3; bit[4] a="1011"; const bit[8] B="11011101"; '
                'bit[4] mask="1101"; int first=0; int last=3; int start=0; int stop=3; int zero=0; '
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
                if line == '@QASM2QCX_POPCOUNT_0_LEFT_OPERAND_DONE':
                    lines.append('PRINTLN 22222')
                if line == '@QASM2QCX_POPCOUNT_0_RIGHT_OPERAND_DONE':
                    lines.append('PRINTLN 33333')
            result = self.execute(lines)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual(result.stdout.split(), list(markers))
            type(self).checked += 1

    def test_measurement_feedback_and_normal_fused_gate_parameters(self):
        for preparation in ('', 'x q[0]; ', 'h q[0]; ', 'h q[0]; x q[0]; '):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; ' + preparation +
                'bit[2] measured=measure q; const bit[2] MASK="10"; int first=0; int last=1; '
                'uint count=popcount(rotl(measured[first:last],1) & MASK[first:last]); '
                'bool valid=count==uint(measured[0]); '
                'if (count==uint(1)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0];',
                (('valid', None), ('feedback', None)), (1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        forms = (('a[first:last]', 'b[first:last]', False), ('a[first:last]', 'b', False), ('b', 'a[first:last]', True))
        chains = ((('rotl', -1, 1), ('<<', 1, 1)), (('>>', 1, 1),), (('rotr', 1, 1), ('>>', 1, 0)))
        for operation, (left, right, reverse), fused in itertools.product(('&', '|', '^'), forms, (False, True)):
            av, bv = [1, 1, 0, 1], [1, 0, 1, 1]
            active = (chains[1], chains[0], chains[2]) if reverse else chains
            if reverse:
                av, bv = bv, av
            count = reference(av, bv, operation, (1, 0, 1), active)
            phase = cmath.exp(count * 1j * math.pi / 8)
            value = expression(left, right, operation, (1, 0, 1), active)
            for gate in (f'p(float(popcount({value}))*pi/8) q;', f'pow(popcount({value})) @ p(pi/8) q;'):
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
    print(f'runtime-range popcount operand-chain programs checked: {BackendTests.checked}; '
          f'bit-vector comparisons: {BackendTests.comparisons}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
