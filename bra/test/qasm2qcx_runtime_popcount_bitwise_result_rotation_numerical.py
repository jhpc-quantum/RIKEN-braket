#!/usr/bin/env python3
"""Audit outer rotations on runtime-range bitwise population counts (#507).

Run with --bra /path/to/bra, including a UBSan build. The reference rotates,
shifts, and complements individual little-endian bits in forward order.
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


def expression(left, right, operation, depths, chain):
    value = binary.expression(left, right, operation, depths)
    for name, distance, depth in chain:
        value = (f'{name}({value}, {distance})' if name in ('rotl', 'rotr')
                 else f'({value}) {name} ({distance})')
        value = '~' * depth + f'({value})'
    return value


def reference(left, right, operation, depths, chain):
    assert len(left) == len(right) and left
    a, b, outer = depths
    combine = {'&': int.__and__, '|': int.__or__, '^': int.__xor__}[operation]
    bits = binary.complemented([combine(x, y) for x, y in zip(
        binary.complemented(left, a), binary.complemented(right, b))], outer)
    for name, distance, depth in chain:
        if name in ('rotl', 'rotr'):
            count = (distance if name == 'rotl' else -distance) % len(bits)
            bits = bits[len(bits) - count:] + bits[:len(bits) - count]
        else:
            count = min(distance, len(bits))
            bits = ([0] * count + bits[:len(bits) - count] if name == '<<'
                    else bits[count:] + [0] * count)
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

    def test_documented_rotated_bitwise_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-popcount-bitwise-result-rotation-example -->'
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        source = source.replace('OPENQASM 3.0;', '', 1)
        self.check_exact(source, tuple((name, None) for name in
            ('left', 'right', 'outer', 'before', 'after', 'mixed', 'fixed', 'reversed',
             'united', 'endpoint', 'promoted', 'filled', 'repeated', 'wide', 'skipped', 'fresh')),
            (2, 2, 2, 2, 1, 2, 1, 2, 3, 2, 2, 3, 3, 64, 0, 1))

    def check_case(self, left, right, constants, depths, chains):
        # Batch independent cases below the per-program output limit, including
        # fixed-width operands and constant lookups for wide registers.
        if len(chains) > 3:
            for first in range(0, len(chains), 3):
                self.check_case(left, right, constants, depths, chains[first:first + 3])
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
            for index, (name, distance, depth) in enumerate(chain):
                token = f'distance{chain_index}_{index}'
                kind = 'uint' if distance >= 0 and index % 2 or distance > 2147483647 else 'int'
                if distance > 4294967295:
                    token = str(distance)
                else:
                    source.append(f'{kind} {token}={distance}; ')
                tokens.append((name, token, depth))
            for operation, groups, (lhs, rhs, lv, rv) in itertools.product(('&', '|', '^'), depths, forms):
                name = f'out{len(outputs)}'
                source.append(f'uint {name}=popcount({expression(lhs, rhs, operation, groups, tokens)}); ')
                outputs.append((name, None))
                expected.append(reference(lv, rv, operation, groups, chain))
        source.append(f'bool same=a=="{pa:0{sa}b}" && b=="{pb:0{sb}b}" && mask=="{mask:0{width}b}"; '
                      'uint fresh=popcount(a[first:stride:last]);')
        self.check_exact(''.join(source), (*outputs, ('same', None), ('fresh', None)), (*expected, 1, sum(av)))
        type(self).comparisons += len(expected)

    def test_exhaustive_small_patterns_operators_and_complement_parities(self):
        chains = ((('rotl', 1, 0),), (('rotr', -1, 1),),
                  (('rotl', 1, 1), ('<<', 1, 1)), (('>>', 1, 1), ('rotr', 1, 0)),
                  (('rotl', -1, 2), ('<<', 1, 3), ('rotr', -1, 1)),
                  (('rotr', 1, 3), ('>>', 0, 2), ('rotl', 1, 0), ('<<', 1, 1)))
        for size in (1, 2):
            for a, b in itertools.product(range(1 << size), repeat=2):
                self.check_case((size, a, (0, 1, size - 1)), (size, b, (size - 1, -1, 0)),
                                (bool(a % 2), bool(b % 2)), tuple(itertools.product((0, 1), repeat=3)), chains)

    def test_native_widths_distance_boundaries_and_long_mixed_chains(self):
        rng = random.Random(507)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (*range(1, native + 2), 65, 129):
            chains = [(('rotl', -1, 1), ('<<', 1, 0), ('rotr', 1, 1)),
                      (('>>', width, 1), ('rotr', -2147483648, 1))]
            if width in (1, 2, native - 1, native, native + 1, 65, 129):
                chains += [((name, distance, 2), ('<<', 1, 1), ('rotr', -1, 3), ('>>', 1, 0))
                           for name, distance in itertools.product(('rotl', 'rotr'),
                               sorted({-2147483648, -width - 1, -width, -1, 0, 1, width, width + 1, 2147483647}))]
                chains += [(('<<', distance, 1), ('rotl', -1, 2), ('>>', 1, 1))
                           for distance in (width + 1, 4294967295, 10**40)]
                for _ in range(3):
                    chain = [('rotl', rng.randrange(-width, width + 1), 1)]
                    for _ in range(7):
                        name = rng.choice(('rotl', 'rotr', '<<', '>>'))
                        distance = rng.randrange(-2, 3) if name in ('rotl', 'rotr') else rng.randrange(3)
                        chain.append((name, distance, rng.randrange(4)))
                    chains.append(tuple(chain))
            self.check_case((width, rng.getrandbits(width), (0, 1, width - 1)),
                            (width, rng.getrandbits(width), (width - 1, -1, 0)),
                            (bool(width % 2), bool(width // 2 % 2)), ((2, 3, 2), (3, 2, 3)), chains)

    def test_aliases_strides_singletons_and_source_kinds(self):
        for sa, ba, sb, bb in ((8, (-1, -2, -7), 129, (119, 3, 128)),
                               (4, (-4, 1, 3), 8, (7, -1, 0)),
                               (1, (-1, 1, 0), 129, (128, -128, 0)),
                               (129, (128, -3, 119), 8, (0, 2, 6)),
                               (129, (2, 4, 5), 1, (0, 1, 0))):
            for constants in itertools.product((False, True), repeat=2):
                self.check_case((sa, int(('11010011' * sa)[:sa], 2), ba),
                                (sb, int(('10110110' * sb)[:sb], 2), bb), constants,
                                ((0, 1, 2), (1, 0, 3)),
                                ((('rotl', -1, 1), ('<<', 1, 2), ('rotr', 1, 3)),
                                 (('>>', 999, 1), ('rotl', 1, 0))))

    def test_computed_fixed_operands_nested_distances_and_promotions(self):
        av = [1, 1, 0, 1]
        operands = (('rotl(~mask,-1)', [1, 0, 0, 0]), ('(~mask)<<1', [0, 0, 1, 0]),
                    ('mask ^ "1010"', [1, 1, 1, 0]), ('mask[{index,0,1,2}]', [1, 1, 0, 1]),
                    ('bit[4](uint[4](mask))', [1, 0, 1, 1]), ('bit[4](angle[4](mask))', [1, 0, 1, 1]))
        source = ['bit[4] a="1011"; bit[4] mask="1101"; int first=0; int last=3; int index=3; bool distance=true; ']
        outputs, expected = [], []
        nested = 'popcount(a[first:last] & mask[first:last])'
        for operation, (fixed, fv), reverse in itertools.product(('&', '|', '^'), operands, (False, True)):
            lhs, rhs, lv, rv = 'a[first:last]', fixed, av, fv
            if reverse:
                lhs, rhs, lv, rv = rhs, lhs, rv, lv
            name = f'out{len(outputs)}'
            value = expression(lhs, rhs, operation, (1, 2, 3),
                               (('rotl', nested, 1), ('>>', '1', 2), ('rotr', 'distance', 1)))
            source.append(f'uint {name}=popcount({value}); ')
            outputs.append((name, None))
            expected.append(reference(lv, rv, operation, (1, 2, 3), (('rotl', 2, 1), ('>>', 1, 2), ('rotr', 1, 1))))
        self.check_exact(''.join(source), outputs, expected)
        type(self).comparisons += len(expected)

    def test_nested_consumers_self_assignment_and_loop_reentry(self):
        self.check_exact('bit[4] a="1011"; const bit[4] B="1101"; int first=0; int last=3; '
            'uint result=1; result=popcount(~(rotl(a[first:last] & B,result) >> 1)); '
            'array[uint,4] values={0,0,0,0}; values[result]=result; uint stored=values[3]; '
            'uint[1] narrow=popcount(~(rotl(a[first:last] & B,1) >> 1)); '
            'bit[2] packed=bit[2](uint[2](popcount(~(rotl(a[first:last] & B,1) >> 1)))); '
            'uint nested=popcount(uint[4](popcount(~(rotl(a[first:last] & B,1) >> 1))));',
            (('result', None), ('stored', None), ('narrow', None), ('packed', 2), ('nested', None)), (3, 3, 1, 3, 2))
        for ca, cb in itertools.product((False, True), repeat=2):
            prefix = ('const ' if ca else '') + 'bit[4] a="1011"; '
            prefix += ('const ' if cb else '') + 'bit[4] b="1101"; '
            counts = [reference(binary.selected(4, 11, (0, 1, stop)), binary.selected(4, 13, (stop, -1, 0)),
                                '^', (1, 0, 0), (('rotl', stop, 1), ('>>', 1, 0), ('rotr', -1, 1)))
                      for stop in (3, 2, 1, 0)]
            self.check_exact(prefix + 'int first=0; int stop=3; uint total=0; uint count=0; '
                'while (stop>=0) { count=popcount(~rotr((~rotl((~a[first:stop]) ^ b[stop:-1:first],stop)) >> 1,-1)); '
                'stop-=1; if (stop==1) { continue; } total+=count; if (stop==-1) { break; } }',
                (('total', None), ('count', None)), (sum(counts[i] for i in (0, 2, 3)), counts[3]))
        self.check_exact('bit[4] a="1111"; bit[4] b="1111"; int first=0; int last=3; int visits=0; '
            'while (popcount(rotl(a[first:last] & b,1) >> 1)>uint(0)) { a>>=1; visits+=1; }',
            (('visits', None), ('a', 4)), (4, 0))

    def test_skipped_paths_and_short_circuiting(self):
        base = ('bit[4] a="1011"; const bit[4] B="1101"; int first=0; int last=3; '
                'int zero=0; uint huge=uint(-1); uint result=7; ')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        for value in ('rotl(a[first:0:999] & B,1/zero)',
                      'rotr((a[first:last] ^ B[first:last]) << huge,huge)',
                      'rotl(a[first:last] | B,-2147483648) >> -1'):
            for wrapper in wrappers:
                self.check_exact(base + wrapper.format(f'result=popcount({value});'), (('result', None),), (7,))
            self.check_exact(base + f'bool x=false && bool(popcount({value})); bool y=true || bool(popcount({value}));',
                             (('x', None), ('y', None)), (0, 1))

    def test_reached_errors_order_and_destination_write_suppression(self):
        conversion = 'UINT to INT conversion out of range'
        cases = (('rotl(a[first:0:last] & B[start:stop],1/zero)', 'assertion failed', False, 1000000),
                 ('rotl(a[first:last] & B[start:1],1/zero)', 'assertion failed', True, 1000000),
                 ('rotl((a[first:last] | B[start:stop]) << huge,1/zero)', 'division by zero', True, 1000000),
                 ('rotr((a[first:last] | B[start:stop]) << huge,huge)', conversion, True, 1000000),
                 ('rotl(a[first:last] & mask,huge)', conversion, False, 1000000),
                 ('rotr(mask & a[first:last],huge)', conversion, False, 1000000),
                 ('rotl((a[first:last] & mask) >> -1,1/zero)', 'assertion failed', False, 1000000),
                 ('rotl((a[first:last] & mask) << huge,1) >> -1', 'assertion failed', False, 1000000),
                 ('rotr(rotl(a[first:last] & mask,huge),1/zero)', conversion, False, 1000000),
                 ('rotl(a[first:2] & B[start/zero:stop],1/zero)', 'assertion failed', False, 2),
                 ('rotl(a[first:last] & ("11" << (1/zero)),huge)', 'assertion failed', False, 1000000),
                 ('rotr(("11" << (1/zero)) & a[first:last],huge)', 'division by zero', False, 1000000))
        cls = self.module.QASM2QCXConverter
        for value, error, left_done, capacity in cases:
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
                'uint count=popcount(~(rotl(measured[first:last] & MASK,1) >> 1)); '
                'bool valid=count==uint(2)-uint(measured[0]); '
                'if (count==uint(1)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0];',
                (('valid', None), ('feedback', None)), (1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        forms = (('a[first:last]', 'b[first:last]'), ('a[first:last]', 'b'), ('b', 'a[first:last]'))
        chain = (('rotl', -1, 1), ('<<', 1, 0), ('rotr', 1, 1), ('>>', 1, 0))
        for operation, (left, right), fused in itertools.product(('&', '|', '^'), forms, (False, True)):
            av, bv = [1, 1, 0, 1], [1, 0, 1, 1]
            if left == 'b':
                av, bv = bv, av
            count = reference(av, bv, operation, (1, 0, 1), chain)
            phase = cmath.exp(count * 1j * math.pi / 8)
            value = expression(left, right, operation, (1, 0, 1), chain)
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
    print(f'rotated runtime-range bitwise popcount programs checked: {BackendTests.checked}; '
          f'bit-vector comparisons: {BackendTests.comparisons}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
