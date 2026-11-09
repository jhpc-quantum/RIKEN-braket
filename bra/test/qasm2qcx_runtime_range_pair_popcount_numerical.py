#!/usr/bin/env python3
"""Audit bitwise population counts over two runtime bit-register ranges (#501).

Run with --bra /path/to/bra, including a UBSan build. The reference selects
individual little-endian bits independently for each range before combining.
"""

import argparse
import cmath
import itertools
import math
import operator
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_bit_cast_numerical as casts


def selected_bits(size, pattern, bounds):
    first, step, last = bounds
    positions = list(range(first, last + (1 if step > 0 else -1), step))
    assert positions and all(-size <= index < size for index in positions)
    return [(pattern >> (index % size)) & 1 for index in positions]


def reference(left, right, operation):
    assert len(left) == len(right)
    combine = {'&': operator.and_, '|': operator.or_, '^': operator.xor}[operation]
    return sum(combine(a, b) for a, b in zip(left, right))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    comparisons = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def check_case(self, left, right, constants):
        # Each operand independently specifies its containing width and range.
        left_size, left_pattern, left_bounds = left
        right_size, right_pattern, right_bounds = right
        a = selected_bits(*left)
        b = selected_bits(*right)
        assert len(a) == len(b)
        source = [('const ' if constants[0] else '') + f'bit[{left_size}] a="{left_pattern:0{left_size}b}"; ',
                  ('const ' if constants[1] else '') + f'bit[{right_size}] b="{right_pattern:0{right_size}b}"; ']
        for name, value in zip(('first', 'stride', 'last', 'start', 'step', 'stop'), (*left_bounds, *right_bounds)):
            source.append(f'int {name}={value}; ')
        outputs, expected = [], []
        for index, (operation, reverse) in enumerate(itertools.product(('&', '|', '^'), (False, True))):
            lhs, rhs = 'a[first:stride:last]', 'b[start:step:stop]'
            if reverse:
                lhs, rhs = rhs, lhs
            name = f'out{index}'
            source.append(f'uint {name}=popcount({lhs} {operation} {rhs}); ')
            outputs.append((name, None))
            expected.append(reference(a, b, operation))
        source.append(f'bool same=a=="{left_pattern:0{left_size}b}" && b=="{right_pattern:0{right_size}b}"; '
                      'uint fresh_left=popcount(a[first:stride:last]); uint fresh_right=popcount(b[start:step:stop]);')
        self.check_exact(''.join(source), (*outputs, ('same', None), ('fresh_left', None), ('fresh_right', None)),
                         (*expected, 1, sum(a), sum(b)))
        type(self).comparisons += len(expected)

    def test_exhaustive_small_patterns_operators_orders_and_source_kinds(self):
        for size in (1, 2, 3):
            for a, b, ca, cb in itertools.product(range(1 << size), range(1 << size), (False, True), (False, True)):
                self.check_case((size, a, (0, 1, size - 1)), (size, b, (0, 1, size - 1)), (ca, cb))

    def test_every_native_width_and_wide_ranges(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (*range(1, native + 2), 65, 129):
            limit = (1 << size) - 1
            alternating = int(('10110010' * size)[:size], 2)
            for a, b in ((0, limit), (limit, 0), (alternating, limit ^ alternating)):
                for constants in itertools.product((False, True), repeat=2):
                    self.check_case((size, a, (0, 1, size - 1)), (size, b, (size - 1, -1, 0)), constants)

    def test_different_widths_nonunit_strides_negative_aliases_and_singletons(self):
        ranges = ((8, (-1, -2, -7), 129, (119, 3, 128)),
                  (4, (-4, 1, 3), 8, (7, -1, 0)),
                  (1, (-1, 1, 0), 129, (128, -128, 0)),
                  (129, (128, -3, 119), 8, (0, 2, 6)),
                  (129, (2, 4, 5), 1, (0, 1, 0)))
        for sa, ba, sb, bb in ranges:
            for constants in itertools.product((False, True), repeat=2):
                self.check_case((sa, int(('11010011' * sa)[:sa], 2), ba),
                                (sb, int(('10110110' * sb)[:sb], 2), bb), constants)
        cls = self.module.QASM2QCXConverter
        for stride in (cls.QCX_INT_MIN, cls.QCX_UINT_MAX):
            self.check_exact('bit[1] a="1"; const bit[129] B="' + '1' * 129 + '"; int first=0; '
                f'uint stop=0; {"int" if stride < 0 else "uint"} step={stride}; '
                'uint out=popcount(a[first:step:stop] ^ B[first:step:stop]);', (('out', None),), (0,))

    def test_nested_consumers_narrow_results_and_self_referencing_destinations(self):
        self.check_exact('bit[4] a="1011"; const bit[4] B="1101"; int first=0; int last=3; '
            'uint result=popcount(a[first:last] & B[first:last]); '
            'array[uint,4] values={0,0,0,0}; values[popcount(B[first:last] ^ a[first:last])]=result; '
            'uint stored=values[2]; uint[1] narrow=popcount(a[first:last] & B[first:last]); '
            'bit[2] packed=bit[2](uint[2](popcount(B[first:last] ^ a[first:last]))); '
            'uint nested=popcount(uint[4](popcount(a[first:last] & B[first:last]))); '
            'uint bounded=popcount(a[int(result):last] & B[int(result):last]); '
            'uint bound=3; bound=popcount(a[first:int(bound)] ^ B[first:int(bound)]); '
            'float root=sqrt(float(popcount(a[first:last] | B[first:last]))); bool valid=root==2.0; '
            'bool constant_ok=B=="1101";',
            (('result', None), ('stored', None), ('narrow', None), ('packed', 2), ('nested', None),
             ('bounded', None), ('bound', None), ('valid', None), ('a', 4), ('constant_ok', None)),
            (2, 2, 0, 2, 1, 1, 2, 1, 11, 1))
        self.check_exact('bit[4] a="1011"; int first=0; int last=3; '
            'uint same=popcount(a[first:last] ^ a[first:last]); '
            'uint reversed=popcount(a[first:last] & a[last:-1:first]);',
            (('same', None), ('reversed', None), ('a', 4)), (0, 2, 11))

    def test_fresh_loop_values_shrinking_lengths_and_transfers(self):
        left, right = 0xB6, 0xD3
        counts = [reference(selected_bits(8, left, (i, 1, i + 3)),
                            selected_bits(8, right, (7 - i, -1, 4 - i)), '^') for i in range(4)]
        for ca, cb in itertools.product((False, True), repeat=2):
            source = ('const ' if ca else '') + 'bit[8] a="10110110"; '
            source += ('const ' if cb else '') + 'bit[8] b="11010011"; '
            self.check_exact(source + 'int first=0; int last=3; int start=7; int stop=4; uint count=0; uint total=0; '
                'while (first<5) { last=first+3; start=7-first; stop=4-first; '
                'count=popcount(a[first:last] ^ b[start:-1:stop]); first+=1; '
                'if (first==2) { continue; } total+=count; if (first==4) { break; } }',
                (('total', None), ('count', None), ('first', None)), (sum(counts[i] for i in (0, 2, 3)), counts[3], 4))
        shrinking = sum(reference(selected_bits(4, 11, (0, 1, stop)), selected_bits(4, 13, (0, 1, stop)), '&')
                        for stop in (3, 2, 1, 0))
        self.check_exact('bit[4] a="1011"; bit[4] b="1101"; int first=0; int stop=3; uint total=0; '
            'while (stop>=0) { total+=popcount(a[first:stop] & b[first:stop]); stop-=1; } '
            'a="1111"; b="1111"; int last=3; int visits=0; '
            'while (popcount(a[first:last] & b[first:last])>uint(0)) { a>>=1; visits+=1; }',
            (('total', None), ('visits', None), ('a', 4), ('b', 4)), (shrinking, 4, 0, 15))

    def test_loop_bounds_sets_and_iterator_shadowing(self):
        self.check_exact('bit[4] a="1011"; const bit[4] B="1111"; int first=0; int last=3; uint total=0; '
            'for int i in [popcount(a[first:first] & B[first:first]):popcount(a[first:last] & B[first:last])] '
            '{ total+=uint(i); first=99; } first=0; '
            'for int i in {popcount(a[first:first] & B[first:first]),popcount(a[first:last] & B[first:last])} '
            '{ total+=uint(i); first=99; } '
            'array[int,2] indices={0,3}; for int first in indices { total+=popcount(a[first:first] & B[first:first]); }',
            (('total', None), ('first', None)), (12, 99))

    def test_skipped_invalid_ranges_and_short_circuiting(self):
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        base = ('bit[4] a="1011"; const bit[8] B="11010011"; uint result=7; int first=0; int last=3; '
                'int start=0; int stop=1; int zero=0; int loop_first=2; int loop_last=1; ')
        for value in ('popcount(a[first:0:999] & B[start/zero:stop])',
                      'popcount(B[start:stop] ^ a[first:last])',
                      'popcount(a[first/zero:last] | B[start:0:stop])'):
            for wrapper in wrappers:
                self.check_exact(base + wrapper.format('result=' + value + ';'), (('result', None),), (7,))
            self.check_exact(base + f'bool a_ok=false && bool({value}); bool b_ok=true || bool({value});',
                             (('a_ok', None), ('b_ok', None)), (0, 1))

    def test_reached_errors_capture_order_mismatches_and_capacity_limits(self):
        cases = (('a[first:0:last] & b[start/zero:stop]', 'assertion failed', False, 1000000),
                 ('a[first/zero:last] ^ b[start:0:stop]', 'division by zero', False, 1000000),
                 ('a[first:-1:last] | b[start/zero:stop]', 'assertion failed', False, 1000000),
                 ('a[first:4] & b[start/zero:stop]', 'assertion failed', False, 1000000),
                 ('a[first:last] & b[start/zero:stop]', 'division by zero', True, 1000000),
                 ('a[first:last] ^ b[start:0:stop]', 'assertion failed', True, 1000000),
                 ('a[first:last] | b[start:1]', 'assertion failed', True, 1000000),
                 ('a[first:last] & b[stop:start]', 'assertion failed', True, 1000000),
                 ('a[first:last] ^ b[start:8]', 'assertion failed', True, 1000000),
                 ('a[first:2] & b[start/zero:stop]', 'assertion failed', False, 2),
                 ('a[first:1] | b[start:2]', 'assertion failed', True, 2))
        cls = self.module.QASM2QCXConverter
        for ca, cb in itertools.product((False, True), repeat=2):
            for value, error, left_done, capacity in cases:
                source = ('const ' if ca else '') + 'bit[4] a="1011"; '
                source += ('const ' if cb else '') + 'bit[8] b="11010011"; '
                source += ('int first=0; int last=3; int start=0; int stop=3; int zero=0; uint result=6; '
                           f'result=popcount({value});')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source)
                with patch.object(cls, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', capacity):
                    converter = cls(program); converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                destination = converter._QASM2QCXConverter__capitalize_variable_name('result')
                lines = []
                for line in converter:
                    if line == '@QASM2QCX_POPCOUNT_0_RIGHT_SNAPSHOT':
                        lines.extend([line, 'PRINTLN 33333'])
                        continue
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
        with patch.object(cls, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 1):
            self.check_exact('bit[129] a="' + '1' * 129 + '"; bit[1] b="1"; int first=128; int start=0; '
                'uint out=popcount(a[first:first] & b[start:start]);', (('out', None),), (1,))

    def test_measurement_feedback_and_normal_fused_gate_parameters(self):
        for preparation in ('', 'x q[0]; ', 'h q[0]; ', 'h q[0]; x q[0]; '):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; ' + preparation +
                'bit[2] measured=measure q; const bit[2] MASK="01"; int first=0; int last=1; '
                'uint count=popcount(measured[first:last] & MASK[first:last]); '
                'bool valid=count==uint(measured[0]); '
                'uint either=popcount(MASK[first:last] | measured[first:last]); bool valid_or=either==uint(1); '
                'uint flipped=popcount(measured[first:last] ^ MASK[first:last]); bool valid_xor=flipped==uint(1)-count; '
                'if (count==uint(1)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0];',
                (('valid', None), ('valid_or', None), ('valid_xor', None), ('feedback', None)), (1, 1, 1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for operation, reverse in itertools.product(('&', '|', '^'), (False, True)):
            count = reference([1, 1, 0, 1], [1, 0, 1, 1], operation)
            phase = cmath.exp(count * 1j * math.pi / 8)
            left, right = 'a[first:last]', 'b[start:stop]'
            if reverse:
                left, right = right, left
            value = f'popcount({left} {operation} {right})'
            for fused in (False, True):
                for gate in (f'p(float({value}) * pi/8) q;', f'pow({value}) @ p(pi/8) q;'):
                    source = ('include "stdgates.inc"; qubit[2] q; bit[4] a="1011"; bit[4] b="1101"; '
                              'int first=0; int last=3; int start=0; int stop=3; h q; ' + gate)
                    BackendAuditTests.check(self, self.program(source, fused=fused),
                                            (0.5, phase/2, phase/2, phase*phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime range-pair popcount programs checked: {BackendTests.checked}; '
          f'bit-vector comparisons: {BackendTests.comparisons}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
