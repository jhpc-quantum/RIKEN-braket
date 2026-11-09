#!/usr/bin/env python3
"""Audit bitwise runtime-range population counts with fixed-width operands (#499).

Run with --bra /path/to/bra, including a UBSan build. The reference reads
individual source positions and combines independent little-endian bit vectors.
"""

import argparse
import cmath
import itertools
import math
import operator
import pathlib
import resource
import unittest

import qasm2qcx_angle_bit_cast_numerical as casts


def reference(bits, mask, operation):
    assert len(bits) == len(mask)
    combine = {'&': operator.and_, '|': operator.or_, '^': operator.xor}[operation]
    return sum(combine(a, b) for a, b in zip(bits, mask))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    comparisons = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def test_documented_bitwise_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-range-popcount-bitwise-example -->'
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        source = source.replace('OPENQASM 3.0;', '', 1)
        self.check_exact(source, tuple((name, None) for name in
            ('left', 'right', 'united', 'different', 'constant_count', 'expression', 'repeated', 'wide', 'skipped', 'fresh')),
            (1, 1, 3, 2, 1, 1, 2, 2, 0, 0))

    def check_case(self, size, pattern, mask, constant, first, step, last):
        positions = list(range(first, last + (1 if step > 0 else -1), step))
        assert positions and all(-size <= index < size for index in positions)
        bits = [(pattern >> (index % size)) & 1 for index in positions]
        width = len(bits)
        fixed = [(mask >> index) & 1 for index in range(width)]
        source = [('const ' if constant else '') + f'bit[{size}] flags="{pattern:0{size}b}"; ',
                  f'bit[{width}] mask="{mask:0{width}b}"; const bit[{width}] MASK="{mask:0{width}b}"; ',
                  f'int first={first}; int step={step}; int last={last}; ']
        outputs, expected = [], []
        for index, (operation, order, operand) in enumerate(itertools.product(
                ('&', '|', '^'), (False, True), ('mask', 'MASK', f'"{mask:0{width}b}"'))):
            left, right = 'flags[first:step:last]', operand
            if order:
                left, right = right, left
            name = f'out{index}'
            source.append(f'uint {name}=popcount({left} {operation} {right}); ')
            outputs.append((name, None))
            expected.append(reference(bits, fixed, operation))
        source.append(f'bool same=flags=="{pattern:0{size}b}" && mask=="{mask:0{width}b}"; '
                      'uint fresh=popcount(flags[first:step:last]);')
        self.check_exact(''.join(source), (*outputs, ('same', None), ('fresh', None)), (*expected, 1, sum(bits)))
        type(self).comparisons += len(expected)

    def test_exhaustive_small_patterns_operators_orders_and_source_kinds(self):
        for size in (1, 2, 3):
            for pattern, mask, constant in itertools.product(range(1 << size), range(1 << size), (False, True)):
                self.check_case(size, pattern, mask, constant, 0, 1, size - 1)

    def test_every_native_width_and_wide_operands(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (*range(1, native + 2), 65, 129):
            limit = (1 << size) - 1
            alternating = int(('10110010' * size)[:size], 2)
            for pattern, mask in ((0, limit), (limit, 0), (alternating, limit ^ alternating)):
                for constant in (False, True):
                    self.check_case(size, pattern, mask, constant, 0, 1, size - 1)

    def test_reverse_nonunit_negative_alias_and_singleton_ranges(self):
        for size, first, step, last in ((8, -1, -2, -7), (8, -8, 1, 7),
                                      (129, 128, -3, 119), (8, 2, 4, 7), (1, -1, 1, 0)):
            width = len(range(first, last + (1 if step > 0 else -1), step))
            for constant in (False, True):
                self.check_case(size, int(('11010011' * size)[:size], 2),
                                int(('10011010' * width)[:width], 2), constant, first, step, last)

    def test_fixed_expression_selections_shifts_rotations_and_casts(self):
        flags = [1, 1, 0, 1]
        expressions = (('~mask', [0, 1, 0, 0]), ('mask >> shift', [0, 1, 1, 0]),
                       ('rotl(mask, rotation)', [0, 1, 1, 1]), ('mask ^ "1010"', [1, 1, 1, 0]),
                       ('mask[3:-1:0]', [1, 1, 0, 1]), ('mask[{index,0,1,2}]', [1, 1, 0, 1]),
                       ('bit[4](uint[4](mask))', [1, 0, 1, 1]),
                       ('bit[4](angle[4](mask))', [1, 0, 1, 1]))
        for constant in (False, True):
            source = [('const ' if constant else '') + 'bit[4] flags="1011"; bit[4] mask="1101"; '
                      'int first=0; int last=3; int index=3; uint shift=1; int rotation=-1; ']
            outputs, expected = [], []
            for index, ((fixed, values), operation, order) in enumerate(itertools.product(
                    expressions, ('&', '|', '^'), (False, True))):
                left, right = 'flags[first:last]', f'({fixed})'
                if order:
                    left, right = right, left
                name = f'out{index}'
                source.append(f'uint {name}=popcount({left} {operation} {right}); ')
                outputs.append((name, None))
                expected.append(reference(flags, values, operation))
            self.check_exact(''.join(source), outputs, expected)
            type(self).comparisons += len(expected)

    def test_nested_consumers_narrow_results_bounds_and_arrays(self):
        self.check_exact('bit[4] flags="1011"; bit[4] mask="1101"; int first=0; int last=3; '
            'uint result=popcount(flags[first:last] & mask); '
            'array[uint,4] values={0,0,0,0}; values[popcount(mask ^ flags[first:last])]=result; '
            'uint stored=values[2]; uint[1] narrow=popcount(flags[first:last] & mask); '
            'bit[2] packed=bit[2](uint[2](popcount(mask ^ flags[first:last]))); '
            'uint nested=popcount(flags[first:last] & rotl(mask, popcount(mask ^ flags[first:last]))); '
            'uint bounded=popcount(flags[int(result):last] & "10"); '
            'float root=sqrt(float(popcount(flags[first:last] | mask))); bool valid=root==2.0;',
            (('result', None), ('stored', None), ('narrow', None), ('packed', 2), ('nested', None),
             ('bounded', None), ('valid', None), ('flags', 4), ('mask', 4)), (2, 2, 0, 2, 2, 1, 1, 11, 13))

    def test_fresh_loop_values_transfers_and_captured_loop_bounds(self):
        bits = [(0xB6 >> index) & 1 for index in range(8)]
        mask = [1, 0, 1, 1]
        counts = [reference(bits[first:first + 4], mask, '^') for first in range(4)]
        for constant in (False, True):
            source = ('const ' if constant else '') + 'bit[8] flags="10110110"; bit[4] mask="1101"; '
            self.check_exact(source + 'int first=0; int last=3; uint count=0; uint total=0; '
                'while (first<5) { last=first+3; count=popcount(mask ^ flags[first:last]); first+=1; '
                'if (first==2) { continue; } total+=count; if (first==4) { break; } }',
                (('total', None), ('count', None), ('first', None)), (sum(counts[i] for i in (0, 2, 3)), counts[3], 4))
        self.check_exact('bit[4] flags="1111"; int first=0; int last=3; int visits=0; '
            'while (popcount(flags[first:last] & "1111")>uint(0)) { flags>>=1; visits+=1; } '
            'uint total=0; flags="1011"; '
            'for int i in [popcount(flags[first:last] & "0001"):popcount("0000" | flags[first:last])] '
            '{ total+=uint(i); first=99; } first=0; '
            'for int i in {popcount(flags[first:last] & "0001"),popcount("0000" | flags[first:last])} '
            '{ total+=uint(i); first=99; }', (('visits', None), ('total', None), ('first', None)), (4, 10, 99))

    def test_skipped_calls_and_short_circuiting_suppress_value_errors(self):
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int i in [0:1] {{ continue; {} }}')
        base = ('bit[4] flags="1011"; bit[4] mask="1101"; uint result=7; int first=0; int last=3; '
                'int zero=0; uint huge=uint(-1); int loop_first=2; int loop_last=1; ')
        for value in ('popcount(flags[first:0:999] & (mask << -1))',
                      'popcount(rotl(mask, huge) ^ flags[first/zero:last])',
                      'popcount(flags[first:last] | ("11" << (1/zero)))'):
            for wrapper in wrappers:
                self.check_exact(base + wrapper.format('result=' + value + ';'), (('result', None),), (7,))
            self.check_exact(base + f'bool a=false && bool({value}); bool b=true || bool({value});',
                             (('a', None), ('b', None)), (0, 1))

    def test_reached_failures_follow_operand_order_without_destination_writes(self):
        cases = (('flags[first/zero:last] & (mask << -1)', 'division by zero', 0),
                 ('(mask << -1) & flags[first/zero:last]', 'assertion failed', 0),
                 ('flags[first:last] & ("11" << (1/zero))', 'assertion failed', 0),
                 ('("11" << (1/zero)) & flags[first:last]', 'division by zero', 0),
                 ('flags[first:0:last] | mask', 'assertion failed', 0),
                 ('mask ^ flags[last:first]', 'assertion failed', 0),
                 ('flags[first:99] & mask', 'assertion failed', 0),
                 ('flags[first:last] & (mask << -1)', 'assertion failed', 4),
                 ('flags[first:last] | (mask << (1/zero))', 'division by zero', 4),
                 ('flags[first:last] ^ rotl(mask, huge)', None, 4),
                 ('rotl(mask, huge) ^ flags[first/zero:last]', None, 0),
                 ('flags[first:last] & mask[{1/zero,0,1,2}]', 'division by zero', 4))
        for constant in (False, True):
            for value, error, reads in cases:
                source = ('const ' if constant else '') + 'bit[4] flags="1011"; bit[4] mask="1101"; '
                source += ('int first=0; int last=3; int zero=0; uint huge=uint(-1); uint result=6; '
                           f'result=popcount({value});')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source)
                converter = self.module.QASM2QCXConverter(program); converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                destination = converter._QASM2QCXConverter__capitalize_variable_name('result')
                lines = []
                for line in converter:
                    if line.startswith('LET QASM2QCX_INT_') and ' := QASM2QCX_POSITIONS_' in line:
                        lines.append('PRINTLN 22222')
                    if line.startswith(f'LET {destination} := QASM2QCX_'):
                        lines.append('PRINTLN 11111')
                    lines.append(line)
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0)
                if error is not None:
                    self.assertIn(error, result.stderr)
                else:
                    self.assertNotIn('division by zero', result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), ['22222'] * reads)
                type(self).checked += 1

    def test_measurement_feedback_and_normal_fused_gate_parameters(self):
        for preparation in ('', 'x q[0]; ', 'h q[0]; ', 'h q[0]; x q[0]; '):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; ' + preparation +
                'bit[2] measured=measure q; int first=0; int last=1; '
                'uint count=popcount(measured[first:last] & "01"); '
                'bool valid=count==uint(measured[0]); '
                'uint either=popcount("01" | measured[first:last]); bool valid_or=either==uint(1); '
                'uint flipped=popcount(measured[first:last] ^ "01"); bool valid_xor=flipped==uint(1)-count; '
                'if (count==uint(1)) { x q[1]; } bit[2] observed=measure q; '
                'bool feedback=observed[1]==measured[0];',
                (('valid', None), ('valid_or', None), ('valid_xor', None), ('feedback', None)), (1, 1, 1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for operation, reverse in itertools.product(('&', '|', '^'), (False, True)):
            count = reference([1, 1, 0, 1], [1, 0, 1, 1], operation)
            phase = cmath.exp(count * 1j * math.pi / 8)
            left, right = 'flags[first:last]', 'mask'
            if reverse:
                left, right = right, left
            value = f'popcount({left} {operation} {right})'
            for fused in (False, True):
                for gate in (f'p(float({value}) * pi/8) q;', f'pow({value}) @ p(pi/8) q;'):
                    source = ('include "stdgates.inc"; qubit[2] q; bit[4] flags="1011"; bit[4] mask="1101"; '
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
    print(f'bitwise runtime range popcount programs checked: {BackendTests.checked}; '
          f'bit-vector comparisons: {BackendTests.comparisons}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
