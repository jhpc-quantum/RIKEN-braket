#!/usr/bin/env python3
"""Audit popcount on direct runtime bit-register ranges (#481)."""

import argparse
import cmath
import math
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_bit_cast_numerical as casts


def count_selected(value, size, first, step, last):
    """Count individual selected bits, preserving normalized aliases."""
    positions = range(first, last + (1 if step > 0 else -1), step)
    assert positions
    assert all(-size <= index < size for index in positions)
    return sum((value >> (index % size)) & 1 for index in positions)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def test_widths_patterns_strides_reverse_and_repeated_normalized_positions(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (*range(1, native + 2), 129):
            for pattern in (0, (1 << size) - 1, int(('10110010' * size)[:size], 2)):
                cases = ((0, 1, size - 1), (size - 1, -1, 0), (-size, 1, size - 1), (0, 2, size - 1))
                for constant in (False, True):
                    source = ('const ' if constant else '') + f'bit[{size}] flags = "{pattern:0{size}b}"; '
                    source += f'int first = 0; int last = {size - 1}; int step = 1; '
                    outputs, expected = [], []
                    for index, (first, step, last) in enumerate(cases):
                        source += (f'first = {first}; step = {step}; last = {last}; '
                                   f'uint result{index} = popcount(flags[first:step:last]); ')
                        outputs.append((f'result{index}', None))
                        expected.append(count_selected(pattern, size, first, step, last))
                    source += f'bool same = flags == "{pattern:0{size}b}";'
                    self.check_exact(source, (*outputs, ('same', None)), (*expected, 1))

    def test_native_range_operands_and_advancement_past_endpoints(self):
        converter = self.module.QASM2QCXConverter
        for size in (1, 4, 33, 129):
            pattern = int(('10110010' * size)[:size], 2)
            cases = ((0, converter.QCX_UINT_MAX, size - 1, 'uint'),
                     (size - 1, converter.QCX_INT_MIN, 0, 'int'),
                     (-size, converter.QCX_INT_MAX, size - 1, 'int'))
            for first, step, last, step_type in cases:
                for constant in (False, True):
                    source = ('const ' if constant else '') + f'bit[{size}] flags = "{pattern:0{size}b}"; '
                    source += (f'int first = {first}; {step_type} step = {step}; int last = {last}; '
                               'uint count = popcount(flags[first:step:last]);')
                    self.check_exact(source, (('count', None),), (count_selected(pattern, size, first, step, last),))

    def test_native_result_width_narrow_assignments_and_unsigned_arithmetic(self):
        maximum = self.module.QASM2QCXConverter.QCX_UINT_MAX
        for size in (8, 33, 129):
            self.check_exact(f'const bit[{size}] C = "' + '1' * size + '"; '
                f'int first = 0; int last = {size - 1}; '
                'uint count = popcount(C[first:last]); uint[3] narrow = popcount(C[first:last]); '
                'uint complement = ~popcount(C[first:last]); uint power = popcount(C[first:last]) ** 3; '
                'bit[8] packed = bit[8](uint[8](popcount(C[first:last])));',
                (('count', None), ('narrow', None), ('complement', None), ('power', None), ('packed', 8)),
                (size, size % 8, maximum - size, size ** 3, size))

    def test_nested_scalar_calls_bound_captures_and_array_consumers(self):
        for pattern in range(16):
            count = pattern.bit_count()
            # A one-bit popcount selects one of the two array elements.
            index = pattern & 1
            bound = count % 4
            self.check_exact(f'bit[4] flags = "{pattern:04b}"; const bit[4] C = "1011"; '
                'int first = 0; int last = 3; array[uint,2] values = {0,0}; '
                'values[popcount(flags[first:first])] = popcount(flags[first:last]) ** 2; '
                'uint saved = values[popcount(flags[first:first])]; '
                'uint nested = popcount(uint[8](popcount(flags[first:last]))); '
                'uint bounded = popcount(flags[first:int(popcount(flags[first:last]) % uint(4))]); '
                'uint step_count = popcount(C[first:int(popcount(flags[first:last]) + uint(1)):last]); '
                'float root = sqrt(popcount(flags[first:last]) + uint(1)); '
                f'bool same = flags == "{pattern:04b}"; uint other = values[{1 - index}]; '
                'bool root_ok = root * root > float(popcount(flags[first:last])) + 0.99 '
                '&& root * root < float(popcount(flags[first:last])) + 1.01;',
                (('saved', None), ('nested', None), ('bounded', None), ('step_count', None),
                 ('same', None), ('other', None), ('root_ok', None)),
                (count ** 2, count.bit_count(), count_selected(pattern, 4, 0, 1, bound),
                 count_selected(11, 4, 0, count + 1, 3), 1, 0, 1))

    def test_runtime_loops_transfers_and_source_mutations(self):
        for constant in (False, True):
            pattern = 0b10110101
            expected = sum(count_selected(pattern, 8, first, 1, first + 3) for first in (0, 2, 3))
            source = ('const ' if constant else '') + 'bit[8] flags = "10110101"; '
            source += ('int first = 0; int last = 3; uint total = 0; uint count = 0; '
                       'while (first < 5) { last = first + 3; count = popcount(flags[first:last]); first += 1; '
                       'if (first == 2) { continue; } total += count; if (first == 4) { break; } }')
            self.check_exact(source, (('total', None), ('count', None), ('first', None)),
                             (expected, count_selected(pattern, 8, 3, 1, 6), 4))
        self.check_exact('bit[4] flags = "1111"; int first = 0; int last = 3; int visits = 0; '
            'while (popcount(flags[first:last]) > 0) { flags >>= 1; visits += 1; }',
            (('visits', None), ('flags', 4)), (4, 0))
        self.check_exact('const bit[4] C = "1011"; int first = 0; int last = 3; uint total = 0; '
            'for int i in [popcount(C[first:first]):popcount(C[first:last])] { total += uint(i); first = 99; } '
            'first = 0; for int i in {popcount(C[first:first]),popcount(C[first:last])} { total += uint(i); first = 99; } '
            'array[int,2] indices = {0,3}; for int first in indices { total += popcount(C[first:first]); } '
            'uint[4] saved = uint[4](popcount(C[0:last]));',
            (('total', None), ('first', None), ('saved', None)), (12, 99, 3))

    def test_skipped_errors_and_short_circuiting(self):
        calls = ('count = popcount(flags[first:0:last]);', 'count = popcount(C[first / zero:last]);',
                 'count = popcount(flags[first:step:last]);')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}',
                    'for int outer in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int outer in [0:1] {{ continue; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('bit[4] flags = "1010"; const bit[4] C = "1011"; uint count = 6; '
                    'int first = 99; int last = 0; int step = 0; int zero = 0; int loop_first = 2; int loop_last = 1; '
                    + wrapper.format(call), (('count', None), ('flags', 4)), (6, 10))
        self.check_exact('bit[4] flags = "1010"; int first = 99; int last = 0; int zero = 0; '
            'bool left = false && bool(popcount(flags[first:0:last])); '
            'bool right = true || bool(popcount(flags[first / zero:last])); '
            'bool once = false || bool(popcount(flags[last:last]));',
            (('left', None), ('right', None), ('once', None)), (0, 1, 0))

    def test_invalid_ranges_capacity_and_later_operand_failures_precede_writes(self):
        converter_type = self.module.QASM2QCXConverter
        cases = (('first = 0; last = 8;', 'popcount(flags[first:last])', 16, False, 'assertion failed'),
                 ('step = 0;', 'popcount(flags[first:step:last])', 16, False, 'assertion failed'),
                 ('first = 3; last = 0;', 'popcount(flags[first:last])', 16, False, 'assertion failed'),
                 ('first = 99;', 'popcount(flags[first:last])', 16, False, 'assertion failed'),
                 ('', 'popcount(flags[first:last])', 2, False, 'assertion failed'),
                 ('', 'popcount(flags[first:int(popcount(flags[first:last])) / zero])', 16, True, 'division by zero'))
        for constant in (False, True):
            for update, expression, capacity, read_before_error, error in cases:
                source = ('const ' if constant else '') + 'bit[8] flags = "10110010"; '
                source += ('int first = 0; int last = 3; int step = 1; int zero = 0; uint count = 6; '
                           + update + f'count = {expression};')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source)
                with patch.object(converter_type, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', capacity):
                    converter = converter_type(program)
                    converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                destination = converter._QASM2QCXConverter__capitalize_variable_name('count')
                lines = []
                for line in converter:
                    if line.startswith('LET QASM2QCX_') and ' += :UINT:' in line:
                        lines.append('PRINTLN 22222')
                    if line.startswith(f'LET {destination} := QASM2QCX_'):
                        lines.append('PRINTLN 11111')
                    lines.append(line)
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), ['22222'] * (4 if read_before_error else 0))
                type(self).checked += 1
        for constant in (False, True):
            source = ('const ' if constant else '') + 'bit[8] flags = "10110010"; int first = 0; int last = 1; '
            with patch.object(converter_type, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 2):
                self.check_exact(source + 'uint count = popcount(flags[first:last]);', (('count', None),), (1,))

    def test_measurement_feedback_and_quantum_parameters_with_fusion(self):
        for _ in range(16):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; '
                'bit[2] measured = measure q; int first = 0; int last = 1; '
                'uint count = popcount(measured[first:last]); bool same = count == uint(measured[0]); '
                'if (bool(count)) { x q[1]; } bit[2] observed = measure q; '
                'bool feedback = observed[1] == measured[0] && observed[0] == measured[0];',
                (('same', None), ('feedback', None)), (1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = cmath.exp(3j * math.pi / 8)
        for fused in (False, True):
            for gate in ('p(float(popcount(flags[first:last])) * pi / 8) q;',
                         'pow(popcount(flags[first:last])) @ p(pi / 8) q;'):
                source = ('include "stdgates.inc"; qubit[2] q; bit[4] flags = "1011"; '
                          'int first = 0; int last = 3; h q; ' + gate)
                BackendAuditTests.check(self, self.program(source, fused=fused), (0.5, phase / 2, phase / 2, phase * phase / 2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime range popcount programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
