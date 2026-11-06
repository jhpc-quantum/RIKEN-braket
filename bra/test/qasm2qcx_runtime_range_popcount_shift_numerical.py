#!/usr/bin/env python3
"""Audit population counts of singly shifted runtime ranges (#485)."""

import argparse
import cmath
import math
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_bit_cast_numerical as casts


def count_shifted(value, size, first, step, last, distance, direction):
    """Reference using arbitrary-width integer shifts on the selected pattern."""
    positions = range(first, last + (1 if step > 0 else -1), step)
    assert positions and distance >= 0
    assert all(-size <= index < size for index in positions)
    width = len(positions)
    if distance >= width:
        return 0
    selected = sum(((value >> (index % size)) & 1) << position for position, index in enumerate(positions))
    shifted = (selected << distance) & ((1 << width) - 1) if direction == '<<' else selected >> distance
    return shifted.bit_count()


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def test_widths_patterns_directions_and_varying_selected_lengths(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (*range(1, native + 2), 129):
            for pattern in (0, (1 << size) - 1, int(('10110010' * size)[:size], 2)):
                cases = ((0, 1, size - 1), (size - 1, -1, 0), (-size, 1, size - 1), (0, 2, size - 1))
                for constant in (False, True):
                    source = ('const ' if constant else '') + f'bit[{size}] flags = "{pattern:0{size}b}"; '
                    source += f'int first = 0; int last = {size - 1}; int step = 1; int distance = 0; '
                    outputs, expected = [], []
                    for case, (first, step, last) in enumerate(cases):
                        width = len(range(first, last + (1 if step > 0 else -1), step))
                        source += f'first = {first}; step = {step}; last = {last}; '
                        for count_index, distance in enumerate((0, 1, width)):
                            source += f'distance = {distance}; '
                            for suffix, operator in (('l', '<<'), ('r', '>>')):
                                name = f'count{case}_{count_index}{suffix}'
                                source += f'uint {name} = popcount(flags[first:step:last] {operator} distance); '
                                outputs.append((name, None))
                                expected.append(count_shifted(pattern, size, first, step, last, distance, operator))
                    source += f'bool same = flags == "{pattern:0{size}b}";'
                    self.check_exact(source, (*outputs, ('same', None)), (*expected, 1))

    def test_native_and_literal_counts_and_storage_reuse_after_zero_branches(self):
        cls = self.module.QASM2QCXConverter
        for size in (1, 4, 33, 129):
            pattern = int(('10110010' * size)[:size], 2)
            for constant in (False, True):
                source = ('const ' if constant else '') + f'bit[{size}] flags = "{pattern:0{size}b}"; '
                source += f'int first = 0; int last = {size - 1}; int distance = {cls.QCX_INT_MAX}; uint u = {cls.QCX_UINT_MAX}; '
                outputs, expected = [], []
                for case, operand in enumerate(('u', 'distance', str(10 ** 70), '0', '1', str(size + 1), 'uint(-1)')):
                    for suffix, operator in (('l', '<<'), ('r', '>>')):
                        name = f'out{case}{suffix}'
                        source += f'uint {name} = popcount(flags[first:last] {operator} {operand}); '
                        outputs.append((name, None))
                        distance = (cls.QCX_UINT_MAX if operand in ('u', 'uint(-1)') else
                                    cls.QCX_INT_MAX if operand == 'distance' else int(operand))
                        expected.append(count_shifted(pattern, size, 0, 1, size - 1, distance, operator))
                source += 'distance = 1; uint fresh = popcount(flags[first:last] << distance); '
                source += 'uint unshifted = popcount(flags[first:last]);'
                self.check_exact(source, (*outputs, ('fresh', None), ('unshifted', None)),
                    (*expected, count_shifted(pattern, size, 0, 1, size - 1, 1, '<<'), pattern.bit_count()))

    def test_native_results_narrow_assignment_and_scalar_arithmetic(self):
        maximum = self.module.QASM2QCXConverter.QCX_UINT_MAX
        for size in (8, 33, 129):
            count = 2 * size - 1
            self.check_exact(f'const bit[{size}] C = "' + '1' * size + '"; '
                f'int first = -{size}; int last = {size - 1}; int distance = 1; '
                'uint count = popcount(C[first:last] << distance); uint[3] narrow = popcount(C[first:last] >> distance); '
                'uint complement = ~popcount(C[first:last] << distance); uint power = popcount(C[first:last] >> distance) ** 3;',
                (('count', None), ('narrow', None), ('complement', None), ('power', None)),
                (count, count % 8, maximum - count, count ** 3))

    def test_nested_counts_bounds_casts_and_array_consumers(self):
        for pattern in range(16):
            left = count_shifted(pattern, 4, 0, 1, 3, 1, '<<')
            index = pattern & 1
            bound = left % 4
            distance = pattern.bit_count()
            self.check_exact(f'bit[4] flags = "{pattern:04b}"; int first = 0; int last = 3; int distance = 1; '
                'array[uint,2] values = {0,0}; '
                'values[popcount(flags[first:first] << 0)] = popcount(flags[first:last] << distance) ** 2; '
                'uint saved = values[popcount(flags[first:first] << 0)]; '
                'uint nested = popcount(uint[8](popcount(flags[first:last] << distance))); '
                'uint counted = popcount(flags[first:last] >> popcount(flags[first:last] << 0)); '
                'uint bounded = popcount(flags[first:int(popcount(flags[first:last] << distance) % uint(4))] >> distance); '
                f'uint other = values[{1 - index}]; bool same = flags == "{pattern:04b}";',
                (('saved', None), ('nested', None), ('counted', None), ('bounded', None), ('other', None), ('same', None)),
                (left ** 2, left.bit_count(), count_shifted(pattern, 4, 0, 1, 3, distance, '>>'),
                 count_shifted(pattern, 4, 0, 1, bound, 1, '>>'), 0, 1))

    def test_runtime_loops_transfers_and_fresh_source_values(self):
        for constant in (False, True):
            pattern = 0b10110101
            expected = sum(count_shifted(pattern, 8, first, 1, first + 3, first % 3, '<<') for first in (0, 2, 3))
            source = ('const ' if constant else '') + 'bit[8] flags = "10110101"; '
            source += ('int first = 0; int last = 3; int distance = 0; uint total = 0; uint count = 0; '
                       'while (first < 5) { last = first + 3; distance = first % 3; '
                       'count = popcount(flags[first:last] << distance); first += 1; '
                       'if (first == 2) { continue; } total += count; if (first == 4) { break; } }')
            self.check_exact(source, (('total', None), ('count', None), ('first', None)),
                             (expected, count_shifted(pattern, 8, 3, 1, 6, 0, '<<'), 4))
        self.check_exact('bit[4] flags = "1111"; int first = 0; int last = 3; int distance = 1; int visits = 0; '
            'while (popcount(flags[first:last] >> distance) > 0) { flags >>= 1; visits += 1; }',
            (('visits', None), ('flags', 4)), (3, 1))
        self.check_exact('const bit[4] C = "1011"; int first = 0; int last = 3; uint total = 0; '
            'for int i in [popcount(C[first:first] << 0):popcount(C[first:last] >> 1)] { total += uint(i); first = 99; } '
            'first = 0; for int i in {popcount(C[first:first] << 0),popcount(C[first:last] >> 1)} { total += uint(i); first = 99; } '
            'array[int,2] indices = {0,3}; for int first in indices { total += popcount(C[first:first] << 0); }',
            (('total', None), ('first', None)), (8, 99))

    def test_skipped_errors_and_short_circuiting(self):
        calls = ('count = popcount(flags[first:0:last] << -1);', 'count = popcount(C[first / zero:last] >> uint(-1));',
                 'count = popcount(flags[first:last] << (1 / zero));')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}', 'for int outer in [1:0] {{ {} }}',
                    'for int outer in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int outer in [0:1] {{ continue; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('bit[4] flags = "1111"; const bit[4] C = "0000"; uint count = 6; '
                    'int first = 99; int last = 0; int zero = 0; int loop_first = 2; int loop_last = 1; '
                    + wrapper.format(call), (('count', None), ('flags', 4)), (6, 15))
        self.check_exact('bit[4] flags = "1010"; int first = 99; int last = 0; int zero = 0; '
            'bool left = false && bool(popcount(flags[first:0:last] << -1)); '
            'bool right = true || bool(popcount(flags[first / zero:last] >> uint(-1))); '
            'bool reached = false || bool(popcount(flags[last:last] << 0));',
            (('left', None), ('right', None), ('reached', None)), (0, 1, 0))

    def test_range_and_count_errors_precede_reads_and_destination_writes(self):
        cls = self.module.QASM2QCXConverter
        cases = (('last = 8;', 'u', 16, 'assertion failed'),
                 ('step = 0;', '1 / zero', 16, 'assertion failed'),
                 ('first = 3; last = 0;', 'u', 16, 'assertion failed'),
                 ('', '-1', 16, 'assertion failed'), ('', 'negative', 16, 'assertion failed'),
                 ('', 'minimum', 16, 'assertion failed'), ('', '1 / zero', 16, 'division by zero'),
                 ('', 'u', 2, 'assertion failed'))
        for constant in (False, True):
            for operator in ('<<', '>>'):
                for update, operand, capacity, error in cases:
                    source = ('const ' if constant else '') + 'bit[8] flags = "00000000"; '
                    source += ('int first = 0; int last = 3; int step = 1; int negative = -1; int zero = 0; '
                               f'int minimum = {cls.QCX_INT_MIN}; uint u = {cls.QCX_UINT_MAX}; uint count = 6; '
                               + update + f'count = popcount(flags[first:step:last] {operator} {operand});')
                    program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source)
                    with patch.object(cls, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', capacity):
                        converter = cls(program); converter.visit(program)
                    self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                    destination = converter._QASM2QCXConverter__capitalize_variable_name('count')
                    lines = []
                    for line in converter:
                        if (line.startswith(f'LET {destination} := QASM2QCX_') or
                                (line.startswith('LET QASM2QCX_') and ' += :UINT:' in line)):
                            lines.append('PRINTLN 22222')
                        lines.append(line)
                    result = self.execute(lines)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)
                    self.assertEqual(result.stdout.split(), [])
                    type(self).checked += 1

    def test_measurement_feedback_and_quantum_parameters_with_fusion(self):
        for _ in range(16):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; bit[2] measured = measure q; '
                'int first = 0; int last = 1; int distance = 1; '
                'uint left = popcount(measured[first:last] << distance); '
                'uint right = popcount(measured[first:last] >> distance); '
                'bool same = left == uint(measured[0]) && right == uint(measured[1]); '
                'if (bool(left)) { x q[1]; } bit[2] observed = measure q; '
                'bool feedback = observed[1] == measured[0] && observed[0] == measured[0];',
                (('same', None), ('feedback', None)), (1, 1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for operator, count in (('<<', 2), ('>>', 2)):
            phase = cmath.exp(count * 1j * math.pi / 8)
            for fused in (False, True):
                operand = f'popcount(flags[first:last] {operator} distance)'
                for gate in (f'p(float({operand}) * pi / 8) q;', f'pow({operand}) @ p(pi / 8) q;'):
                    source = ('include "stdgates.inc"; qubit[2] q; bit[4] flags = "1011"; '
                              'int first = 0; int last = 3; int distance = 1; h q; ' + gate)
                    BackendAuditTests.check(self, self.program(source, fused=fused), (0.5, phase / 2, phase / 2, phase * phase / 2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'shifted runtime range popcount programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
