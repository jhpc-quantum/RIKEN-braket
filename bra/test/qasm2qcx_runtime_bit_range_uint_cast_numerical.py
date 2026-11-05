#!/usr/bin/env python3
"""Audit direct UINT casts from runtime bit-register ranges (#473)."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_bit_cast_numerical as casts


def selected_value(pattern, size, first, step, last):
    """Reference the raw inclusive progression, then normalize each index."""
    positions = tuple(range(first, last + (1 if step > 0 else -1), step))
    if not positions or any(not -size <= index < size for index in positions):
        raise ValueError("reference requires a nonempty range of valid source indices")
    return sum(((pattern >> (index % size)) & 1) << offset
               for offset, index in enumerate(positions))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def test_all_native_widths_ascending_descending_and_source_preservation(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        size = native + 3
        for width in range(1, native + 1):
            for first, step, last in ((1, 1, width), (width, -1, 1)):
                for pattern in (0, (1 << size) - 1, int(('10110010' * size)[:size], 2)):
                    expected = selected_value(pattern, size, first, step, last)
                    bits = f'{pattern:0{size}b}'
                    for constant in (False, True):
                        self.check_exact(('const ' if constant else '') + f'bit[{size}] flags = "{bits}"; '
                            f'int first = {first}; int step = {step}; int last = {last}; '
                            f'uint[{width}] packed = uint[{width}](flags[first:step:last]); '
                            f'bit[{width}] back = bit[{width}](packed); bool same = flags == "{bits}";',
                            (('packed', None), ('back', width), ('same', None)), (expected, expected, 1))

    def test_unsized_uint_and_singleton_sources_keep_all_native_result_bits(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for pattern in (1 << (native - 1), (1 << native) - 1):
            self.check_exact(f'bit[{native}] flags = "{pattern:0{native}b}"; int first = 0; '
                f'uint last = {native - 1}; uint packed = uint(flags[first:last]); '
                f'bit[{native}] back = bit[{native}](packed);',
                (('packed', None), ('back', native), ('flags', native)), (pattern,) * 3)
        for constant in (False, True):
            for bit in (0, 1):
                self.check_exact(('const ' if constant else '') + f'bit[1] flags = "{bit}"; int first = -1; '
                    'uint[1] single = uint[1](flags[first:-1]); uint[2] aliases = uint[2](flags[first:0]);',
                    (('single', None), ('aliases', None)), (bit, 3 * bit))

    def test_raw_progression_nonunit_steps_negative_indices_and_repeated_aliases(self):
        cases = ((-4, 1, -1), (-1, -1, -4), (-1, 1, 3), (3, -1, -1), (-4, 2, 3),
                 (0, 2, 3), (3, -2, 0), (-4, 3, 3), (-1, 2, 3), (0, 10, 8),
                 (3, -12, -8), (2, -3, 2))
        for first, step, last in cases:
            width = len(range(first, last + (1 if step > 0 else -1), step))
            for pattern in (0, 9, 15):
                for constant in (False, True):
                    expected = selected_value(pattern, 4, first, step, last)
                    self.check_exact(('const ' if constant else '') + f'bit[4] flags = "{pattern:04b}"; '
                        f'int first = {first}; int step = {step}; int last = {last}; '
                        f'uint[{width}] packed = uint[{width}](flags[first:step:last]); '
                        f'bool same = flags == "{pattern:04b}";',
                        (('packed', None), ('same', None)), (expected, 1))

    def test_mixed_constant_runtime_bounds_and_unsigned_steps(self):
        for operand, expected in (('flags[first:3]', 9), ('flags[0:last]', 9),
                                   ('flags[0:step:3]', 9), ('flags[first:step:last]', 9),
                                   ('flags[first:1:last]', 9)):
            self.check_exact('const bit[4] flags = "1001"; int first = 0; uint last = 3; uint step = 1; '
                f'uint[4] packed = uint[4]({operand});', (('packed', None),), (expected,))
        limits = self.module.QASM2QCXConverter
        for declaration, operand, expected in (
                (f'uint step = {limits.QCX_UINT_MAX}; int last = {limits.QCX_INT_MAX};', 'flags[0:step:last]', 1),
                (f'int step = {limits.QCX_INT_MIN}; int last = -1;', 'flags[3:step:last]', 1),
                (f'int first = 0; uint step = {limits.QCX_UINT_MAX}; int last = 3;', 'flags[first:step:last]', 1)):
            self.check_exact('bit[4] flags = "1001"; ' + declaration + f'uint[1] packed = uint[1]({operand});',
                             (('packed', None), ('flags', 4)), (expected, 9))

    def test_constant_lookup_spans_full_and_partial_native_words(self):
        word = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (word - 1, word, word + 1, 2 * word - 1, 2 * word, 2 * word + 1, 129):
            pattern = int(('10010110' * size)[:size], 2)
            for first in sorted({0, max(0, word - 2), size - 4}):
                for start, step, last in ((first, 1, first + 3), (first + 3 - size, -1, first - size)):
                    if max(start, last) >= size:
                        continue
                    expected = selected_value(pattern, size, start, step, last)
                    self.check_exact(f'const bit[{size}] C = "{pattern:0{size}b}"; '
                        f'int first = {start}; int step = {step}; int last = {last}; '
                        'uint[4] packed = uint[4](C[first:step:last]);', (('packed', None),), (expected,))

    def test_overlapping_assignments_use_complete_original_snapshot(self):
        for pattern in range(16):
            reversed_bits = int(f'{pattern:04b}'[::-1], 2)
            self.check_exact(f'bit[4] flags = "{pattern:04b}"; int first = 3; int last = 0; '
                'uint[4] saved = uint[4](flags[first:-1:last]); '
                'flags = bit[4](uint[4](flags[first:-1:last])); '
                'first = 0; last = 3; uint[4] fresh = uint[4](flags[first:last]);',
                (('saved', None), ('flags', 4), ('fresh', None)), (reversed_bits,) * 3)

    def test_range_failures_precede_selected_reads_and_destination_writes(self):
        limits = self.module.QASM2QCXConverter
        cases = (('int first = 0; int step = 0; int last = 3;', 'first:step:last', 4, 'assertion failed'),
                 ('int first = 0; uint step = 0; int last = 0;', 'first:step:last', 1, 'assertion failed'),
                 ('int first = 0; int step = -1; int last = 3;', 'first:step:last', 4, 'assertion failed'),
                 ('int first = 3; int last = 0;', 'first:last', 4, 'assertion failed'),
                 ('int first = 0; int last = 3;', 'first:last', 3, 'assertion failed'),
                 ('int first = 0; int last = 2;', 'first:last', 4, 'assertion failed'),
                 ('int first = -9; int last = -6;', 'first:last', 4, 'assertion failed'),
                 ('int first = 5; int last = 8;', 'first:last', 4, 'assertion failed'),
                 (f'int first = {limits.QCX_INT_MIN}; int last = 0;', 'first:last', 4, 'assertion failed'),
                 (f'uint first = {limits.QCX_UINT_MAX}; uint last = {limits.QCX_UINT_MAX};', 'first:last', 1, 'assertion failed'),
                 (f'int first = 0; uint last = {limits.QCX_UINT_MAX};', 'first:last', 4, 'assertion failed'),
                 ('int first = 0; int last = 3; int zero = 0;', 'first:1 / zero:last', 4, 'division by zero'),
                 ('int first = 0; int last = 3; int zero = 0;', 'first:last / zero', 4, 'division by zero'),
                 ('int first = 0;', 'first:999999999999999999999999999999', 4, 'assertion failed'))
        for constant in (False, True):
            for declarations, operand, width, error in cases:
                source = ('const ' if constant else '') + 'bit[8] flags = "11010011"; '
                program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source + declarations +
                    f' uint[{width}] packed = 0; packed = uint[{width}](flags[{operand}]);')
                converter = self.module.QASM2QCXConverter(program)
                converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                flags = converter._QASM2QCXConverter__capitalize_variable_name('flags')
                destination = converter._QASM2QCXConverter__capitalize_variable_name('packed')
                instrumented = []
                previous = ''
                for line in converter:
                    if (f' := {flags}:QASM2QCX_' in line
                            or (' := :INT:QASM2QCX_UINT_' in line and previous.endswith(' &= 1'))):
                        instrumented.append('PRINTLN 22222')
                    if line.startswith(f'LET {destination} := QASM2QCX_'):
                        instrumented.append('PRINTLN 11111')
                    instrumented.append(line)
                    previous = line
                self.assertIn('PRINTLN 22222', instrumented)
                self.assertIn('PRINTLN 11111', instrumented)
                result = self.execute(instrumented)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertNotIn('22222', result.stdout)
                self.assertNotIn('11111', result.stdout)
                type(self).checked += 1


    def test_numeric_register_and_array_consumers_preserve_widths(self):
        for pattern in range(16):
            value = int(f'{pattern:04b}'[::-1], 2)
            rotated = ((value << 1) | (value >> 3)) & 15
            self.check_exact(f'bit[4] flags = "{pattern:04b}"; const bit[4] C = "1010"; '
                'int first = 3; int last = 0; '
                'uint[4] packed = uint[4](flags[first:-1:last]); '
                'uint[4] added = uint[4](flags[first:-1:last]) + uint[4](1); '
                'uint[4] combined = uint[4](flags[first:-1:last]) ^ uint[4](C[first:-1:last]); '
                'angle[4] phase = angle[4](bit[4](uint[4](flags[first:-1:last]))); '
                'uint[4] rotated = uint[4](rotl(bit[4](uint[4](flags[first:-1:last])),1)); '
                'uint ones = popcount(bit[4](uint[4](flags[first:-1:last]))); '
                'array[uint[4],2] values = {uint[4](flags[first:-1:last]), uint[4](C[first:-1:last])}; '
                'values[0] += values[1]; uint[4] element = values[0]; '
                'bit[4] shifted = "1111"; shifted >>= uint[4](flags[first:-1:last]); '
                f'bool same = flags == "{pattern:04b}";',
                (('packed', None), ('added', None), ('combined', None), ('phase', None), ('rotated', None),
                 ('ones', None), ('element', None), ('shifted', 4), ('same', None)),
                (value, (value + 1) & 15, value ^ 5, value, rotated, value.bit_count(),
                 (value + 5) & 15, 15 >> value, 1))

    def test_nested_casts_in_range_bounds_steps_and_array_destinations(self):
        for pattern in range(16):
            for base in (0, 1):
                index = (pattern >> base) & 3
                expected = (pattern >> index) & 1
                self.check_exact(f'bit[4] flags = "{pattern:04b}"; int base = {base}; '
                    'array[uint[4],2] values = {0,0}; '
                    'values[uint[1](flags[base:base])] = uint[1](flags['
                    'uint[2](flags[base:base + 1]):uint[2](flags[base:base + 1])]); '
                    'uint[4] out = values[uint[1](flags[base:base])]; '
                    'uint[1] stepped = uint[1](flags[0:(uint[2](flags[base:base + 1]) | uint[2](1)):0]); '
                    f'bool same = flags == "{pattern:04b}";',
                    (('out', None), ('stepped', None), ('same', None)), (expected, pattern & 1, 1))

    def test_while_conditions_and_transfers_use_fresh_sources_and_bounds(self):
        for constant in (False, True):
            self.check_exact(('const ' if constant else '') + 'bit[8] flags = "10110101"; '
                'int first = 0; int last = 3; uint total = 0; uint[4] value = 0; '
                'while (first < 5) { last = first + 3; value = uint[4](flags[first:last]); '
                'first += 1; if (first == 2) { continue; } total += value; '
                'if (first == 4) { break; } }',
                (('total', None), ('value', None), ('first', None), ('last', None)), (24, 6, 4, 6))
        self.check_exact('bit[4] flags = "0000"; int first = 0; int last = 3; uint[4] packed = 0; int visits = 0; '
            'while (uint[4](flags[first:last]) != uint[4](8)) { '
            'packed = uint[4](flags[first:last]) + uint[4](1); flags = bit[4](packed); visits += 1; }',
            (('visits', None), ('packed', None), ('flags', 4)), (8, 8, 8))

    def test_loop_bounds_sets_and_iterator_shadows_preserve_capture_and_scope(self):
        self.check_exact('const bit[4] C = "1001"; bit[4] flags = "0101"; int first = 0; int last = 2; uint total = 0; '
            'for int i in [uint[1](C[first:first]):uint[3](flags[first:last])] { '
            'total += uint(i); first = 9; last = 9; flags = "0000"; } '
            'array[uint[4],2] indices = {0,3}; for uint[4] first in indices { total += uint[1](C[first:first]); } '
            'for int first in [-4:-1] { total += uint[1](C[first:first]); }',
            (('total', None), ('first', None), ('last', None)), (19, 9, 9))
        self.check_exact('const bit[4] C = "1001"; uint total = 0; '
            'for int outer in [0:1] { '
            'for int i in [uint[1](C[outer:outer]):uint[2](C[outer:outer + 1])] { total += uint(i); } '
            'for int i in {uint[1](C[outer:outer]), uint[2](C[outer:outer + 1])} { total += uint(i); } }',
            (('total', None),), (3,))

    def test_skipped_paths_and_short_circuiting_suppress_reached_range_failures(self):
        calls = ('packed = uint[4](flags[first:step:last]);',
                 'packed = uint[4](C[first:999]);',
                 'packed = uint[4](flags[first / zero:last]);',
                 'packed = uint[4](flags[positions[bad]:last]);',
                 'packed = uint[4](flags[first:999999999999999999999999]);')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                    'for int outer in [1:0] {{ {} }}', 'for int outer in [loop_first:loop_last] {{ {} }}',
                    'while (visits < 2) {{ visits += 1; continue; {} }}', 'while (true) {{ break; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('bit[4] flags = "1010"; const bit[4] C = "1001"; uint[4] packed = 6; '
                    'array[int,2] positions = {0,1}; int first = 0; int last = -9; int step = 0; int zero = 0; '
                    'int bad = 3; int visits = 0; int loop_first = 2; int loop_last = 1; ' + wrapper.format(call),
                    (('packed', None), ('flags', 4)), (6, 10))
        for condition, expected in (('false && bool(uint[4](flags[first:step:last]))', 0),
                                    ('true || bool(uint[4](flags[first / zero:last]))', 1)):
            self.check_exact('bit[4] flags = "1010"; int first = 0; int step = 0; int last = 999; int zero = 0; '
                f'bool result = {condition};', (('result', None), ('flags', 4)), (expected, 10))

    def test_later_iteration_failures_preserve_prior_successful_assignments(self):
        for constant in (False, True):
            for operand, update, error in (('first:step:last', 'step = 0;', 'assertion failed'),
                                           ('first:step:last', 'last = 2;', 'assertion failed'),
                                           ('first:step:last', 'first = -9;', 'assertion failed'),
                                           ('first / divisor:step:last', 'divisor = 0;', 'division by zero')):
                program = self.module.openqasm3.parser.parse('OPENQASM 3; ' +
                    ('const ' if constant else '') + 'bit[4] flags = "1101"; '
                    'int first = 0; int step = 1; int last = 3; int divisor = 1; int iteration = 0; uint[4] packed = 6; '
                    f'while (iteration < 2) {{ packed = uint[4](flags[{operand}]); iteration += 1; {update} }}')
                converter = self.module.QASM2QCXConverter(program)
                converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                destination = converter._QASM2QCXConverter__capitalize_variable_name('packed')
                instrumented = []
                for line in converter:
                    if line.startswith(f'LET {destination} := QASM2QCX_'):
                        instrumented.extend(['PRINTLN 11111', line, f'PRINTLN {destination}'])
                    else:
                        instrumented.append(line)
                self.assertIn('PRINTLN 11111', instrumented)
                result = self.execute(instrumented)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), ['11111', '13'])
                type(self).checked += 1

    def test_measurement_range_casts_drive_feedback_from_live_bits(self):
        for anticorrelated in (False, True):
            for _ in range(16):
                self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0],q[1]; '
                    + ('x q[1]; ' if anticorrelated else '') +
                    'bit[2] measured = measure q; int first = 1; int last = 0; '
                    'uint[2] packed = uint[2](measured[first:-1:last]); bit[2] back = bit[2](packed); '
                    'bool same = back[0] == measured[1] && back[1] == measured[0]; '
                    'if (bool(packed)) { x q[0]; } bit[2] observed = measure q; '
                    'bool feedback = bool(observed[0]) == (bool(measured[0]) != bool(packed)) '
                    '&& observed[1] == measured[1]; '
                    f'bool correlation = (back[0] != back[1]) == {str(anticorrelated).lower()};',
                    (('same', None), ('feedback', None), ('correlation', None)), (1, 1, 1))

    def test_quantum_parameters_and_integer_powers_with_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for fused in (False, True):
            for gate in ('p(angle[4](bit[4](uint[4](flags[first:-1:last])))) q;',
                         'pow(uint[4](flags[first:-1:last])) @ p(pi / 8) q;'):
                source = 'include "stdgates.inc"; qubit q; bit[4] flags = "0110"; int first = 3; int last = 0; h q; ' + gate
                BackendAuditTests.check(self, self.program(source, fused=fused),
                    (1 / math.sqrt(2), cmath.exp(3j * math.pi / 4) / math.sqrt(2)))
            source = ('include "stdgates.inc"; qubit q; bit[4] flags = "0110"; int first = 3; int last = 0; '
                      'rx(float(uint[4](flags[first:-1:last])) / 4.0) q;')
            BackendAuditTests.check(self, self.program(source, fused=fused), (math.cos(0.75), -1j * math.sin(0.75)))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime bit-range UINT cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
