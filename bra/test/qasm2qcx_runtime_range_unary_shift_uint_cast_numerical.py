#!/usr/bin/env python3
"""Audit complement/shift chains over runtime ranges in UINT casts (#475)."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_bit_cast_numerical as casts
from qasm2qcx_runtime_bit_range_uint_cast_numerical import selected_value


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def test_all_native_widths_keep_each_intermediate_within_the_selection(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        size = native + 3
        for width in range(1, native + 1):
            mask = (1 << width) - 1
            for pattern in (0, (1 << size) - 1, int(('10110010' * size)[:size], 2)):
                raw = selected_value(pattern, size, 1, 1, width)
                for constant in (False, True):
                    source = ('const ' if constant else '') + f'bit[{size}] flags = "{pattern:0{size}b}"; '
                    source += f'int first = 1; int last = {width}; int distance = 1; '
                    source += f'uint[{width}] inverted = uint[{width}](~flags[first:last]); '
                    source += f'uint[{width}] left = uint[{width}](flags[first:last] << distance); '
                    source += f'uint[{width}] right = uint[{width}](flags[first:last] >> distance); '
                    source += f'uint[{width}] after = uint[{width}](~(flags[first:last] << distance)); '
                    source += f'uint[{width}] before = uint[{width}]((~flags[first:last]) << distance); '
                    source += f'uint[{width}] truncated = uint[{width}]((flags[first:last] << distance) >> distance); '
                    source += f'bool same = flags == "{pattern:0{size}b}";'
                    self.check_exact(source, (('inverted', None), ('left', None), ('right', None),
                        ('after', None), ('before', None), ('truncated', None), ('same', None)),
                        (raw ^ mask, (raw << 1) & mask, raw >> 1, ((raw << 1) & mask) ^ mask,
                         ((raw ^ mask) << 1) & mask, ((raw << 1) & mask) >> 1, 1))

    def test_literal_and_runtime_counts_cover_zero_width_thresholds_and_native_endpoints(self):
        limits = self.module.QASM2QCXConverter
        for width in (1, 4, limits.QCX_UINT_WIDTH):
            mask = (1 << width) - 1
            raw = mask ^ (mask >> 2)
            for count in sorted({0, 1, width - 1, width, width + 1, limits.QCX_INT_MAX, limits.QCX_UINT_MAX}):
                forms = [('literal', str(count)), ('uint', 'distance')]
                if count <= limits.QCX_INT_MAX:
                    forms.append(('int', 'distance'))
                for kind, operand in forms:
                    declaration = '' if kind == 'literal' else f'{kind} distance = {count}; '
                    left = (raw << count) & mask if count < width else 0
                    right = raw >> count if count < width else 0
                    self.check_exact(f'bit[{width}] flags = "{raw:0{width}b}"; int first = 0; '
                        f'int last = {width - 1}; ' + declaration +
                        f'uint[{width}] left = uint[{width}](flags[first:last] << {operand}); '
                        f'uint[{width}] right = uint[{width}](flags[first:last] >> {operand}); '
                        f'uint[{width}] inverted = uint[{width}](~(flags[first:last] << {operand}));',
                        (('left', None), ('right', None), ('inverted', None), ('flags', width)),
                        (left, right, left ^ mask, raw))
        self.check_exact('bit[4] flags = "1011"; int first = 0; int last = 3; '
            'uint[4] left = uint[4](flags[first:last] << 999999999999999999999999); '
            'uint[4] right = uint[4](~(flags[first:last] >> 999999999999999999999999));',
            (('left', None), ('right', None)), (0, 15))

    def test_unsized_result_preserves_native_high_bits_and_metadata(self):
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        mask = (1 << width) - 1
        for raw in (0, 1, mask, 1 << (width - 1)):
            self.check_exact(f'bit[{width}] flags = "{raw:0{width}b}"; int first = 0; int last = {width - 1}; '
                'uint distance = 1; uint inverted = uint(~flags[first:last]); '
                'uint result = uint(~((flags[first:last] << distance) >> distance));',
                (('inverted', None), ('result', None), ('flags', width)),
                (raw ^ mask, (((raw << 1) & mask) >> 1) ^ mask, raw))

    def test_negative_aliases_strides_and_constant_lookup_words_preserve_order(self):
        word = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        cases = ((1, -1, 1, 0), (4, -1, 1, 3), (4, 3, -1, -1), (4, -4, 2, 3),
                 (word + 1, word, -1, word - 3), (129, 128, -3, 119))
        for size, first, step, last in cases:
            width = len(range(first, last + (1 if step > 0 else -1), step))
            mask = (1 << width) - 1
            for pattern in (0, (1 << size) - 1, int(('10110010' * size)[:size], 2)):
                raw = selected_value(pattern, size, first, step, last)
                for constant in (False, True):
                    self.check_exact(('const ' if constant else '') + f'bit[{size}] flags = "{pattern:0{size}b}"; '
                        f'int first = {first}; int step = {step}; int last = {last}; uint distance = 1; '
                        f'uint[{width}] inverted = uint[{width}](~flags[first:step:last]); '
                        f'uint[{width}] chain = uint[{width}](~((~flags[first:step:last] << distance) >> distance)); '
                        f'bool same = flags == "{pattern:0{size}b}";',
                        (('inverted', None), ('chain', None), ('same', None)),
                        (raw ^ mask, ((((raw ^ mask) << 1) & mask) >> 1) ^ mask, 1))

    def test_overlapping_assignments_snapshot_before_operations_and_writes(self):
        for raw in range(16):
            expected = ((raw << 1) & 15) ^ 15
            self.check_exact(f'bit[4] flags = "{raw:04b}"; int first = 0; int last = 3; int distance = 1; '
                'uint[4] saved = uint[4](~(flags[first:last] << distance)); '
                'flags = bit[4](uint[4](~(flags[first:last] << distance))); '
                'uint[4] fresh = uint[4](~(flags[first:last] >> distance));',
                (('saved', None), ('flags', 4), ('fresh', None)), (expected, expected, (expected >> 1) ^ 15))

    def test_range_checks_precede_counts_and_all_failures_precede_destination_writes(self):
        limits = self.module.QASM2QCXConverter
        cases = (('int first = 0; int last = 2; int zero = 0;', 'flags[first:last] << (1 / zero)', 0, 'assertion failed'),
                 ('int first = -9; int last = 3; int zero = 0;', '~(flags[first:last] >> (1 / zero))', 0, 'assertion failed'),
                 ('int first = 0; int last = 3; int step = 0; int zero = 0;', 'flags[first:step:last] << (1 / zero)', 0, 'assertion failed'),
                 ('int first = 0; int last = 3; int distance = -1;', 'flags[first:last] << distance', 4, 'assertion failed'),
                 (f'int first = 0; int last = 3; int distance = {limits.QCX_INT_MIN};', 'flags[first:last] >> distance', 4, 'assertion failed'),
                 ('int first = 0; int last = 3; int zero = 0;', 'flags[first:last] << (1 / zero)', 4, 'division by zero'),
                 ('int first = 0; int last = 3; int zero = 0;', '(flags[first:last] << 4) >> (1 / zero)', 4, 'division by zero'),
                 ('int first = 0; int last = 3; int zero = 0;', '(flags[first:last] << -1) >> (1 / zero)', 4, 'assertion failed'))
        for constant in (False, True):
            for declarations, operand, read_count, error in cases:
                program = self.module.openqasm3.parser.parse('OPENQASM 3; ' +
                    ('const ' if constant else '') + 'bit[4] flags = "1101"; uint[4] packed = 6; ' +
                    declarations + f'packed = uint[4]({operand});')
                converter = self.module.QASM2QCXConverter(program)
                converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                self.assertEqual(converter._QASM2QCXConverter__captured_bit_register_values, {})
                flags = converter._QASM2QCXConverter__capitalize_variable_name('flags')
                destination = converter._QASM2QCXConverter__capitalize_variable_name('packed')
                lines = list(converter)
                instrumented = []
                for i, line in enumerate(lines):
                    if (f' := {flags}:QASM2QCX_' in line
                            or (' := :INT:QASM2QCX_UINT_' in line and lines[i - 1].endswith(' &= 1'))):
                        instrumented.append('PRINTLN 22222')
                    if line.startswith(f'LET {destination} := QASM2QCX_'):
                        instrumented.append('PRINTLN 11111')
                    instrumented.append(line)
                self.assertIn('PRINTLN 11111', instrumented)
                result = self.execute(instrumented)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), ['22222'] * read_count)
                type(self).checked += 1


    def test_numeric_array_and_register_consumers_preserve_target_widths(self):
        for pattern in range(16):
            raw = int(f'{pattern:04b}'[::-1], 2)
            inverted = raw ^ 15
            left = (raw << 1) & 15
            self.check_exact(f'bit[4] flags = "{pattern:04b}"; int first = 3; int last = 0; int distance = 1; '
                'uint[4] packed = uint[4](~flags[first:-1:last]); '
                'uint[4] combined = uint[4](~flags[first:-1:last]) + uint[4](flags[first:-1:last] << distance); '
                'angle[4] phase = angle[4](bit[4](uint[4](~flags[first:-1:last]))); '
                'uint ones = popcount(bit[4](uint[4](~flags[first:-1:last]))); '
                'array[uint[4],2] values = {uint[4](~flags[first:-1:last]), uint[4](flags[first:-1:last] << distance)}; '
                'values[0] += values[1]; uint[4] element = values[0]; '
                'bit[4] shifted = "1111"; shifted >>= uint[4](~flags[first:-1:last]); '
                f'bool same = flags == "{pattern:04b}";',
                (('packed', None), ('combined', None), ('phase', None), ('ones', None),
                 ('element', None), ('shifted', 4), ('same', None)),
                (inverted, (inverted + left) & 15, inverted, inverted.bit_count(),
                 (inverted + left) & 15, 15 >> inverted, 1))

    def test_nested_range_casts_in_bounds_counts_and_indexed_destinations(self):
        for pattern in range(16):
            for base in (0, 1):
                index = ((pattern >> base) & 3) ^ 3
                count = ((pattern >> base) & 1) ^ 1
                bit = (pattern >> index) & 1
                expected = ((bit << count) & 1) ^ 1
                nested_count = (pattern ^ 15).bit_count()
                self.check_exact(f'bit[4] flags = "{pattern:04b}"; int base = {base}; int first = 0; int last = 3; '
                    'array[uint[4],2] values = {0,0}; '
                    'values[uint[1](~flags[base:base])] = uint[1](~(flags['
                    'uint[2](~flags[base:base + 1]):uint[2](~flags[base:base + 1])] << uint[1](~flags[base:base]))); '
                    'uint[4] out = values[uint[1](~flags[base:base])]; '
                    'uint[4] from_count = uint[4](flags[first:last] << popcount(bit[4](uint[4](~flags[first:last])))); '
                    f'bool same = flags == "{pattern:04b}";',
                    (('out', None), ('from_count', None), ('same', None)),
                    (expected, (pattern << nested_count) & 15, 1))

    def test_while_loops_use_fresh_ranges_counts_and_sources_with_transfers(self):
        for constant in (False, True):
            self.check_exact(('const ' if constant else '') + 'bit[8] flags = "10110101"; '
                'int first = 0; int last = 3; int distance = 0; uint total = 0; uint[4] value = 0; '
                'while (first < 5) { last = first + 3; distance = first % 3; '
                'value = uint[4](~(flags[first:last] << distance)); first += 1; '
                'if (first == 2) { continue; } total += value; if (first == 4) { break; } }',
                (('total', None), ('value', None), ('first', None), ('last', None)), (30, 9, 4, 6))
        self.check_exact('bit[4] flags = "0000"; int first = 0; int last = 3; uint[4] packed = 0; int visits = 0; '
            'while (uint[4](~flags[first:last]) != uint[4](7)) { '
            'packed = uint[4]((~~flags[first:last]) << 0) + uint[4](1); flags = bit[4](packed); visits += 1; }',
            (('visits', None), ('packed', None), ('flags', 4)), (8, 8, 8))

    def test_loop_bounds_sets_and_iterator_shadows_preserve_capture_and_scope(self):
        self.check_exact('const bit[4] C = "1001"; bit[4] flags = "0101"; int first = 0; int last = 2; '
            'int distance = 0; uint total = 0; '
            'for int i in [uint[1](~C[first:first]):uint[3](~(flags[first:last] << distance))] { '
            'total += uint(i); first = 9; last = 9; distance = -1; flags = "0000"; } '
            'array[uint[4],2] indices = {0,3}; for uint[4] first in indices { total += uint[1](~C[first:first]); } '
            'for int first in [-4:-1] { total += uint[1](~C[first:first]); }',
            (('total', None), ('first', None), ('last', None), ('distance', None)), (5, 9, 9, -1))
        self.check_exact('const bit[4] C = "1001"; uint total = 0; '
            'for int outer in [0:1] { '
            'for int i in [uint[1](~C[outer:outer]):uint[2](~C[outer:outer + 1])] { total += uint(i); } '
            'for int i in {uint[1](~C[outer:outer]),uint[2](~C[outer:outer + 1])} { total += uint(i); } }',
            (('total', None),), (15,))

    def test_skipped_paths_and_short_circuiting_suppress_operand_failures(self):
        calls = ('packed = uint[4](~flags[first:0:last] << distance);',
                 'packed = uint[4](~(flags[first:last] << (1 / zero)));',
                 'packed = uint[4]((~C[first:last]) >> positions[bad]);',
                 'packed = uint[4](~(flags[first:last] << 999999999999999999999999));',
                 'packed = uint[4]((flags[first:last] << 4) >> (1 / zero));')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                    'for int outer in [1:0] {{ {} }}', 'for int outer in [loop_first:loop_last] {{ {} }}',
                    'while (visits < 2) {{ visits += 1; continue; {} }}', 'while (true) {{ break; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('bit[4] flags = "1010"; const bit[4] C = "1001"; uint[4] packed = 6; '
                    'array[int,2] positions = {0,1}; int first = 0; int last = -9; int distance = -1; int zero = 0; '
                    'int bad = 3; int visits = 0; int loop_first = 2; int loop_last = 1; ' + wrapper.format(call),
                    (('packed', None), ('flags', 4)), (6, 10))
        for condition, expected in (('false && bool(uint[4](~flags[first:last] << distance))', 0),
                                    ('true || bool(uint[4](flags[first:last] >> (1 / zero)))', 1)):
            self.check_exact('bit[4] flags = "1010"; int first = 0; int distance = -1; int last = 999; int zero = 0; '
                f'bool result = {condition};', (('result', None), ('flags', 4)), (expected, 10))

    def test_later_iteration_failures_preserve_prior_successful_assignments(self):
        for constant in (False, True):
            for update, error in (('distance = -1;', 'assertion failed'), ('last = 2;', 'assertion failed'),
                                  ('divisor = 0;', 'division by zero'), ('step = 0;', 'assertion failed')):
                program = self.module.openqasm3.parser.parse('OPENQASM 3; ' +
                    ('const ' if constant else '') + 'bit[4] flags = "1101"; '
                    'int first = 0; int last = 3; int step = 1; int distance = 1; int divisor = 1; '
                    'int iteration = 0; uint[4] packed = 6; while (iteration < 2) { '
                    'packed = uint[4](~((flags[first:step:last] << distance) >> (1 / divisor))); '
                    f'iteration += 1; {update} }}')
                converter = self.module.QASM2QCXConverter(program)
                converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                self.assertEqual(converter._QASM2QCXConverter__captured_bit_register_values, {})
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
                self.assertEqual(result.stdout.split(), ['11111', '10'])
                type(self).checked += 1

    def test_measurement_derived_chains_drive_feedback_from_live_values(self):
        for anticorrelated in (False, True):
            for distance in (0, 1):
                same = ('bool(back[0]) == !bool(measured[1]) && bool(back[1]) == !bool(measured[0])'
                        if distance == 0 else 'bool(back[0]) == !bool(measured[0]) && bool(back[1])')
                for _ in range(8):
                    self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0],q[1]; '
                        + ('x q[1]; ' if anticorrelated else '') +
                        f'bit[2] measured = measure q; int first = 1; int last = 0; int distance = {distance}; '
                        'uint[2] packed = uint[2](~(measured[first:-1:last] >> distance)); bit[2] back = bit[2](packed); '
                        f'bool same = {same}; '
                        'if (bool(packed)) { x q[0]; } bit[2] observed = measure q; '
                        'bool feedback = bool(observed[0]) == (bool(measured[0]) != bool(packed)) '
                        '&& observed[1] == measured[1]; '
                        f'bool correlation = (measured[0] != measured[1]) == {str(anticorrelated).lower()};',
                        (('same', None), ('feedback', None), ('correlation', None)), (1, 1, 1))

    def test_quantum_parameters_powers_and_broadcasting_with_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = cmath.exp(9j * math.pi / 8)
        for fused in (False, True):
            for gate in ('p(angle[4](bit[4](uint[4](~flags[first:-1:last])))) q;',
                         'pow(uint[4](~flags[first:-1:last])) @ p(pi / 8) q;'):
                source = 'include "stdgates.inc"; qubit[2] q; bit[4] flags = "0110"; int first = 3; int last = 0; h q; ' + gate
                BackendAuditTests.check(self, self.program(source, fused=fused), (0.5, phase / 2, phase / 2, phase * phase / 2))
            source = ('include "stdgates.inc"; qubit q; bit[4] flags = "0110"; int first = 0; int last = 3; int count = 1; '
                      'rx(float(uint[4](~(flags[first:last] << count))) / 4.0) q;')
            BackendAuditTests.check(self, self.program(source, fused=fused), (math.cos(0.375), -1j * math.sin(0.375)))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime range unary/shift UINT cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
