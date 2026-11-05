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
