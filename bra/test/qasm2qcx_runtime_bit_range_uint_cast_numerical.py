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
