#!/usr/bin/env python3
"""Audit runtime angle bit-range capture (#432) with a non-MPI bra.

Stage 2 exercises the capture helper independently of loop-body lowering.
"""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_array_numerical as arrays


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arrays.BackendTests.setUp
    execute = arrays.BackendTests.execute

    def capture_lines(self, prefix, selection, *, skipped=False, mutate=False):
        program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + prefix)
        converter = self.module.QASM2QCXConverter(program)
        converter.visit(program)
        loop = self.module.openqasm3.parser.parse(
            f'OPENQASM 3; for bit b in {selection} {{}}').statements[0]
        info = converter._QASM2QCXConverter__angle_selection_loop_info(loop.set_declaration, loop.type)
        lines = converter._QASM2QCXConverter__qcx_lines
        previous_temporaries = converter._QASM2QCXConverter__declared_temporary_variables.copy()
        if skipped:
            lines.append('JUMP QASM2QCX_AUDIT_SKIPPED')
        with converter._QASM2QCXConverter__captured_angle_bit_range(info) as (source, positions, count):
            if mutate:
                name = converter._QASM2QCXConverter__capitalize_variable_name
                lines.extend([f'LET {name("element")} := 1', f'LET {name("first")} := 100',
                              f'LET {name("stride")} := 0', f'LET {name("last")} := -100',
                              f'LET {name("phases")}:0 := 9', f'PRINTLN {source}'])
            index = converter._QASM2QCXConverter__add_new_temporary_variable(self.module.ValueType.INT)
            lines.extend([
                f'PRINTLN {count}', f'LET {index} := 0', '@QASM2QCX_AUDIT_PRINT',
                f'JUMPIF QASM2QCX_AUDIT_DONE {index} >= {count}',
                f'PRINTLN {positions}:{index}', f'LET {index} += 1', 'JUMP QASM2QCX_AUDIT_PRINT',
                '@QASM2QCX_AUDIT_DONE',
            ])
        if skipped:
            lines.append('@QASM2QCX_AUDIT_SKIPPED')
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        converter._QASM2QCXConverter__hoist_temporary_declarations(previous_temporaries)
        return list(converter)

    def check_range(self, width, first, stride, last, kinds=('int', 'int', 'int'), *, default_step=False):
        # Independent mathematical oracle: a monotone progression is in bounds
        # iff its first and last selected positions are. Do not iterate a huge
        # invalid range, nor require the stop itself to be a selected position.
        count = 0
        if stride and (last - first) * stride >= 0:
            count = abs(last - first) // abs(stride) + 1
        final = first + (count - 1) * stride if count else first
        valid = count > 0 and -width <= min(first, final) and max(first, final) < width
        prefix = (f'angle[{width}] a; {kinds[0]} first = {first}; '
                  f'{kinds[1]} stride = {stride}; {kinds[2]} last = {last};')
        selector = 'a[first:last]' if default_step else 'a[first:stride:last]'
        lines = self.capture_lines(prefix, selector)
        result = self.execute(lines)
        self.assertNotIn('runtime error:', result.stderr)
        context = f'width={width}, range={first}:{stride}:{last}, types={kinds}\n{result.stderr}'
        if valid:
            self.assertEqual(result.returncode, 0, context)
            expected = [count] + [(first + index * stride) % width for index in range(count)]
            self.assertEqual(result.stdout.split(), [str(value) for value in expected], context)
        else:
            self.assertNotEqual(result.returncode, 0, context)
            self.assertEqual(result.stdout.strip(), '', context)
        type(self).checked += 1

    def test_signed_ranges_match_mathematical_progression_and_bounds(self):
        for width in (1, 4, self.module.QASM2QCXConverter.QCX_UINT_WIDTH):
            for first in sorted({-width, -1, 0, width - 1}):
                for last in sorted({-width - 1, -width, -1, 0, width - 1, width, width + 1}):
                    for stride in sorted({-2 * width - 1, -width, -2, -1, 0, 1, 2, width, 2 * width + 1}):
                        with self.subTest(width=width, first=first, stride=stride, last=last):
                            self.check_range(width, first, stride, last)

    def test_every_width_preserves_order_aliases_and_default_step(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            self.check_range(width, -width, 1, width - 1, default_step=True)
            self.check_range(width, width - 1, -1, -width)
            self.check_range(width, 0, width + 1, width)

    def test_native_signed_extremes_never_overflow(self):
        limits = self.module.QASM2QCXConverter
        for width in (1, 4, limits.QCX_UINT_WIDTH):
            for first in sorted({-width, 0, width - 1}):
                for stride in (limits.QCX_INT_MIN, -1, 1, limits.QCX_INT_MAX):
                    for last in (limits.QCX_INT_MIN, limits.QCX_INT_MAX):
                        self.check_range(width, first, stride, last)

    def test_full_width_unsigned_bounds_and_strides_preserve_their_values(self):
        limits = self.module.QASM2QCXConverter
        for first in (-4, 0, 3):
            for stride in (0, 1, 8, limits.QCX_UINT_MAX - 3, limits.QCX_UINT_MAX):
                for last in (0, 3, 7, limits.QCX_INT_MAX, limits.QCX_UINT_MAX):
                    self.check_range(4, first, stride, last, ('int', 'uint', 'uint'))
        for first in (0, 3, limits.QCX_UINT_MAX):
            for stride in (limits.QCX_INT_MIN, -1, 1, limits.QCX_INT_MAX):
                for last in (0, 3, limits.QCX_UINT_MAX):
                    self.check_range(4, first, stride, last, ('uint', 'int', 'uint'))
        for last in (-5, -1, 0, 3, limits.QCX_INT_MAX):
            self.check_range(4, 0, limits.QCX_UINT_MAX, last, ('uint', 'uint', 'int'))

    def test_capture_precedes_bound_mutations_and_preserves_live_element_address(self):
        lines = self.capture_lines('array[angle[4], 2] phases = {pi, pi}; '
            'int element = 0; int first = -4; int stride = 2; int last = 3;',
            'phases[element][first:stride:last]', mutate=True)
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        self.assertEqual(result.stdout.split(), ['9', '4', '0', '2', '0', '2'])
        type(self).checked += 1

    def test_invalid_captures_fail_only_when_reached_and_before_consumer(self):
        prefix = ('array[angle[4], 2] phases; angle[4] a; int bad = 2; '
                  'int zero = 0; int first = 3; int stride = 0; int last = 0;')
        for selection in ('a[first:last]', 'a[first:stride:last]', 'a[1 / zero:last]',
                          'a[first:1 / zero:last]', 'a[first:1:1 / zero]',
                          'a[0:1:bad + 3]', 'phases[bad][0:last]',
                          'a[0:stride:999999999999999999999]'):
            for skipped in (False, True):
                result = self.execute(self.capture_lines(prefix, selection, skipped=skipped))
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.strip(), '')
                if skipped:
                    self.assertEqual(result.returncode, 0, selection + '\n' + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0, selection)
                type(self).checked += 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f'runtime angle bit-range programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
