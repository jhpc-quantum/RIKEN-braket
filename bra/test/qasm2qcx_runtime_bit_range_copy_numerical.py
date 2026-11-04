#!/usr/bin/env python3
"""Audit copies from runtime bit-register ranges into whole registers (#456)."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_array_numerical as arrays


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arrays.BackendTests.setUp
    program = arrays.BackendTests.program
    execute = arrays.BackendTests.execute
    check_initializers = arrays.BackendTests.check_initializers

    def test_mutable_and_constant_sources_preserve_order_and_source_bits(self):
        for width in (1, 2, 4, 33, 65, 129):
            ranges = {(0, 1, width - 1), (width - 1, -1, 0), (-width, 1, -1),
                      (-1, -1, -width), (0, 2, width - 1), (width - 1, -2, 0),
                      (0, width + 1, width), (-1, 1, width - 1)}
            for start, step, stop in sorted(ranges):
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                positions = [index % width for index in raw]
                for pattern in sorted({'0' * width, '1' * width, ('100110' * width)[:width]}):
                    bits = list(map(int, reversed(pattern)))
                    expected = ''.join(str(bits[index]) for index in reversed(positions))
                    for constant in (False, True):
                        for mixed in (False, True):
                            declarations = ''.join(
                                f'{"uint" if mixed and value >= 0 else "int"} {name} = {value};'
                                for name, value in (('start', start), ('stride', step), ('stop', stop)))
                            self.check_initializers(
                                f'{"const " if constant else ""}bit[{width}] flags = "{pattern}"; '
                                f'bit[{len(raw)}] result; ' + declarations +
                                'result = flags[start:stride:stop]; '
                                f'bool same = result == "{expected}"; bool unchanged = flags == "{pattern}";',
                                (), (), ('same', 'unchanged', 'start', 'stride', 'stop'),
                                (1, 1, start, step, stop))

    def test_in_place_reverse_snapshots_all_bits_before_writes(self):
        for width in (1, 2, 4, 33, 65, 129):
            patterns = ({f'{value:04b}' for value in range(16)} if width == 4
                        else {'0' * width, '1' * width, ('100110' * width)[:width]})
            for pattern in sorted(patterns):
                for start, step, stop in ((width - 1, -1, 0), (-1, -1, -width)):
                    self.check_initializers(f'bit[{width}] flags = "{pattern}"; '
                        f'int start = {start}; int stride = {step}; int stop = {stop}; '
                        'flags = flags[start:stride:stop]; '
                        f'bool same = flags == "{pattern[::-1]}";', (), (), ('same',), (1,))

    def test_default_step_repeated_normalized_positions_and_singletons(self):
        for constant in (False, True):
            prefix = ('const ' if constant else '') + 'bit[4] flags = "1010"; '
            self.check_initializers(prefix + 'bit[5] result; int start = -1; int stop = 3; '
                'result = flags[start:stop]; bool same = result == "10101";', (), (), ('same',), (1,))
            self.check_initializers(prefix + 'bit[8] result; int start = -4; int stop = 3; '
                'result = flags[start:stop]; bool same = result == "10101010";', (), (), ('same',), (1,))
            self.check_initializers(prefix + 'bit[1] result; int start = -1; '
                'result = flags[start:start]; bool same = result == "1";', (), (), ('same',), (1,))
            self.check_initializers(prefix + 'bit[4] result; array[int, 2] bounds = {-4, -1}; int i = 0; '
                'result = flags[bounds[i]:bounds[i + 1]]; bool same = result == "1010";',
                (), (), ('same',), (1,))

    def test_native_signed_unsigned_limits_and_unselected_endpoints(self):
        limits = self.module.QASM2QCXConverter
        for kind, step, start, stop in (('uint', limits.QCX_UINT_MAX, 0, 3),
                                        ('int', limits.QCX_INT_MAX, -1, 3),
                                        ('int', limits.QCX_INT_MIN, 0, -4),
                                        ('uint', limits.QCX_UINT_MAX, 0, limits.QCX_UINT_MAX - 1)):
            expected = '1' if start == -1 else '0'
            self.check_initializers('bit[4] flags = "1010"; bit[1] result; '
                f'{kind} stride = {step}; int start = {start}; {kind} stop = {stop}; '
                f'result = flags[start:stride:stop]; bool same = result == "{expected}";',
                (), (), ('same', 'stride', 'stop'), (1, step, stop))

    def test_bad_ranges_and_lengths_fail_before_any_source_read_or_destination_write(self):
        cases = (
            ('int start = 0; int stop = 3; int step = 0;', 'start:step:stop', 4, 'assertion failed'),
            ('int start = 2; int stop = 0;', 'start:stop', 1, 'assertion failed'),
            ('int start = -5; int stop = -4;', 'start:stop', 2, 'assertion failed'),
            ('int start = 0; int stop = 4;', 'start:stop', 5, 'assertion failed'),
            ('int start = 0; int stop = 1;', 'start:stop', 3, 'assertion failed'),
            ('int start = 0; int stop = 3;', 'start:stop', 3, 'assertion failed'),
            ('int start = -1; int stop = 3;', 'start:stop', 4, 'assertion failed'),
            (f'uint start = {self.module.QASM2QCXConverter.QCX_UINT_MAX}; uint stop = start;',
             'start:stop', 1, 'assertion failed'),
            ('int start = 0; int stop = 1; int zero = 0;', 'start:1 / zero:stop', 2, 'division by zero'),
            ('array[int, 2] bounds = {0, 1}; int i = 2; int stop = 1;',
             'bounds[i]:stop', 2, 'assertion failed'),
        )
        for declarations, bounds, width, error in cases:
            with self.subTest(bounds=bounds, declarations=declarations, width=width):
                lines = self.program('bit[4] flags = "1010"; '
                    f'bit[{width}] result = "' + '1' * width + '"; ' + declarations +
                    f'result = flags[{bounds}];')
                instrumented = []
                for line in lines:
                    if (line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_')
                            or line.startswith('LET RESULT63:QASM2QCX_')
                            or line.startswith('LET RESULT63 := QASM2QCX_')):
                        instrumented.append('PRINTLN 12345')
                    instrumented.append(line)
                self.assertIn('PRINTLN 12345', instrumented)
                result = self.execute(instrumented)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertNotIn('12345', result.stdout)
                type(self).checked += 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime bit-range copy programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
