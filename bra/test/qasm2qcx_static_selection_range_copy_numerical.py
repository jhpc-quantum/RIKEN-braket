#!/usr/bin/env python3
"""Audit runtime bit-range copies into static register selections (#458)."""

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

    def test_static_ranges_and_sets_preserve_order_and_unselected_bits(self):
        for width in (1, 2, 4, 33, 65, 129):
            ranges = {(0, 1, width - 1), (width - 1, -1, 0), (-width, 1, -1),
                      (-1, -1, -width), (0, 2, width - 1), (width - 1, -2, 0),
                      (-1, 1, width - 1), (-width, 1, width - 1)}
            for start, step, stop in sorted(ranges):
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                count = len(raw)
                target_width = 2 * count + 1
                before = ('01101001' * target_width)[:target_width]
                permuted = tuple(2 * i + 1 for i in range(count))
                permuted = permuted[1:] + permuted[:1]
                targets = ((f'0:{count - 1}', tuple(range(count))),
                           (f'-1:-2:{-(2 * count - 1)}', tuple(range(2 * count, 0, -2))),
                           ('{' + ','.join(map(str, permuted)) + '}', permuted))
                declarations = ''.join(
                    f'{"uint" if value >= 0 else "int"} {name} = {value};'
                    for name, value in (('start', start), ('step', step), ('stop', stop)))
                for pattern in sorted({'0' * width, '1' * width, ('100110' * width)[:width]}):
                    source_bits = list(map(int, reversed(pattern)))
                    selected = [source_bits[index % width] for index in raw]
                    for constant in (False, True):
                        for selector, positions in targets:
                            expected = list(map(int, reversed(before)))
                            for index, value in zip(positions, selected):
                                expected[index] = value
                            text = ''.join(map(str, reversed(expected)))
                            self.check_initializers(f'{"const " if constant else ""}bit[{width}] source = "{pattern}"; '
                                f'bit[{target_width}] result = "{before}"; ' + declarations +
                                f'result[{selector}] = source[start:step:stop]; '
                                f'bool same = result == "{text}"; bool unchanged = source == "{pattern}";',
                                (), (), ('same', 'unchanged', 'start', 'step', 'stop'), (1, 1, start, step, stop))

    def test_overlapping_self_copies_use_original_source_bits(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            for start, step, stop, destinations in (
                    (0, 1, 2, ((1, 2, 3), (3, 0, 1))),
                    (3, -1, 0, ((3, 2, 1, 0), (2, 0, 3, 1))),
                    (-1, -1, -4, ((0, 1, 2, 3), (1, 3, 0, 2))),
                    (-1, 4, 3, ((0, 1), (3, 0)))):
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                selected = [bits[index % 4] for index in raw]
                for positions in destinations:
                    expected = bits.copy()
                    for index, value in zip(positions, selected):
                        expected[index] = value
                    text = ''.join(map(str, reversed(expected)))
                    selector = '{' + ','.join(map(str, positions)) + '}'
                    self.check_initializers(f'bit[4] flags = "{pattern:04b}"; '
                        f'int start = {start}; int step = {step}; int stop = {stop}; '
                        f'flags[{selector}] = flags[start:step:stop]; bool same = flags == "{text}";',
                        (), (), ('same',), (1,))

    def test_singleton_storage_default_steps_and_negative_destinations(self):
        for constant in (False, True):
            for value in ('0', '1'):
                for destination in ('one[0:0]', 'one[{-1}]', 'flags[4:4]', 'flags[{-1}]'):
                    expected = ('00010000' if destination == 'flags[4:4]' else '10000000') if value == '1' else '00000000'
                    self.check_initializers(f'{"const " if constant else ""}bit[1] source = "{value}"; '
                        'bit[1] one = "0"; bit[8] flags = "00000000"; int index = -1; '
                        f'{destination} = source[index:index]; '
                        f'bool same = flags == "{expected if destination.startswith("flags") else "00000000"}"; '
                        f'bool singleton = one == "{value if destination.startswith("one") else "0"}";',
                        (), (), ('same', 'singleton'), (1, 1))
        self.check_initializers('bit[8] source = "10101010"; bit[8] result = "00000000"; '
            'array[int, 2] bounds = {1, 3}; int index = 0; '
            'result[5:-2:1] = source[bounds[index]:bounds[index + 1]]; '
            'bool same = result == "00100010";', (), (), ('same',), (1,))

    def test_wide_constant_word_boundaries_and_partial_final_words(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (native - 1, native, native + 1, 2 * native + 1, 129):
            for position in sorted({0, width - 1, min(native - 1, width - 1), min(native, width - 1)}):
                pattern = f'{1 << position:0{width}b}'
                for index in (position, position - width):
                    self.check_initializers(f'const bit[{width}] C = "{pattern}"; bit[129] result = "' + '0' * 129 +
                        f'"; int index = {index}; '
                        'result[{128}] = C[index:index]; '
                        'bool same = result == "1' + '0' * 128 + '";', (), (), ('same',), (1,))

    def test_native_endpoint_strides_and_unselected_stops(self):
        limits = self.module.QASM2QCXConverter
        for kind, step, start, stop in (('uint', limits.QCX_UINT_MAX, 0, limits.QCX_UINT_MAX - 1),
                                        ('int', limits.QCX_INT_MIN, 0, -4),
                                        ('int', limits.QCX_INT_MAX, -1, 3)):
            expected = '10000000' if start == -1 else '00000000'
            self.check_initializers('bit[4] source = "1010"; bit[8] result = "00000000"; '
                f'int start = {start}; {kind} step = {step}; {kind} stop = {stop}; '
                f'result[7:7] = source[start:step:stop]; bool same = result == "{expected}";',
                (), (), ('same', 'step', 'stop'), (1, step, stop))

    def test_invalid_source_ranges_and_counts_fail_before_reads_or_any_destination_write(self):
        cases = (
            ('int start = 0; int stop = 1; int step = 0;', 'start:step:stop', 'assertion failed'),
            ('int start = 1; int stop = 0;', 'start:stop', 'assertion failed'),
            ('int start = -5; int stop = -4;', 'start:stop', 'assertion failed'),
            ('int start = 3; int stop = 4;', 'start:stop', 'assertion failed'),
            ('int start = 0; int stop = 0;', 'start:stop', 'assertion failed'),
            ('int start = 0; int stop = 2;', 'start:stop', 'assertion failed'),
            ('int start = 0; int stop = 1; int zero = 0;', 'start:1 / zero:stop', 'division by zero'),
            ('array[int, 2] bounds = {0, 1}; int index = 2; int stop = 1;', 'bounds[index]:stop', 'assertion failed'))
        for selector in ('0:1', '7:-3:4', '{6, 1}'):
            for declarations, bounds, error in cases:
                lines = self.program('bit[4] source = "1010"; bit[8] result = "11001100"; '
                                     + declarations + f'result[{selector}] = source[{bounds}];')
                instrumented = []
                for line in lines:
                    if (line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_')
                            or (line.startswith('LET RESULT63:') and ' := QASM2QCX_RANGE_COPY_SOURCE_' in line)):
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
    print(f'static selection range-copy programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
