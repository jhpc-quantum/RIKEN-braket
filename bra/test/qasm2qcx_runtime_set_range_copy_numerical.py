#!/usr/bin/env python3
"""Audit runtime bit-range copies into runtime index-set destinations (#460)."""

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

    def test_mixed_runtime_indices_preserve_order_sources_and_unselected_bits(self):
        for width in (1, 4, 33, 65, 129):
            ranges = {(0, 1, width - 1), (width - 1, -1, max(0, width - 3)),
                      (-1, -1, max(-width, -3)), (0, 2, min(width - 1, 5)),
                      (-1, 1, min(width - 1, 1))}
            for start, step, stop in sorted(ranges):
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                count = len(raw)
                target_width = 2 * count + 1
                before = ('01101001' * target_width)[:target_width]
                odd = tuple(2 * i + 1 for i in range(count))
                layouts = (tuple(range(count)), odd[1:] + odd[:1])
                declarations = ''.join(f'{"uint" if value >= 0 else "int"} {name} = {value};'
                    for name, value in (('start', start), ('step', step), ('stop', stop)))
                for pattern in sorted({'0' * width, '1' * width, ('100110' * width)[:width]}):
                    bits = list(map(int, reversed(pattern)))
                    values = [bits[index % width] for index in raw]
                    for constant in (False, True):
                        for positions in layouts:
                            expected = list(map(int, reversed(before)))
                            for index, value in zip(positions, values):
                                expected[index] = value
                            text = ''.join(map(str, reversed(expected)))
                            signed = ','.join(str(position - target_width) for position in positions)
                            unsigned = ','.join(map(str, positions))
                            entries = ','.join(f'{"signed_indices" if i % 2 == 0 else "unsigned_indices"}[{i}]'
                                               for i in range(count))
                            self.check_initializers(f'{"const " if constant else ""}bit[{width}] source = "{pattern}"; '
                                f'bit[{target_width}] result = "{before}"; '
                                f'array[int, {count}] signed_indices = {{{signed}}}; '
                                f'array[uint, {count}] unsigned_indices = {{{unsigned}}}; ' + declarations +
                                f'result[{{{entries}}}] = source[start:step:stop]; '
                                f'bool same = result == "{text}"; bool unchanged = source == "{pattern}";',
                                (), (), ('same', 'unchanged', 'start', 'step', 'stop'), (1, 1, start, step, stop))

    def test_overlapping_self_copies_snapshot_sources_before_writes(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            for start, step, stop, positions in ((0, 1, 2, (1, 2, 3)),
                                                (3, -1, 0, (2, 0, 3, 1)),
                                                (-1, -1, -4, (0, 1, 2, 3)),
                                                (-1, 4, 3, (3, 0))):
                raw = list(range(start, stop + (1 if step > 0 else -1), step))
                values = [bits[index % 4] for index in raw]
                expected = bits.copy()
                for index, value in zip(positions, values):
                    expected[index] = value
                text = ''.join(map(str, reversed(expected)))
                declarations = ''.join(f'int d{i} = {position - 4};' for i, position in enumerate(positions))
                entries = ','.join(f'd{i}' for i in range(len(positions)))
                self.check_initializers(f'bit[4] flags = "{pattern:04b}"; ' + declarations +
                    f'int start = {start}; int step = {step}; int stop = {stop}; '
                    f'flags[{{{entries}}}] = flags[start:step:stop]; bool same = flags == "{text}";',
                    (), (), ('same',), (1,))

    def test_self_referential_destination_indices_use_original_values(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            positions = (bits[0], bits[1] + 2)
            values = bits[:2]
            expected = bits.copy()
            for index, value in zip(positions, values):
                expected[index] = value
            text = ''.join(map(str, reversed(expected)))
            self.check_initializers(f'bit[4] flags = "{pattern:04b}"; int start = 0; int stop = 1; '
                'flags[{int(flags[0]), int(flags[1]) + 2}] = flags[start:stop]; '
                f'bool same = flags == "{text}";', (), (), ('same',), (1,))

    def test_singletons_and_mixed_constant_entries(self):
        for constant in (False, True):
            for value in ('0', '1'):
                self.check_initializers(f'{"const " if constant else ""}bit[1] source = "{value}"; '
                    'bit[1] one = "0"; bit[8] result = "00000000"; int low = -1; uint high = 7; '
                    'one[{low}] = source[low:low]; result[{high}] = source[low:low]; '
                    f'bool same = one == "{value}" && result == "{value}' + '0' * 7 + '";',
                    (), (), ('same',), (1,))
        self.check_initializers('const bit[4] C = "1010"; bit[8] result = "00000000"; '
            'int index = -1; int start = -1; int stop = 1; '
            'result[{0, index, 3}] = C[start:stop]; bool same = result == "00001001";',
            (), (), ('same', 'index'), (1, -1))

    def test_wide_constant_word_boundaries_and_partial_final_words(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (native, native + 1, 2 * native + 1, 129):
            for position in sorted({0, native - 1, min(native, width - 1), width - 1}):
                pattern = f'{1 << position:0{width}b}'
                self.check_initializers(f'const bit[{width}] C = "{pattern}"; '
                    'bit[129] result = "' + '0' * 129 + f'"; int index = -1; int source_index = {position - width}; '
                    'result[{index}] = C[source_index:source_index]; '
                    'bool same = result == "1' + '0' * 128 + '";', (), (), ('same',), (1,))

    def test_native_source_steps_and_repeated_source_positions(self):
        limits = self.module.QASM2QCXConverter
        for kind, step, start, stop in (('uint', limits.QCX_UINT_MAX, 0, limits.QCX_UINT_MAX - 1),
                                        ('int', limits.QCX_INT_MIN, 0, -4),
                                        ('int', limits.QCX_INT_MAX, -1, 3)):
            expected = '10000000' if start == -1 else '00000000'
            self.check_initializers('bit[4] source = "1010"; bit[8] result = "00000000"; uint index = 7; '
                f'int start = {start}; {kind} step = {step}; {kind} stop = {stop}; '
                f'result[{{index}}] = source[start:step:stop]; bool same = result == "{expected}";',
                (), (), ('same',), (1,))
        self.check_initializers('bit[4] source = "1010"; bit[8] result = "00000000"; '
            'int index = 0; int start = -1; int stop = 3; '
            'result[{index, 1, 2, 3, 4}] = source[start:stop]; bool same = result == "00010101";',
            (), (), ('same',), (1,))

    def test_destination_failures_precede_source_evaluation_and_all_writes(self):
        cases = (('int i = 8;', 'i, 0', 'assertion failed'),
                 ('int i = -9;', '0, i', 'assertion failed'),
                 ('int i = 1;', 'i, i', 'assertion failed'),
                 ('int i = -1;', 'i, 7', 'assertion failed'),
                 (f'uint i = {self.module.QASM2QCXConverter.QCX_UINT_MAX};', 'i, 0',
                  'UINT to INT conversion out of range'),
                 ('int i = 0;', 'i / zero, 1', 'division by zero'),
                 ('array[int, 2] indices = {0, 1}; int i = 2;', 'indices[i], 1', 'assertion failed'))
        for declarations, entries, error in cases:
            lines = self.program('bit[4] source = "1010"; bit[8] result = "11001100"; '
                'int zero = 0; int start = 0; int stop = 1; ' + declarations +
                f'result[{{{entries}}}] = source[start:1 / zero:stop];')
            # A marker before source-step arithmetic proves destination errors
            # take priority, even over an RHS division by zero.
            source_division = max(i for i, line in enumerate(lines) if '/= ZERO15' in line)
            instrumented = []
            for i, line in enumerate(lines):
                if i == source_division:
                    instrumented.append('PRINTLN 12345')
                if line.startswith('LET RESULT63:') and ' := QASM2QCX_RANGE_COPY_SOURCE_' in line:
                    instrumented.append('PRINTLN 67890')
                instrumented.append(line)
            self.assertIn('PRINTLN 12345', instrumented)
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertNotIn('12345', result.stdout)
            self.assertNotIn('67890', result.stdout)
            type(self).checked += 1

    def test_source_failures_after_destination_capture_precede_reads_and_writes(self):
        for declarations, bounds, error in (
                ('int start = 0; int stop = 1; int step = 0;', 'start:step:stop', 'assertion failed'),
                ('int start = 1; int stop = 0;', 'start:stop', 'assertion failed'),
                ('int start = -5; int stop = -4;', 'start:stop', 'assertion failed'),
                ('int start = 3; int stop = 4;', 'start:stop', 'assertion failed'),
                ('int start = 0; int stop = 0;', 'start:stop', 'assertion failed'),
                ('int start = 0; int stop = 2;', 'start:stop', 'assertion failed'),
                ('int start = 0; int stop = 1; int zero = 0;', 'start:1 / zero:stop', 'division by zero')):
            lines = self.program('bit[4] source = "1010"; bit[8] result = "11001100"; int index = -1; '
                + declarations + f'result[{{index, 0}}] = source[{bounds}];')
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
    print(f'runtime set range-copy programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
