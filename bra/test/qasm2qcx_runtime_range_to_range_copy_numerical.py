#!/usr/bin/env python3
"""Audit runtime bit-register range-to-range copies (#462)."""

import argparse
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_array_numerical as arrays


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arrays.BackendTests.setUp
    program = arrays.BackendTests.program
    execute = arrays.BackendTests.execute
    check_initializers = arrays.BackendTests.check_initializers

    @staticmethod
    def declarations(values):
        return ''.join(f'{"uint" if value >= 0 else "int"} {name} = {value};' for name, value in values)

    def test_independent_ranges_preserve_order_sources_and_unselected_destination_bits(self):
        for width in (1, 4, 33, 65, 129):
            sources = {(0, 1, width - 1), (width - 1, -1, 0), (-1, -1, -width),
                       (0, 2, width - 1), (-1, 1, width - 1), (-width, 1, width - 1)}
            for sstart, sstep, sstop in sorted(sources):
                raw_source = list(range(sstart, sstop + (1 if sstep > 0 else -1), sstep))
                count = len(raw_source)
                target_width = 2 * count + 3
                before = ('01101001' * target_width)[:target_width]
                destinations = ((1, 1, count), (target_width - 2, -2, 2),
                                (-target_width + 1, 2, -target_width + 1 + 2 * (count - 1)))
                for dstart, dstep, dstop in destinations:
                    raw_destination = list(range(dstart, dstop + (1 if dstep > 0 else -1), dstep))
                    self.assertEqual(len(raw_destination), count)
                    positions = [index % target_width for index in raw_destination]
                    declarations = self.declarations((('dstart', dstart), ('dstep', dstep), ('dstop', dstop),
                                                      ('sstart', sstart), ('sstep', sstep), ('sstop', sstop)))
                    for pattern in sorted({'0' * width, '1' * width, ('100110' * width)[:width]}):
                        bits = list(map(int, reversed(pattern)))
                        values = [bits[index % width] for index in raw_source]
                        expected = list(map(int, reversed(before)))
                        for position, value in zip(positions, values):
                            expected[position] = value
                        text = ''.join(map(str, reversed(expected)))
                        for constant in (False, True):
                            self.check_initializers(f'{"const " if constant else ""}bit[{width}] source = "{pattern}"; '
                                f'bit[{target_width}] result = "{before}"; ' + declarations +
                                'result[dstart:dstep:dstop] = source[sstart:sstep:sstop]; '
                                f'bool same = result == "{text}"; bool unchanged = source == "{pattern}";',
                                (), (), ('same', 'unchanged'), (1, 1))

    def test_overlapping_self_copies_use_original_bits_and_captured_ranges(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            cases = ((1, 1, 3, 0, 1, 2), (3, -1, 0, 0, 1, 3),
                     (0, 1, 3, -1, -1, -4), (-1, -1, -3, 0, 1, 2), (0, 1, 1, -1, 4, 3))
            for dstart, dstep, dstop, sstart, sstep, sstop in cases:
                positions = list(range(dstart, dstop + (1 if dstep > 0 else -1), dstep))
                raw = list(range(sstart, sstop + (1 if sstep > 0 else -1), sstep))
                values = [bits[index % 4] for index in raw]
                expected = bits.copy()
                for position, value in zip(positions, values):
                    expected[position % 4] = value
                text = ''.join(map(str, reversed(expected)))
                declarations = self.declarations((('dstart', dstart), ('dstep', dstep), ('dstop', dstop),
                                                  ('sstart', sstart), ('sstep', sstep), ('sstop', sstop)))
                self.check_initializers(f'bit[4] flags = "{pattern:04b}"; ' + declarations +
                    'flags[dstart:dstep:dstop] = flags[sstart:sstep:sstop]; '
                    f'bool same = flags == "{text}";', (), (), ('same',), (1,))
        self.check_initializers('bit[4] flags = "0101"; int first = 0; int last = 1; '
            'flags[int(flags[first]):int(flags[last]) + 3] = flags[first:last + 1]; '
            'bool same = flags == "1011";', (), (), ('same',), (1,))

    def test_default_steps_singleton_storage_and_repeated_scalar_storage_source(self):
        for constant in (False, True):
            for value in ('0', '1'):
                self.check_initializers(f'{"const " if constant else ""}bit[1] source = "{value}"; '
                    'bit[1] one = "0"; bit[8] result = "00000000"; int low = -1; uint high = 7; '
                    'one[low:low] = source[low:low]; result[high:high] = source[low:low]; '
                    f'bool same = one == "{value}" && result == "{value}' + '0' * 7 + '";',
                    (), (), ('same',), (1,))
                repeated = value * 2
                self.check_initializers(f'{"const " if constant else ""}bit[1] source = "{value}"; '
                    'bit[2] result = "00"; int low = 0; int high = 1; int first = -1; int last = 0; '
                    f'result[low:high] = source[first:last]; bool same = result == "{repeated}";',
                    (), (), ('same',), (1,))

    def test_native_endpoint_steps_and_unselected_stops_on_both_sides(self):
        limits = self.module.QASM2QCXConverter
        steps = (('uint', limits.QCX_UINT_MAX, 0, limits.QCX_UINT_MAX - 1),
                 ('int', limits.QCX_INT_MIN, 0, -4), ('int', limits.QCX_INT_MAX, -1, 3))
        for dkind, dstep, dstart, dstop in steps:
            for skind, sstep, sstart, sstop in steps:
                expected = ['0'] * 4
                expected[dstart % 4] = '1' if sstart == -1 else '0'
                text = ''.join(reversed(expected))
                self.check_initializers('bit[4] source = "1010"; bit[4] result = "0000"; '
                    f'int dstart = {dstart}; {dkind} dstep = {dstep}; {dkind} dstop = {dstop}; '
                    f'int sstart = {sstart}; {skind} sstep = {sstep}; {skind} sstop = {sstop}; '
                    'result[dstart:dstep:dstop] = source[sstart:sstep:sstop]; '
                    f'bool same = result == "{text}";', (), (), ('same',), (1,))

    def test_small_capacity_allows_short_copies_on_wide_registers_and_bounds_all_buffers(self):
        for constant in (False, True):
            with patch.object(self.module.QASM2QCXConverter, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 3):
                self.check_initializers(f'{"const " if constant else ""}bit[129] source = "1' + '0' * 127 + '1"; '
                    'bit[129] result = "' + '0' * 129 + '"; int dstart = -2; int dstop = -1; '
                    'int sstart = 127; uint sstop = 128; result[dstart:dstop] = source[sstart:sstop]; '
                    'bool same = result == "1' + '0' * 128 + '";', (), (), ('same',), (1,))
        for dstart, dstop, sstart, sstop in ((0, 3, 0, 3), (0, 1, 0, 3), (0, 1, 0, 2)):
            with patch.object(self.module.QASM2QCXConverter, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 3):
                lines = self.program('bit[8] source = "10101010"; bit[8] result = "11001100"; '
                    f'int dstart = {dstart}; int dstop = {dstop}; int sstart = {sstart}; int sstop = {sstop}; '
                    'result[dstart:dstop] = source[sstart:sstop];')
            self.assert_failure_before_reads_or_writes(lines, 'assertion failed')

    def assert_failure_before_reads_or_writes(self, lines, error):
        instrumented = []
        for line in lines:
            if (line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_')
                    or (line.startswith('LET RESULT63') and ' := QASM2QCX_RANGE_COPY_SOURCE_' in line)):
                instrumented.append('PRINTLN 67890')
            instrumented.append(line)
        self.assertIn('PRINTLN 67890', instrumented)
        result = self.execute(instrumented)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(error, result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        self.assertNotIn('67890', result.stdout)
        type(self).checked += 1

    def test_destination_errors_precede_source_bound_evaluation(self):
        cases = ((0, 0, 1), (2, 1, 0), (-5, 1, -4), (3, 1, 4), (-1, 4, 3), (-1, 1, 3))
        for dstart, dstep, dstop in cases:
            lines = self.program('bit[4] source = "1010"; bit[4] result = "1100"; '
                f'int dstart = {dstart}; int dstep = {dstep}; int dstop = {dstop}; '
                'int sstart = 0; int sstop = 1; int zero = 0; '
                'result[dstart:dstep:dstop] = source[sstart:1 / zero:sstop];')
            source_division = next(i for i, line in enumerate(lines) if '/= ZERO15' in line)
            lines.insert(source_division, 'PRINTLN 12345')
            instrumented = []
            for line in lines:
                if line.startswith('LET RESULT63:') and ' := QASM2QCX_RANGE_COPY_SOURCE_' in line:
                    instrumented.append('PRINTLN 67890')
                instrumented.append(line)
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('assertion failed', result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertNotIn('12345', result.stdout)
            self.assertNotIn('67890', result.stdout)
            type(self).checked += 1

    def test_source_errors_and_count_mismatches_precede_all_reads_and_writes(self):
        for sstart, sstep, sstop in ((0, 0, 1), (2, 1, 0), (-5, 1, -4), (3, 1, 4),
                                    (0, 1, 0), (0, 1, 2)):
            lines = self.program('bit[4] source = "1010"; bit[4] result = "1100"; '
                'int dstart = 1; int dstop = 2; '
                f'int sstart = {sstart}; int sstep = {sstep}; int sstop = {sstop}; '
                'result[dstart:dstop] = source[sstart:sstep:sstop];')
            self.assert_failure_before_reads_or_writes(lines, 'assertion failed')
        lines = self.program('bit[4] source = "1010"; bit[4] result = "1100"; '
            'int dstart = 1; int dstop = 2; int sstart = 0; int sstop = 1; int zero = 0; '
            'result[dstart:dstop] = source[sstart:1 / zero:sstop];')
        self.assert_failure_before_reads_or_writes(lines, 'division by zero')


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime range-to-range copy programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
