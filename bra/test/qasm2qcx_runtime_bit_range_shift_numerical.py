#!/usr/bin/env python3
"""Audit shifts within runtime bit-register range destinations (#452)."""

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

    def test_runtime_lengths_directions_counts_and_wide_registers(self):
        for width in (1, 4, 33, 65, 129):
            stride = max(1, (width - 1) // 2)
            for start, step, stop in {(0, 1, width - 1), (-1, -1, -width), (0, stride, width - 1)}:
                positions = [i % width for i in range(start, stop + (1 if step > 0 else -1), step)]
                length = len(positions)
                for before in sorted({'1' * width, ('101100' * width)[:width]}):
                    bits = list(map(int, reversed(before)))
                    selected = [bits[i] for i in positions]
                    for distance in sorted({0, 1, length - 1, length, length + 1,
                                            self.module.QASM2QCXConverter.QCX_UINT_MAX}):
                        for operator in ('<<', '>>'):
                            expected = bits.copy()
                            for i, position in enumerate(positions):
                                source = i - distance if operator == '<<' else i + distance
                                expected[position] = selected[source] if 0 <= source < length else 0
                            text = ''.join(map(str, reversed(expected)))
                            kinds = ('uint', 'int') if distance <= self.module.QASM2QCXConverter.QCX_INT_MAX else ('uint',)
                            for kind in kinds:
                                self.check_initializers(f'bit[{width}] flags = "{before}"; int start = {start}; '
                                    f'int step = {step}; int stop = {stop}; {kind} distance = {distance}; '
                                    f'flags[start:step:stop] {operator}= distance; bool same = flags == "{text}";',
                                    (), (), ('same', 'distance'), (1, distance))

    def test_literal_counts_default_step_and_singleton_storage(self):
        for count in (0, 1, 4, 999999999999999999999999999999999999):
            for operator in ('<<', '>>'):
                expected = '1010' if count == 0 else '0100' if count == 1 and operator == '<<' else '0101' if count == 1 else '0000'
                self.check_initializers('bit[4] flags = "1010"; int start = 0; int stop = 3; '
                    f'flags[start:stop] {operator}= {count}; bool same = flags == "{expected}";',
                    (), (), ('same',), (1,))
        for index in (-1, 0):
            for count in (0, 1, 2):
                self.check_initializers(f'bit[1] flags = "1"; int start = {index}; uint count = {count}; '
                    'flags[start:start] <<= count;', (), (), ('flags',), (int(count == 0),))

    def test_self_referential_counts_and_nested_bound_expressions(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            for operator in ('<<', '>>'):
                distance = bits[0] + bits[3]
                positions = [3, 1]
                expected = bits.copy()
                for i, position in enumerate(positions):
                    source = i - distance if operator == '<<' else i + distance
                    expected[position] = bits[positions[source]] if 0 <= source < 2 else 0
                text = ''.join(map(str, reversed(expected)))
                for count in ('int(flags[low]) + int(flags[3])', 'popcount(flags[{low, 3}])'):
                    self.check_initializers(f'bit[4] flags = "{pattern:04b}"; int index = 0; int low = 0; '
                        'array[int, 2] bounds = {3, 0}; '
                        f'flags[bounds[index]:-2:bounds[index + 1]] {operator}= {count}; '
                        f'bool same = flags == "{text}";', (), (), ('same',), (1,))

    def test_capacity_limits_allow_short_ranges_and_reject_overflow_before_rhs(self):
        with patch.object(self.module.QASM2QCXConverter, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 3):
            self.check_initializers('bit[129] flags = "' + '1' + '0' * 128 + '"; int start = 128; '
                'int stop = 0; int step = -64; flags[start:step:stop] <<= 1; '
                'bool same = flags == "' + '0' * 64 + '1' + '0' * 64 + '";', (), (), ('same',), (1,))
            lines = self.program('bit[4] flags = "1111"; int start = 0; int stop = 3; int zero = 0; '
                                 'flags[start:stop] <<= 1 / zero;')
            result = self.execute(lines)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('assertion failed', result.stderr)
            self.assertNotIn('division by zero', result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            type(self).checked += 1

    def test_normalized_alias_checks_against_independent_sets(self):
        for width in (2, 3, 4, 5, 6):
            for start in (-width, -1, 0, width - 1):
                for stop in (-width, 0, width - 1):
                    for step in sorted({-width, -2, -1, 1, 2, width}):
                        raw = list(range(start, stop + (1 if step > 0 else -1), step))
                        if not raw:
                            continue
                        selected = [value % width for value in raw]
                        source = (f'bit[{width}] flags = "' + '1' * width + '"; '
                                  f'int start = {start}; int step = {step}; int stop = {stop}; '
                                  'flags[start:step:stop] <<= 0;')
                        if len(selected) == len(set(selected)):
                            self.check_initializers(source + 'bool same = flags == "' + '1' * width + '";',
                                                    (), (), ('same',), (1,))
                        else:
                            result = self.execute(self.program(source))
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn('assertion failed', result.stderr)
                            self.assertNotIn('runtime error:', result.stderr)
                            type(self).checked += 1

    def test_native_range_endpoints_and_unselected_stops(self):
        limits = self.module.QASM2QCXConverter
        for kind, step, start, stop in (('uint', limits.QCX_UINT_MAX, 0, 3),
                                        ('int', limits.QCX_INT_MIN, -1, -4)):
            for count in (0, 1):
                expected = '1001' if count == 0 else '1000' if start == 0 else '0001'
                self.check_initializers(f'bit[4] flags = "1001"; {kind} step = {step}; '
                    f'int start = {start}; int stop = {stop}; uint count = {count}; '
                    f'flags[start:step:stop] >>= count; bool same = flags == "{expected}";',
                    (), (), ('same',), (1,))

    def test_range_count_and_arithmetic_failures_precede_all_writes(self):
        cases = (('start:0:stop', '1 / zero', 'assertion failed'),
                 ('stop:start', '1 / zero', 'assertion failed'),
                 ('bad:stop', '1 / zero', 'assertion failed'),
                 ('-2:2:stop + 1', '1 / zero', 'assertion failed'),
                 ('-1:stop + 2', '1 / zero', 'assertion failed'),
                 ('start:1 / zero:stop', '0', 'division by zero'),
                 ('start:stop', '-1', 'assertion failed'),
                 ('start:stop', str(self.module.QASM2QCXConverter.QCX_INT_MIN), 'assertion failed'),
                 ('start:stop', '1 / zero', 'division by zero'),
                 ('start:stop', 'int(flags[bad])', 'assertion failed'))
        for operator in ('<<=', '>>='):
            for selector, count, error in cases:
                with self.subTest(selector=selector, count=count, operator=operator):
                    lines = self.program('bit[4] flags = "1010"; int start = 0; int stop = 1; '
                        f'int bad = 4; int zero = 0; flags[{selector}] {operator} {count};')
                    instrumented = []
                    for line in lines:
                        if line.startswith('LET FLAGS31:QASM2QCX_'):
                            instrumented.append('PRINTLN 12345')
                        instrumented.append(line)
                    self.assertIn('PRINTLN 12345', instrumented)
                    result = self.execute(instrumented)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    if error == 'assertion failed' and '1 / zero' in count:
                        self.assertNotIn('division by zero', result.stderr)
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
    print(f'runtime bit-range shift programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
