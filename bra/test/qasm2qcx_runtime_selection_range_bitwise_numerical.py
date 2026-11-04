#!/usr/bin/env python3
"""Audit runtime-selected bitwise assignments with runtime range RHS values (#466)."""

import argparse
import operator
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
    operations = (('&=', operator.and_), ('|=', operator.or_), ('^=', operator.xor))

    @staticmethod
    def declarations(values):
        return ''.join(f'{"uint" if value >= 0 else "int"} {name} = {value}; ' for name, value in values)

    def test_independent_set_and_range_destinations_preserve_order_and_unselected_bits(self):
        for width in (1, 4, 33, 65, 129):
            for start, step, stop in ((0, 1, width - 1), (width - 1, -1, 0),
                                       (-1, -1, -width), (0, 2, width - 1), (-1, width, width - 1)):
                raw_source = list(range(start, stop + (1 if step > 0 else -1), step))
                count = len(raw_source)
                target_width = 2 * count + 3
                before = ('01101001' * target_width)[:target_width]
                source = ('100110' * width)[:width]
                source_bits = list(map(int, reversed(source)))
                selected = [source_bits[index % width] for index in raw_source]
                descending = tuple(range(target_width - 2, target_width - 2 - 2 * count, -2))
                permuted = descending[1:] + descending[:1]
                targets = (('result[dstart:dstep:dstop]', descending,
                            self.declarations((('dstart', descending[0]), ('dstep', -2), ('dstop', descending[-1])))),
                           ('result[dstart:dstep:dstop]', tuple(range(1, 2 * count, 2)),
                            self.declarations((('dstart', -target_width + 1), ('dstep', 2), ('dstop', -4)))),
                           ('result[{index' + ''.join(f', {i}' for i in permuted[1:]) + '}]', permuted,
                            f'int index = {permuted[0] - target_width}; '))
                for target, positions, declarations in targets:
                    for constant in (False, True):
                        for symbol, function in self.operations:
                            expected = list(map(int, reversed(before)))
                            for destination, bit in zip(positions, selected):
                                expected[destination] = function(expected[destination], bit)
                            after = ''.join(map(str, reversed(expected)))
                            self.check_initializers(('const ' if constant else '') +
                                f'bit[{width}] source = "{source}"; bit[{target_width}] result = "{before}"; '
                                + declarations + self.declarations((('start', start), ('step', step), ('stop', stop))) +
                                f'{target} {symbol} source[start:step:stop]; '
                                f'bool same = result == "{after}"; bool unchanged = source == "{source}";',
                                (), (), ('same', 'unchanged'), (1, 1))

    def test_self_referential_destination_indices_and_rhs_bounds_use_original_values(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            for target, positions in (('flags[{int(flags[0]), int(flags[1]) + 2}]', (bits[0], bits[1] + 2)),
                                       ('flags[int(flags[0]):int(flags[0]) + 1]', (bits[0], bits[0] + 1))):
                for symbol, function in self.operations:
                    expected = bits.copy()
                    for destination, source in zip(positions, (bits[2], bits[2] + 1)):
                        expected[destination] = function(bits[destination], bits[source])
                    after = ''.join(map(str, reversed(expected)))
                    self.check_initializers(f'bit[4] flags = "{pattern:04b}"; '
                        f'{target} {symbol} flags[int(flags[2]):int(flags[2]) + 1]; '
                        f'bool same = flags == "{after}";', (), (), ('same',), (1,))

    def test_singleton_storage_native_extreme_strides_and_small_position_limits(self):
        limits = self.module.QASM2QCXConverter
        cases = ((0, limits.QCX_UINT_MAX, limits.QCX_UINT_MAX - 1, 'uint'),
                 (0, limits.QCX_INT_MIN, -4, 'int'), (-1, limits.QCX_INT_MAX, 3, 'int'))
        for target in ('one[{index}]', 'one[index:index]'):
            for start, step, stop, kind in cases:
                for symbol, function in self.operations:
                    expected = function(1, int(start == -1))
                    self.check_initializers('const bit[4] C = "1000"; bit[1] one = "1"; int index = -1; '
                        f'int start = {start}; {kind} step = {step}; {kind} stop = {stop}; '
                        f'{target} {symbol} C[start:step:stop]; bool same = one == "{expected}";',
                        (), (), ('same',), (1,))
        for target in ('result[{first, last}]', 'result[first:128:last]'):
            for symbol, function in self.operations:
                with patch.object(limits, 'MAX_RUNTIME_BIT_RANGE_POSITIONS', 2):
                    expected = function(0, 1) * ((1 << 128) | 1)
                    self.check_initializers('const bit[129] C = "' + '1' + '0' * 127 + '1"; '
                        'bit[129] result = "' + '0' * 129 + '"; int first = 0; uint last = 128; '
                        f'{target} {symbol} C[first:128:last]; bool same = result == "{expected:0129b}";',
                        (), (), ('same',), (1,))

    def assert_failure(self, source, error, original_reads):
        lines = self.program(source)
        instrumented = []
        for line in lines:
            if line.startswith('LET QASM2QCX_RANGE_BITWISE_ORIGINAL_') and ' := RESULT63' in line:
                instrumented.append('PRINTLN 12345')
            if (line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_') or
                    (line.startswith('LET RESULT63:') and ' := QASM2QCX_RANGE_BITWISE_ORIGINAL_' in line)):
                instrumented.append('PRINTLN 67890')
            instrumented.append(line)
        self.assertIn('PRINTLN 12345', instrumented)
        self.assertIn('PRINTLN 67890', instrumented)
        result = self.execute(instrumented)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(error, result.stderr)
        self.assertNotIn('runtime error:', result.stderr)
        self.assertEqual(result.stdout.split(), ['12345'] * original_reads)
        type(self).checked += 1

    def test_destination_failures_precede_original_reads_and_rhs_division(self):
        targets = ('result[{index, 0}]', 'result[index:step:stop]')
        cases = ((-5, 1, -4), (4, 1, 5), (0, 0, 1), (1, -1, 2), (-1, 4, 3))
        for symbol, _ in self.operations:
            for target in targets:
                for index, step, stop in cases:
                    # Sets need a bad/duplicate entry; step/stop are irrelevant.
                    if '{' in target and index not in (-5, 4):
                        index = 0
                    self.assert_failure('const bit[4] C = "1010"; bit[4] result = "1100"; '
                        f'int index = {index}; int step = {step}; int stop = {stop}; int start = 0; int zero = 0; '
                        f'{target} {symbol} C[start:1 / zero:stop];', 'assertion failed', 0)

    def test_source_range_failures_allow_original_snapshot_but_no_rhs_reads_or_writes(self):
        cases = (('0', '0', '1'), ('1', '1', '0'), ('-5', '1', '-4'),
                 ('3', '1', '4'), ('0', '1', '0'), ('0', '1', '2'))
        for symbol, _ in self.operations:
            for target in ('result[{left, right}]', 'result[left:right]'):
                for start, step, stop in cases:
                    self.assert_failure('bit[4] source = "1010"; bit[4] result = "1100"; '
                        'int left = 1; int right = 2; '
                        f'int start = {start}; int step = {step}; int stop = {stop}; '
                        f'{target} {symbol} source[start:step:stop];', 'assertion failed', 2)
                self.assert_failure('bit[4] source = "1010"; bit[4] result = "1100"; '
                    'int left = 1; int right = 2; int start = 0; int stop = 1; int zero = 0; '
                    f'{target} {symbol} source[start:1 / zero:stop];', 'division by zero', 2)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime selection range bitwise programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
