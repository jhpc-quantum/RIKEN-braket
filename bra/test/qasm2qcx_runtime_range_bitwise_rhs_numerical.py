#!/usr/bin/env python3
"""Audit whole/static bitwise assignments with runtime range RHS values (#464)."""

import argparse
import operator
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
    operations = (('&=', operator.and_), ('|=', operator.or_), ('^=', operator.xor))

    def check_assignment(self, source_bits, original, positions, selector, raw, constant, operation):
        symbol, function = operation
        expected = original.copy()
        for destination, index in zip(positions, raw):
            expected[destination] = function(original[destination], source_bits[index % len(source_bits)])
        source_text = ''.join(map(str, reversed(source_bits)))
        before = ''.join(map(str, reversed(original)))
        after = ''.join(map(str, reversed(expected)))
        step = raw[1] - raw[0] if len(raw) > 1 else 1
        declarations = ''.join(f'{"uint" if value >= 0 else "int"} {name} = {value}; '
                               for name, value in (('start', raw[0]), ('step', step), ('stop', raw[-1])))
        self.check_initializers(f'{"const " if constant else ""}bit[{len(source_bits)}] source = "{source_text}"; '
            f'bit[{len(original)}] result = "{before}"; ' + declarations +
            f'result{selector} {symbol} source[start:step:stop]; '
            f'bool same = result == "{after}"; bool unchanged = source == "{source_text}";',
            (), (), ('same', 'unchanged'), (1, 1))

    def test_whole_static_ranges_and_sets(self):
        for width in (1, 2, 4, 33, 65, 129):
            source = [int(i % 3 == 0) for i in range(width)]
            for raw in (list(range(width)), list(range(-1, -width - 1, -1)),
                        list(range(0, width, 2)), [-1, width - 1]):
                count = len(raw)
                positions = tuple(range(2 * count, 0, -2))
                targets = ((list(range(count)), '', count),
                           (positions, f'[{2 * count}:-2:2]', 2 * count + 1),
                           (positions[1:] + positions[:1],
                            '[{' + ','.join(map(str, positions[1:] + positions[:1])) + '}]', 2 * count + 1))
                for selected, selector, target_width in targets:
                    original = [i % 2 for i in range(target_width)]
                    for constant in (False, True):
                        for operation in self.operations:
                            self.check_assignment(source, original, selected, selector, raw, constant, operation)

    def test_self_overlap_uses_original_values(self):
        for pattern in range(16):
            bits = [(pattern >> i) & 1 for i in range(4)]
            for raw, positions, selector in (([3, 2, 1, 0], (0, 1, 2, 3), ''),
                                              ([0, 1, 2], (3, 0, 1), '[{3,0,1}]'),
                                              ([-1, 3], (3, 0), '[3:-3:0]')):
                step = raw[1] - raw[0]
                for symbol, function in self.operations:
                    expected = bits.copy()
                    for destination, index in zip(positions, raw):
                        expected[destination] = function(bits[destination], bits[index % 4])
                    after = ''.join(map(str, reversed(expected)))
                    self.check_initializers(f'bit[4] flags = "{pattern:04b}"; '
                        f'int start = {raw[0]}; int step = {step}; int stop = {raw[-1]}; '
                        f'flags{selector} {symbol} flags[start:step:stop]; bool same = flags == "{after}";',
                        (), (), ('same',), (1,))

    def test_singletons_default_steps_and_native_extremes(self):
        limits = self.module.QASM2QCXConverter
        for start, step, stop, kind in ((0, limits.QCX_UINT_MAX, limits.QCX_UINT_MAX - 1, 'uint'),
                                         (0, limits.QCX_INT_MIN, -4, 'int'),
                                         (-1, limits.QCX_INT_MAX, 3, 'int')):
            for symbol, function in self.operations:
                value = function(1, int(start == -1))
                self.check_initializers('const bit[4] C = "1000"; bit[1] one = "1"; '
                    f'int start = {start}; {kind} step = {step}; {kind} stop = {stop}; '
                    f'one[{{-1}}] {symbol} C[start:step:stop]; bool same = one == "{value}";',
                    (), (), ('same',), (1,))
        for constant in (False, True):
            for symbol, function in self.operations:
                for target in ('one', 'one[0:0]', 'one[{-1}]'):
                    self.check_initializers(f'{"const " if constant else ""}bit[1] source = "1"; '
                        'bit[1] one = "1"; int index = -1; '
                        f'{target} {symbol} source[index:index]; bool same = one == "{function(1, 1)}";',
                        (), (), ('same',), (1,))

    def test_invalid_ranges_fail_before_rhs_reads_or_writes(self):
        cases = (('0', '0', '1'), ('1', '1', '0'), ('-5', '1', '-4'),
                 ('3', '1', '4'), ('0', '1', '0'), ('0', '1', '2'))
        for start, step, stop in cases:
            for target in ('result', 'result[{1,0}]'):
                for symbol, _ in self.operations:
                    lines = self.program('bit[4] source = "1010"; bit[2] result = "10"; '
                        f'int start = {start}; int step = {step}; int stop = {stop}; '
                        f'{target} {symbol} source[start:step:stop];')
                    instrumented = []
                    for line in lines:
                        if (line.startswith('LET QASM2QCX_RANGE_COPY_SOURCE_') or
                                (line.startswith('LET RESULT63:') and ' := QASM2QCX_RANGE_BITWISE_ORIGINAL_' in line)):
                            instrumented.append('PRINTLN 12345')
                        instrumented.append(line)
                    self.assertIn('PRINTLN 12345', instrumented)
                    result = self.execute(instrumented)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('assertion failed', result.stderr)
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
    print(f'runtime range bitwise RHS programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
