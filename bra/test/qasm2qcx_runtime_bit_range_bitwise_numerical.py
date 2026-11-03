#!/usr/bin/env python3
"""Audit bitwise compound assignments into runtime bit-register ranges (#450)."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_array_numerical as arrays


def combine(left, right, operator):
    return left & right if operator == '&' else left | right if operator == '|' else left ^ right


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arrays.BackendTests.setUp
    program = arrays.BackendTests.program
    execute = arrays.BackendTests.execute
    check_initializers = arrays.BackendTests.check_initializers

    def test_all_operators_ranges_and_wide_registers_against_bit_list_references(self):
        for width in (1, 4, 33, 65, 129):
            stride = max(1, (width - 1) // 2)
            for start, step, stop in {(0, stride, width - 1), (width - 1, -stride, 0), (-1, 1, -1)}:
                positions = [i % width for i in range(start, stop + (1 if step > 0 else -1), step)]
                count = len(positions)
                for before in sorted({'0' * width, ('101100' * width)[:width]}):
                    for payload in sorted({'0' * count, '1' * count, ('10' * count)[:count]}):
                        for operator in ('&', '|', '^'):
                            bits = list(map(int, reversed(before)))
                            for position, bit in zip(positions, reversed(payload)):
                                bits[position] = combine(bits[position], int(bit), operator)
                            expected = ''.join(map(str, reversed(bits)))
                            for rhs in ('source', f'"{payload}"'):
                                self.check_initializers(f'bit[{width}] flags = "{before}"; '
                                    f'bit[{count}] source = "{payload}"; int start = {start}; '
                                    f'{"uint" if step > 0 else "int"} step = {step}; int stop = {stop}; '
                                    f'flags[start:step:stop] {operator}= {rhs}; bool same = flags == "{expected}";',
                                    (), (), ('same', 'start', 'step', 'stop'), (1, start, step, stop))

    def test_overlapping_explicit_rhs_selections_use_original_bits(self):
        for pattern in range(16):
            for start, step, stop in ((0, 1, 3), (3, -1, 0), (0, 2, 3)):
                positions = list(range(start, stop + (1 if step > 0 else -1), step))
                rhs_positions = list(reversed(positions))
                rhs = '{' + ','.join('index' if i == rhs_positions[0] else str(i) for i in rhs_positions) + '}'
                for operator in ('&', '|', '^'):
                    bits = [(pattern >> i) & 1 for i in range(4)]
                    expected = bits.copy()
                    for destination, source in zip(positions, rhs_positions):
                        expected[destination] = combine(bits[destination], bits[source], operator)
                    text = ''.join(map(str, reversed(expected)))
                    self.check_initializers(f'bit[4] flags = "{pattern:04b}"; int start = {start}; '
                        f'int step = {step}; int stop = {stop}; uint index = {rhs_positions[0]}; '
                        f'flags[start:step:stop] {operator}= flags[{rhs}]; bool same = flags == "{text}";',
                        (), (), ('same',), (1,))

    def test_nested_bounds_and_register_rhs_consumers(self):
        for operator in ('&', '|', '^'):
            for rhs in ('rotl(pair, count)', '~pair', 'a[0:1]', 'C[{0, 3}]',
                        'flags[{index, 3}]', '"01" << count'):
                old = [1, 0, 0, 1]
                for position, value in zip((3, 1), (0, 1)):
                    old[position] = combine(old[position], value, operator)
                expected = ''.join(map(str, reversed(old)))
                self.check_initializers('bit[4] flags = "1001"; bit[2] pair = "01"; '
                    'const bit[4] C = "1000"; angle[2] a = angle[2]("10"); uint count = 1; int index = 2; '
                    'array[int, 2] bounds = {3, 1}; int start = 0; '
                    f'flags[bounds[start]:-2:bounds[start + 1]] {operator}= {rhs}; '
                    f'bool same = flags == "{expected}";', (), (), ('same',), (1,))
        self.check_initializers('bit[4] flags = "1001"; int index = 0; '
            'flags[int(flags[index]):int(flags[3]) + 1:3] ^= "11"; bool same = flags == "0011";',
            (), (), ('same',), (1,))

    def test_full_width_self_operations_and_native_endpoint_steps(self):
        for width in (1, 33, 65, 129):
            before = ('101100' * width)[:width]
            for operator in ('&', '|', '^'):
                expected = '0' * width if operator == '^' else before
                self.check_initializers(f'bit[{width}] flags = "{before}"; int start = 0; '
                    f'int stop = {width - 1}; flags[start:stop] {operator}= flags; '
                    f'bool same = flags == "{expected}";', (), (), ('same',), (1,))
        limits = self.module.QASM2QCXConverter
        for kind, step, start, stop in (('uint', limits.QCX_UINT_MAX, 0, 3),
                                        ('int', limits.QCX_INT_MIN, -1, -4)):
            self.check_initializers(f'bit[4] flags = "1001"; {kind} step = {step}; '
                f'int start = {start}; int stop = {stop}; flags[start:step:stop] ^= "1"; '
                f'bool same = flags == "{"1000" if start == 0 else "0001"}";',
                (), (), ('same',), (1,))

    def test_destination_and_rhs_failures_happen_before_any_write(self):
        cases = (
            ('start:0:stop', '"11"', 'assertion failed'),
            ('start:stop + 1', '"11"', 'assertion failed'),
            ('start:start', '"11"', 'assertion failed'),
            ('stop:start', '"11"', 'assertion failed'),
            ('-1:stop + 2', '"11111"', 'assertion failed'),
            ('bad:stop', '"11"', 'assertion failed'),
            ('start:1 / zero:stop', '"11"', 'division by zero'),
            ('start:stop', 'flags << (1 / zero)', 'assertion failed'),
            ('start:stop + 2', 'flags << (1 / zero)', 'division by zero'),
            ('start:stop', 'flags[{start, bad}]', 'assertion failed'),
        )
        for operator in ('&=', '|=', '^='):
            for selector, rhs, error in cases:
                with self.subTest(operator=operator, selector=selector, rhs=rhs):
                    lines = self.program('bit[4] flags = "1010"; int start = 0; int stop = 1; '
                        f'int bad = 4; int zero = 0; flags[{selector}] {operator} {rhs};')
                    instrumented = []
                    for line in lines:
                        if line.startswith('LET FLAGS31:QASM2QCX_'):
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
    print(f'runtime bit-range bitwise programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
