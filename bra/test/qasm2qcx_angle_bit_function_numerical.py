#!/usr/bin/env python3
"""Audit angle rotl/rotr/popcount with independent bit-string results (#408).

Run with --bra /path/to/bra (including a UBSan build).
"""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_arithmetic_numerical as arithmetic


def rotate(bits, width, distance, name):
    pattern = f'{bits:0{width}b}'
    offset = (distance if name == 'rotl' else -distance) % width
    return int(pattern[offset:] + pattern[:offset], 2)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arithmetic.BackendTests.setUp
    program = arithmetic.BackendTests.program
    execute = arithmetic.BackendTests.execute
    exact = arithmetic.BackendTests.exact

    def test_all_widths_patterns_and_native_distance_boundaries(self):
        converter = self.module.QASM2QCXConverter
        for width in range(1, converter.QCX_UINT_WIDTH + 1):
            mask = (1 << width) - 1
            for bits in sorted({0, 1, 1 << (width - 1), mask, mask // 3}):
                for distance in sorted({0, 1, -1, width, -width, width * 3 + 1, converter.QCX_INT_MIN, converter.QCX_INT_MAX}):
                    self.exact(f'''angle[{width}] a; int distance = {distance};
                        angle[{width}] left = rotl(a, distance); angle[{width}] right = rotr(a, distance);
                        uint count = popcount(a); uint rotated_count = popcount(rotl(a, distance));''',
                        ('a', 'distance', 'left', 'right', 'count', 'rotated_count'),
                        (bits, distance, rotate(bits, width, distance, 'rotl'), rotate(bits, width, distance, 'rotr'),
                         bin(bits).count('1'), bin(bits).count('1')), raw=(('a', bits),))

    def test_unsized_angles_literal_calls_and_distance_promotions(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        bits = (1 << native) - 2
        self.exact('''angle a; angle result = rotl(a, 1); uint count = popcount(a);''',
                   ('a', 'result', 'count'), (bits, rotate(bits, native, 1, 'rotl'), native - 1), raw=(('a', bits),))
        self.exact('''angle[4] a; uint[4] distance = 15;
            angle[4] result = rotl(a, distance); angle[4] promoted = rotl(a, true);
            angle[4] literal = rotr(angle[4](9 * pi / 8), -1);
            uint count = popcount(angle[4](9 * pi / 8));''',
            ('a', 'distance', 'result', 'promoted', 'literal', 'count'), (9, 15, 12, 3, 3, 2), raw=(('a', 9),))

    def test_checked_uint_distance_conversion_precedes_modulo_and_identity(self):
        for width in (1, 4, self.module.QASM2QCXConverter.QCX_UINT_WIDTH):
            for name in ('rotl', 'rotr'):
                prefix = f'angle[{width}] a; uint distance = {self.module.QASM2QCXConverter.QCX_UINT_MAX}; '
                for distance in ('distance', 'uint(-1)'):
                    expression = f'{name}(a, {distance})'
                    result = self.execute(self.program(prefix + f'angle[{width}] result = {expression};'))
                    self.assertNotEqual(result.returncode, 0, expression)
                    type(self).checked += 1
                    for body in (f'if (false) {{ a = {expression}; }}',
                                 f'bool skipped = false && bool({expression});',
                                 f'for int i in [1:0] {{ a = {expression}; }}'):
                        self.exact(prefix + body, ('a',), (0,))

    def test_operand_errors_are_not_suppressed_by_zero_bits_or_distances(self):
        prefix = 'angle[4] a; int zero = 0; array[int, 2] distances = {1, 2}; int index = 2; '
        for expression, declaration in (('rotl(angle[4](1e309), 0)', 'angle[4] result'),
                                        ('rotr(a, 1 / zero)', 'angle[4] result'),
                                        ('rotl(a, distances[index])', 'angle[4] result'),
                                        ('popcount(angle[4](1e309))', 'uint result')):
            result = self.execute(self.program(prefix + f'{declaration} = {expression};'))
            self.assertNotEqual(result.returncode, 0, expression)
            type(self).checked += 1
            self.exact(prefix + f'bool skipped = true || bool({expression});', ('a',), (0,))

    def test_nested_calls_arithmetic_and_repeated_execution(self):
        self.exact('''angle[4] a; array[int, 3] distances = {1, -1, 4}; int index = 0;
            angle[4] result; uint count;
            while (index < 3) { result = rotl(a, distances[index]); count = popcount(result); index += 1; }
            angle[8] wide = rotl(a, 1) | angle[8](pi / 128);
            uint[4] ratio = rotr(a, 1) / angle[4](pi / 8);
            bool order = rotl(a, 1) < a;''', ('a', 'result', 'count', 'wide', 'ratio', 'order'),
            (9, 9, 2, 49, 12, 1), raw=(('a', 9),))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle bit-function programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
