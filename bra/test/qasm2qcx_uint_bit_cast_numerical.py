#!/usr/bin/env python3
"""Audit explicit same-width UINT/bit-register casts (#468)."""

import argparse
import pathlib
import random
import resource
import unittest

import qasm2qcx_angle_bit_cast_numerical as angle_casts


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = angle_casts.BackendTests.setUp
    execute = angle_casts.BackendTests.execute
    program = angle_casts.BackendTests.program
    check_exact = angle_casts.BackendTests.check_exact

    def test_all_widths_constant_and_runtime_round_trips_preserve_sources(self):
        random_source = random.Random(468)
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            patterns = (range(modulus) if width <= 6 else
                        sorted({0, 1, modulus - 1, modulus // 2, modulus // 2 - 1}
                               | {random_source.randrange(modulus) for _ in range(4)}))
            for value in patterns:
                self.check_exact(f'const uint[{width}] F = uint[{width}]("{value:0{width}b}"); '
                    f'const bit[{width}] C = bit[{width}](F); '
                    f'bit[{width}] bits = "{value:0{width}b}"; uint[{width}] packed = uint[{width}](bits); '
                    f'bit[{width}] back = bit[{width}](packed); uint[{width}] number = {value}; '
                    f'bit[{width}] unpacked = bit[{width}](number); uint[{width}] again = uint[{width}](unpacked); '
                    f'bit[{width}] constant_bits = C; uint[{width}] folded = F;',
                    (('bits', width), ('packed', None), ('back', width), ('number', None),
                     ('unpacked', width), ('again', None), ('constant_bits', width), ('folded', None)), (value,) * 8)

    def test_static_selection_order_repeated_indices_and_wide_source_positions(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for source_width in (8, native + 1, 65, 129):
            pattern = int(('10010110' * source_width)[:source_width], 2)
            for positions in ((source_width - 1,), (source_width - 1, 0, 3, 1), (0, 0, 1, 1),
                              tuple(range(source_width - 1, source_width - min(native, source_width) - 1, -1))):
                width = len(positions)
                expected = sum(((pattern >> position) & 1) << index for index, position in enumerate(positions))
                selector = '{' + ','.join(map(str, positions)) + '}'
                for constant in (False, True):
                    self.check_exact(('const ' if constant else '') + f'bit[{source_width}] source = "{pattern:0{source_width}b}"; '
                        f'uint[{width}] packed = uint[{width}](source[{selector}]); '
                        f'bit[{width}] back = bit[{width}](packed);',
                        (('packed', None), ('back', width)), (expected, expected))

    def test_overlapping_register_assignments_and_narrow_uint_results(self):
        for value in range(16):
            for selector, positions in (('3:-1:0', (3, 2, 1, 0)), ('{1,3,0,2}', (1, 3, 0, 2))):
                expected = sum(((value >> source) & 1) << destination for source, destination in enumerate(positions))
                self.check_exact(f'bit[4] bits = "{value:04b}"; uint[4] number = {value}; '
                    f'bits[{selector}] = bit[4](uint[4](bits)); '
                    'number = uint[4](bit[4](number + uint[4](1)));',
                    (('bits', 4), ('number', None)), (expected, (value + 1) % 16))
        self.check_exact('const uint[4] A = 14; const uint[4] B = 3; const bit[4] C = bit[4](A + B); '
            'uint[4] a = 14; uint[4] b = 3; bit[4] result = bit[4](a + b); '
            'uint[4] nested = uint[4](bit[4](a + b)) + uint[4](1); bit[4] folded = C; '
            'uint[4] rotated = uint[4](rotl(bit[4](a), 1));',
            (('result', 4), ('nested', None), ('folded', 4), ('rotated', None)), (1, 2, 1, 13))

    def test_unsigned_casts_supply_shift_counts_independent_of_shifted_register_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for count_width in (4, native):
            for count in sorted({0, 1, 7, 8, (1 << count_width) - 1, 1 << (count_width - 1)}):
                self.check_exact(f'bit[{count_width}] count_bits = "{count:0{count_width}b}"; '
                    'bit[8] left = "10001111"; bit[8] right = "10001111"; '
                    f'left <<= uint[{count_width}](count_bits); right >>= uint[{count_width}](count_bits);',
                    (('left', 8), ('right', 8), ('count_bits', count_width)),
                    ((143 << count) & 255 if count < 8 else 0, 143 >> count if count < 8 else 0, count))
        self.check_exact(f'uint source = {self.module.QASM2QCXConverter.QCX_UINT_MAX}; '
                         f'bit[{native}] bits = bit[{native}](source); uint packed = uint(bits);',
                         (('source', None), ('bits', native), ('packed', None)),
                         (self.module.QASM2QCXConverter.QCX_UINT_MAX,) * 3)

    def test_runtime_failures_precede_destination_writes(self):
        cases = (('uint[4] u = 0; bit[4] bits = "1001"; int count = -1; u = uint[4](bits << count);',
                  'U1', 'assertion failed'),
                 ('bit[4] bits = "1001"; array[uint[4],2] values = {1,2}; int index = 2; bits = bit[4](values[index]);',
                  'BITS15:', 'assertion failed'),
                 ('bit[4] bits = "1001"; uint[4] u = 1; uint[4] zero = 0; bits = bit[4](u / zero);',
                  'BITS15:', 'division by zero'))
        for source, destination, error in cases:
            lines = self.program(source)
            instrumented = []
            for line in lines:
                if line.startswith(f'LET {destination}') and ' := QASM2QCX_' in line:
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
    print(f'UINT/bit-register cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
