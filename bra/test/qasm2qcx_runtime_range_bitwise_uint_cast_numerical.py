#!/usr/bin/env python3
"""Audit binary runtime-range register trees inside UINT casts (#477)."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_bit_cast_numerical as casts
from qasm2qcx_runtime_bit_range_uint_cast_numerical import selected_value


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    check_exact = casts.BackendTests.check_exact

    def test_every_native_width_with_ranges_on_either_or_both_sides(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        size = native + 3
        for width in range(1, native + 1):
            fixed = int(('10100110' * width)[:width], 2)
            for pattern in (0, (1 << size) - 1, int(('10010110' * size)[:size], 2)):
                other = pattern ^ ((1 << size) - 1)
                left = selected_value(pattern, size, 1, 1, width)
                right = selected_value(other, size, width, -1, 1)
                expected = (left & fixed, fixed & right, left & right,
                            left | fixed, fixed | right, left | right,
                            left ^ fixed, fixed ^ right, left ^ right, 1)
                for constant in (False, True):
                    prefix = 'const ' if constant else ''
                    source = (f'{prefix}bit[{size}] left = "{pattern:0{size}b}"; '
                              f'{prefix}bit[{size}] right = "{other:0{size}b}"; '
                              f'const bit[{width}] F = "{fixed:0{width}b}"; int first = 1; int last = {width}; ')
                    outputs = []
                    for operator, mnemonic in (('&', 'and'), ('|', 'or'), ('^', 'xor')):
                        for suffix, lhs, rhs in (('lf', 'left[first:last]', 'F'),
                                                 ('fr', 'F', 'right[last:-1:first]'),
                                                 ('lr', 'left[first:last]', 'right[last:-1:first]')):
                            name = mnemonic + suffix
                            source += f'uint[{width}] {name} = uint[{width}]({lhs} {operator} {rhs}); '
                            outputs.append((name, None))
                    source += f'bool same = left == "{pattern:0{size}b}" && right == "{other:0{size}b}";'
                    self.check_exact(source, (*outputs, ('same', None)), expected)

    def test_supported_fixed_register_operands_and_explicit_numeric_bridges(self):
        prefix = ('bit[8] left = "11010011"; const bit[8] C = "10100110"; bit[8] wide = "10100110"; '
                  'const bit[4] F = "1100"; bit[4] bits = "1010"; uint[4] number = 3; angle[4] a = pi; '
                  'int first = 1; int step = 2; int last = 7; int index = 7; int low = 0; int high = 3;')
        for operand, expected in (('"0101"', 5), ('F', 12), ('bits', 10), ('C[0:3]', 6),
                                  ('C[{7,0,5,1}]', 13), ('wide[{index,0,5,1}]', 13),
                                  ('bit[4](number)', 3), ('bit[4](a)', 8), ('a[0:3]', 8),
                                  ('rotl(F,1)', 9), ('rotr(bits,1)', 5),
                                  ('bit[4](uint[4](C[low:high]))', 6)):
            for lhs, rhs in (('left[first:step:last]', operand), (operand, 'left[first:step:last]')):
                self.check_exact(prefix + f'uint[4] both = uint[4]({lhs} & {rhs}); '
                    f'uint[4] either = uint[4]({lhs} | {rhs}); uint[4] different = uint[4]({lhs} ^ {rhs});',
                    (('both', None), ('either', None), ('different', None), ('left', 8)),
                    (9 & expected, 9 | expected, 9 ^ expected, 211))

    def test_nested_complement_shift_trees_strides_and_negative_aliases(self):
        word = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        cases = ((1, -1, 1, 0, 0, -1, -1), (4, -1, 1, 3, 3, -1, -1),
                 (8, -8, 2, -1, 7, -2, 0), (word + 1, word, -1, word - 3, 0, 1, 3),
                 (129, 128, -3, 119, 0, 1, 3))
        for size, first, step, last, rfirst, rstep, rlast in cases:
            width = len(range(first, last + (1 if step > 0 else -1), step))
            mask = (1 << width) - 1
            pattern = int(('10110010' * size)[:size], 2)
            other = pattern ^ ((1 << size) - 1)
            lhs = selected_value(pattern, size, first, step, last)
            rhs = selected_value(other, size, rfirst, rstep, rlast)
            for left_constant in (False, True):
                for right_constant in (False, True):
                    prefix = (('const ' if left_constant else '') + f'bit[{size}] left = "{pattern:0{size}b}"; '
                              + ('const ' if right_constant else '') + f'bit[{size}] right = "{other:0{size}b}"; '
                              f'int first = {first}; int step = {step}; int last = {last}; '
                              f'int rfirst = {rfirst}; int rstep = {rstep}; int rlast = {rlast}; ')
                    for count in (0, 1, width, self.module.QASM2QCXConverter.QCX_UINT_MAX):
                        lshift = (lhs << count) & mask if count < width else 0
                        rshift = rhs >> count if count < width else 0
                        self.check_exact(prefix + f'uint count = {count}; '
                            f'uint[{width}] value = uint[{width}]((left[first:step:last] << count) & ~right[rfirst:rstep:rlast]); '
                            f'uint[{width}] other_value = uint[{width}](~((left[first:step:last] | right[rfirst:rstep:rlast]) >> count)); '
                            f'uint[{width}] nested = uint[{width}]((~left[first:step:last] | (right[rfirst:rstep:rlast] >> count)) ^ "{mask:0{width}b}");',
                            (('value', None), ('other_value', None), ('nested', None)),
                            (lshift & (rhs ^ mask), ((lhs | rhs) >> count if count < width else 0) ^ mask,
                             ((lhs ^ mask) | rshift) ^ mask))

    def test_unsized_uint_results_preserve_native_high_bits(self):
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        mask = (1 << width) - 1
        for pattern in (0, 1, mask, 1 << (width - 1)):
            other = pattern ^ mask
            self.check_exact(f'bit[{width}] left = "{pattern:0{width}b}"; bit[{width}] right = "{other:0{width}b}"; '
                f'int first = 0; int last = {width - 1}; '
                'uint different = uint(left[first:last] ^ right[first:last]); '
                'uint value = uint(~(left[first:last] & right[first:last])); '
                f'bit[{width}] back = bit[{width}](different);',
                (('different', None), ('value', None), ('back', width)), (mask, mask, mask))

    def test_repeated_accesses_and_overlapping_assignments_use_original_values(self):
        for pattern in range(16):
            reverse = int(f'{pattern:04b}'[::-1], 2)
            expected = pattern ^ reverse
            self.check_exact(f'bit[4] flags = "{pattern:04b}"; int first = 0; int last = 3; '
                'uint[4] repeated = uint[4](flags[first:last] ^ flags[first:last]); '
                'uint[4] saved = uint[4](flags[first:last] ^ flags[last:-1:first]); '
                'flags = bit[4](uint[4](flags[first:last] ^ flags[last:-1:first])); '
                'uint[4] fresh = uint[4](flags[first:last] & "1111");',
                (('repeated', None), ('saved', None), ('flags', 4), ('fresh', None)), (0, expected, expected, expected))

    def test_operand_failures_preserve_order_and_precede_destination_writes(self):
        cases = (('last = 2;', 'left[first:last] ^ right[1 / zero:3]', (0, 0, 0), 'assertion failed'),
                 ('distance = -1;', '(left[first:last] << distance) ^ right[1 / zero:3]', (4, 0, 0), 'assertion failed'),
                 ('last = 2;', 'left[0:high] ^ right[first:last]', (4, 0, 0), 'assertion failed'),
                 ('last = 2;', 'fixed ^ right[first:last]', (0, 0, 4), 'assertion failed'),
                 ('last = 2;', '(left[0:high] << 4) & right[first:last]', (4, 0, 0), 'assertion failed'),
                 ('', 'left[first:last] ^ (right[first:last] << (1 / zero))', (4, 4, 0), 'division by zero'),
                 ('', 'rotl(fixed,1 / zero) ^ right[first:last]', (0, 0, 0), 'division by zero'))
        for update, operand, reads, error in cases:
            program = self.module.openqasm3.parser.parse('OPENQASM 3; bit[4] left = "1101"; bit[4] right = "1010"; '
                'bit[4] fixed = "0011"; int first = 0; int last = 3; int high = 3; int distance = 1; int zero = 0; '
                'uint[4] packed = 6; ' + update + f'packed = uint[4]({operand});')
            converter = self.module.QASM2QCXConverter(program)
            converter.visit(program)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
            self.assertEqual(converter._QASM2QCXConverter__captured_bit_register_values, {})
            names = [converter._QASM2QCXConverter__capitalize_variable_name(name) for name in ('left', 'right', 'fixed')]
            destination = converter._QASM2QCXConverter__capitalize_variable_name('packed')
            instrumented = []
            for line in converter:
                for name, marker in zip(names, (22222, 33333, 44444)):
                    if line.startswith('LET QASM2QCX_') and f' := {name}:' in line:
                        instrumented.append(f'PRINTLN {marker}')
                if line.startswith(f'LET {destination} := QASM2QCX_'):
                    instrumented.append('PRINTLN 11111')
                instrumented.append(line)
            self.assertIn('PRINTLN 11111', instrumented)
            result = self.execute(instrumented)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual(result.stdout.split(), ['22222'] * reads[0] + ['33333'] * reads[1] + ['44444'] * reads[2])
            type(self).checked += 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime range bitwise UINT cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
