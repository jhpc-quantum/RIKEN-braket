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
    program = casts.BackendTests.program
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


    def test_numeric_array_and_register_consumers_preserve_target_widths(self):
        for pattern in range(16):
            base = pattern ^ 9
            value = base | 2
            self.check_exact(f'bit[4] flags = "{pattern:04b}"; const bit[4] B = "1001"; int first = 0; int last = 3; '
                'uint[4] base_value = uint[4](flags[first:last] ^ B); '
                'uint[4] value = uint[4]((flags[first:last] ^ B) | "0010"); '
                'uint[4] added = base_value + value; '
                'array[uint[4],2] values = {uint[4](flags[first:last] & B),uint[4](flags[first:last] | B)}; '
                'values[0] ^= values[1]; uint[4] element = values[0]; '
                'angle[4] phase = angle[4](bit[4](uint[4](flags[first:last] ^ B))); '
                'uint ones = popcount(bit[4](uint[4](flags[first:last] | B))); '
                'bit[4] shifted = "1111"; shifted >>= uint[4](flags[first:last] ^ B); '
                f'bool same = flags == "{pattern:04b}";',
                (('base_value', None), ('value', None), ('added', None), ('element', None),
                 ('phase', None), ('ones', None), ('shifted', 4), ('same', None)),
                (base, value, (base + value) & 15, base, base, (pattern | 9).bit_count(), 15 >> base, 1))

    def test_nested_trees_in_bounds_counts_and_indexed_destinations(self):
        for pattern in range(16):
            for base in (0, 1):
                index = ((pattern >> base) & 3) ^ 3
                source_bit = (pattern >> base) & 1
                expected = ((pattern >> index) & 1) ^ source_bit
                self.check_exact(f'bit[4] flags = "{pattern:04b}"; int base = {base}; int first = 0; int last = 3; '
                    'array[uint[4],2] values = {0,0}; '
                    'values[uint[1](flags[base:base] ^ "1")] = uint[1](flags['
                    'uint[2](flags[base:base + 1] ^ "11"):uint[2](flags[base:base + 1] ^ "11")] ^ flags[base:base]); '
                    'uint[4] out = values[uint[1](flags[base:base] ^ "1")]; '
                    'uint[4] shifted = uint[4]((flags[first:last] ^ "0011") << uint[1](flags[base:base] & "1")); '
                    f'bool same = flags == "{pattern:04b}";',
                    (('out', None), ('shifted', None), ('same', None)),
                    (expected, ((pattern ^ 3) << source_bit) & 15, 1))

    def test_while_loops_use_fresh_operands_and_counts_with_transfers(self):
        for constant in (False, True):
            self.check_exact(('const ' if constant else '') + 'bit[8] flags = "10110101"; const bit[8] C = "01101001"; '
                'int first = 0; int last = 3; int distance = 0; uint total = 0; uint[4] value = 0; '
                'while (first < 5) { last = first + 3; distance = first % 3; '
                'value = uint[4](~((flags[first:last] ^ C[first:last]) << distance)); first += 1; '
                'if (first == 2) { continue; } total += value; if (first == 4) { break; } }',
                (('total', None), ('value', None), ('first', None), ('last', None)), (10, 4, 4, 6))
        self.check_exact('bit[4] flags = "0000"; int first = 0; int last = 3; uint[4] packed = 0; int visits = 0; '
            'while (uint[4](flags[first:last] ^ "0000") != uint[4](8)) { '
            'packed = uint[4](flags[first:last] ^ "0000") + uint[4](1); flags = bit[4](packed); visits += 1; }',
            (('visits', None), ('packed', None), ('flags', 4)), (8, 8, 8))

    def test_loop_bounds_sets_and_iterator_shadows_preserve_capture_and_scope(self):
        self.check_exact('const bit[4] C = "1001"; bit[4] flags = "0101"; int first = 0; int last = 2; uint total = 0; '
            'for int i in [uint[1](C[first:first] ^ "1"):uint[3](flags[first:last] & "011")] { '
            'total += uint(i); first = 9; last = 9; flags = "0000"; } '
            'array[uint[4],2] indices = {0,3}; for uint[4] first in indices { total += uint[1](C[first:first] ^ "1"); } '
            'for int first in [-4:-1] { total += uint[1](C[first:first] ^ "1"); }',
            (('total', None), ('first', None), ('last', None)), (3, 9, 9))
        self.check_exact('const bit[4] C = "1001"; uint total = 0; '
            'for int outer in [0:1] { '
            'for int i in [uint[1](C[outer:outer] ^ "1"):uint[2](C[outer:outer + 1] ^ "11")] { total += uint(i); } '
            'for int i in {uint[1](C[outer:outer] ^ "1"),uint[2](C[outer:outer + 1] ^ "11")} { total += uint(i); } }',
            (('total', None),), (15,))

    def test_skipped_paths_and_short_circuiting_suppress_runtime_failures(self):
        calls = ('packed = uint[4]((flags[first:last] << distance) ^ C[first:last]);',
                 'packed = uint[4](flags[first:last] & C[1 / zero:last]);',
                 'packed = uint[4](flags[first:last] | wide[{index,999,-999,index}]);',
                 'packed = uint[4]("0000" & flags[first:0:last]);',
                 'packed = uint[4](~((flags[first:last] ^ C[first:last]) >> (1 / zero)));')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                    'for int outer in [1:0] {{ {} }}', 'for int outer in [loop_first:loop_last] {{ {} }}',
                    'while (visits < 2) {{ visits += 1; continue; {} }}', 'while (true) {{ break; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('bit[4] flags = "1010"; const bit[4] C = "1001"; bit[8] wide = "10100110"; uint[4] packed = 6; '
                    'int first = 0; int last = -9; int distance = -1; int zero = 0; int index = 0; '
                    'int visits = 0; int loop_first = 2; int loop_last = 1; ' + wrapper.format(call),
                    (('packed', None), ('flags', 4)), (6, 10))
        for condition, expected in (('false && bool(uint[4](flags[first:last] ^ "0000"))', 0),
                                    ('true || bool(uint[4](flags[first:last] & C[1 / zero:last]))', 1)):
            self.check_exact('bit[4] flags = "1010"; const bit[4] C = "1001"; int first = 0; int last = 999; int zero = 0; '
                f'bool result = {condition};', (('result', None), ('flags', 4)), (expected, 10))

    def test_later_operand_failures_preserve_prior_successful_assignments(self):
        for constant in (False, True):
            for update, error in (('last = 2;', 'assertion failed'), ('distance = -1;', 'assertion failed'),
                                  ('rlast = 2;', 'assertion failed'), ('divisor = 0;', 'division by zero')):
                prefix = 'const ' if constant else ''
                program = self.module.openqasm3.parser.parse('OPENQASM 3; ' +
                    prefix + 'bit[4] left = "1101"; ' + prefix + 'bit[4] right = "1010"; '
                    'int first = 0; int last = 3; int rlast = 3; int distance = 1; int divisor = 1; '
                    'int iteration = 0; uint[4] packed = 6; while (iteration < 2) { '
                    'packed = uint[4]((left[first:last] << distance) ^ (right[first:rlast] >> (1 / divisor))); '
                    f'iteration += 1; {update} }}')
                converter = self.module.QASM2QCXConverter(program)
                converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
                self.assertEqual(converter._QASM2QCXConverter__captured_bit_register_values, {})
                destination = converter._QASM2QCXConverter__capitalize_variable_name('packed')
                instrumented = []
                for line in converter:
                    if line.startswith(f'LET {destination} := QASM2QCX_'):
                        instrumented.extend(['PRINTLN 11111', line, f'PRINTLN {destination}'])
                    else:
                        instrumented.append(line)
                self.assertIn('PRINTLN 11111', instrumented)
                result = self.execute(instrumented)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout.split(), ['11111', '15'])
                type(self).checked += 1

    def test_measurement_derived_trees_drive_classical_feedback(self):
        for anticorrelated in (False, True):
            for _ in range(16):
                self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; cx q[0],q[1]; '
                    + ('x q[1]; ' if anticorrelated else '') +
                    'bit[2] measured = measure q; int first = 0; int last = 1; '
                    'uint[2] packed = uint[2](measured[first:last] ^ measured[last:-1:first]); '
                    f'bool same = packed == uint[2]({3 if anticorrelated else 0}); '
                    'uint[2] preserved = uint[2](measured[first:last] & "11"); '
                    'bool bits_same = bit[2](preserved) == measured; '
                    'if (bool(packed)) { x q[0]; } bit[2] observed = measure q; '
                    'bool feedback = bool(observed[0]) == (bool(measured[0]) != bool(packed)) '
                    '&& observed[1] == measured[1];',
                    (('same', None), ('bits_same', None), ('feedback', None)), (1, 1, 1))

    def test_quantum_parameters_powers_and_broadcasting_with_fusion(self):
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = cmath.exp(5j * math.pi / 8)
        for fused in (False, True):
            for gate in ('p(angle[4](bit[4](uint[4](flags[first:last] ^ other[first:last])))) q;',
                         'pow(uint[4](flags[first:last] ^ other[first:last])) @ p(pi / 8) q;'):
                source = ('include "stdgates.inc"; qubit[2] q; bit[4] flags = "0110"; const bit[4] other = "0011"; '
                          'int first = 0; int last = 3; h q; ' + gate)
                BackendAuditTests.check(self, self.program(source, fused=fused), (0.5, phase / 2, phase / 2, phase * phase / 2))
            source = ('include "stdgates.inc"; qubit q; bit[4] flags = "0110"; const bit[4] other = "0011"; '
                      'int first = 0; int last = 3; int count = 1; '
                      'rx(float(uint[4](~((flags[first:last] ^ other[first:last]) << count))) / 4.0) q;')
            BackendAuditTests.check(self, self.program(source, fused=fused), (math.cos(0.625), -1j * math.sin(0.625)))


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
