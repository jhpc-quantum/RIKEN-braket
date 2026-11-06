#!/usr/bin/env python3
"""Audit runtime-range rotations inside UINT casts (#479)."""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_bit_cast_numerical as casts
from qasm2qcx_runtime_bit_range_uint_cast_numerical import selected_value


def rotate(value, width, distance):
    """Reference circular permutation of individual little-endian bits."""
    return sum(((value >> index) & 1) << ((index + distance) % width)
               for index in range(width))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = casts.BackendTests.setUp
    execute = casts.BackendTests.execute
    program = casts.BackendTests.program
    check_exact = casts.BackendTests.check_exact

    def test_documented_rotation_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-range-rotation-uint-cast-example -->'
        source = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
        source = source.replace('OPENQASM 3.0;', '', 1)
        self.check_exact(source, (('left', None), ('right', None), ('from_constant', None),
                                 ('restored', None), ('combined', None), ('shifted', None),
                                 ('flags', 8), ('fresh', None)), (3, 12, 7, 9, 1, 4, 220, 9))

    def test_every_native_width_directions_distances_and_constant_sources(self):
        converter = self.module.QASM2QCXConverter
        for width in range(1, converter.QCX_UINT_WIDTH + 1):
            size = width + 2
            pattern = int(('10110010' * size)[:size], 2)
            value = selected_value(pattern, size, 1, 1, width)
            for distance in (0, 1, -1, width, width + 1, converter.QCX_INT_MIN, converter.QCX_INT_MAX):
                for constant in (False, True):
                    source = ('const ' if constant else '') + f'bit[{size}] flags = "{pattern:0{size}b}"; '
                    source += f'int first = 1; int last = {width}; int distance = {distance}; '
                    source += f'uint[{width}] left = uint[{width}](rotl(flags[first:last],distance)); '
                    source += f'uint[{width}] right = uint[{width}](rotr(flags[first:last],distance)); '
                    source += f'uint[{width}] literal = uint[{width}](rotl(flags[first:last],{distance})); '
                    source += f'uint[{width}] back = uint[{width}](rotr(rotl(flags[first:last],distance),distance)); '
                    source += f'bool same = flags == "{pattern:0{size}b}";'
                    self.check_exact(source, (('left', None), ('right', None), ('literal', None), ('back', None), ('same', None)),
                        (rotate(value, width, distance), rotate(value, width, -distance),
                         rotate(value, width, distance), value, 1))

    def test_strided_reverse_negative_indices_and_nested_expression_trees(self):
        for size, first, step, last in ((8, 1, 2, 7), (8, -1, -2, -7), (129, 128, -3, 119)):
            width = len(range(first, last + (1 if step > 0 else -1), step))
            mask = (1 << width) - 1
            pattern = int(('10010110' * size)[:size], 2)
            value = selected_value(pattern, size, first, step, last)
            for constant in (False, True):
                for distance in (-9, 0, 1, 9):
                    source = ('const ' if constant else '') + f'bit[{size}] flags = "{pattern:0{size}b}"; '
                    source += (f'int first = {first}; int step = {step}; int last = {last}; int distance = {distance}; '
                               f'const bit[{width}] MASK = "{mask:0{width}b}"; '
                               f'uint[{width}] nested = uint[{width}](rotr(~rotl(flags[first:step:last] ^ MASK,distance),1)); '
                               f'uint[{width}] shifted = uint[{width}](rotl((flags[first:step:last] << 1) | "{1:0{width}b}",distance)); '
                               f'uint[{width}] both = uint[{width}](rotl(flags[first:step:last],distance) ^ rotr(flags[last:-step:first],distance));')
                    reverse = selected_value(pattern, size, last, -step, first)
                    self.check_exact(source, (('nested', None), ('shifted', None), ('both', None)),
                        (rotate(rotate(value ^ mask, width, distance) ^ mask, width, -1),
                         rotate(((value << 1) & mask) | 1, width, distance),
                         rotate(value, width, distance) ^ rotate(reverse, width, -distance)))

    def test_unsigned_boolean_and_nested_cast_distances(self):
        maximum = self.module.QASM2QCXConverter.QCX_INT_MAX
        for distance in (0, 1, maximum):
            self.check_exact('bit[4] flags = "1101"; int first = 0; int last = 3; '
                f'uint distance = {distance}; bool condition = true; '
                'uint[4] left = uint[4](rotl(flags[first:last],distance)); '
                'uint[4] right = uint[4](rotr(flags[first:last],condition)); '
                'uint[4] nested = uint[4](rotl(flags[first:last],uint[2](rotr(flags[first:first + 1],condition))));',
                (('left', None), ('right', None), ('nested', None)),
                (rotate(13, 4, distance), rotate(13, 4, -1), rotate(13, 4, rotate(1, 2, -1))))

    def test_unsized_uint_and_overlapping_register_destinations(self):
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for pattern in (0, 1, 1 << (width - 1), (1 << width) - 1):
            self.check_exact(f'bit[{width}] flags = "{pattern:0{width}b}"; int first = 0; int last = {width - 1}; '
                'int distance = -1; uint value = uint(rotl(flags[first:last],distance)); '
                f'bit[{width}] back = bit[{width}](value);',
                (('value', None), ('back', width)), (rotate(pattern, width, -1),) * 2)
        for pattern in range(16):
            for distance in (0, 1, -1):
                rotated = rotate(pattern, 4, distance)
                reverse = int(f'{rotated:04b}'[::-1], 2)
                self.check_exact(f'bit[4] flags = "{pattern:04b}"; int first = 0; int last = 3; int distance = {distance}; '
                    'uint[4] original = uint[4](rotl(flags[first:last],distance)); '
                    'flags[{3,2,1,0}] = bit[4](uint[4](rotl(flags[first:last],distance))); '
                    'uint[4] fresh = uint[4](rotr(flags[first:last],distance));',
                    (('original', None), ('flags', 4), ('fresh', None)), (rotated, reverse, rotate(reverse, 4, -distance)))

    def test_arrays_scalar_contexts_and_loop_control_use_fresh_values(self):
        for constant in (False, True):
            pattern = 0b10110101
            expected = 0
            for first in (0, 2, 3):
                expected += rotate((pattern >> first) & 15, 4, first - 2)
            source = ('const ' if constant else '') + 'bit[8] flags = "10110101"; '
            source += ('int first = 0; int last = 3; int distance = -2; uint total = 0; '
                       'array[uint[4],2] values = {0,0}; while (first < 5) { last = first + 3; distance = first - 2; '
                       'values[0] = uint[4](rotl(flags[first:last],distance)); first += 1; '
                       'if (first == 2) { continue; } total += values[0]; if (first == 4) { break; } } '
                       'uint[4] value = values[0];')
            self.check_exact(source, (('total', None), ('value', None), ('first', None)),
                             (expected, rotate((pattern >> 3) & 15, 4, 1), 4))
        self.check_exact('bit[4] flags = "0001"; int first = 0; int last = 3; int distance = 1; int visits = 0; '
            'while (uint[4](rotl(flags[first:last],distance)) != uint[4](1)) { '
            'flags = bit[4](uint[4](rotl(flags[first:last],distance))); visits += 1; }',
            (('visits', None), ('flags', 4)), (3, 8))
        self.check_exact('const bit[4] flags = "1001"; int first = 0; int last = 1; uint total = 0; '
            'for int i in [uint[2](rotl(flags[first:last],1)):uint[2](rotr(flags[first:last],1))] { total += uint(i); } '
            'for int i in {uint[2](rotl(flags[first:last],1)),uint[2](rotr(flags[first:last],1))} { total += uint(i); }',
            (('total', None),), (6,))

    def test_skipped_paths_suppress_runtime_errors(self):
        calls = ('packed = uint[4](rotl(flags[first:last],1 / zero));',
                 'packed = uint[4](rotr(flags[first:last],distance));',
                 'packed = uint[4](rotl(flags[first:0:last],1));')
        wrappers = ('if (false) {{ {} }}', 'while (false) {{ {} }}',
                    'for int outer in [1:0] {{ {} }}', 'for int outer in [loop_first:loop_last] {{ {} }}',
                    'while (true) {{ break; {} }}', 'for int outer in [0:1] {{ continue; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('bit[4] flags = "1010"; uint[4] packed = 6; int first = 0; int last = 999; '
                    'int zero = 0; uint distance = uint(-1); int loop_first = 2; int loop_last = 1; ' + wrapper.format(call),
                    (('packed', None), ('flags', 4)), (6, 10))
        self.check_exact('bit[4] flags = "1010"; int first = 0; int last = 999; int zero = 0; '
            'bool left = false && bool(uint[4](rotl(flags[first:last],1 / zero))); '
            'bool right = true || bool(uint[4](rotr(flags[first:last],uint(-1))));',
            (('left', None), ('right', None)), (0, 1))

    def test_failures_follow_operand_order_and_precede_destination_writes(self):
        maximum = self.module.QASM2QCXConverter.QCX_UINT_MAX
        cases = (('last = 2;', 'rotl(left[first:last],1 / zero)', (0, 0), 'assertion failed'),
                 ('first = 99;', 'rotr(left[first:last],1 / zero)', (0, 0), 'assertion failed'),
                 ('', 'rotl(left[first:0:last],1 / zero)', (0, 0), 'assertion failed'),
                 ('', 'rotl(left[first:last],1 / zero) ^ right[first:last]', (4, 0), 'division by zero'),
                 ('', 'rotr(left[first:last],huge) ^ right[first:last]', (4, 0), 'UINT to INT conversion out of range'),
                 ('rlast = 2;', 'rotl(left[first:last],1) ^ rotr(right[first:rlast],1 / zero)', (4, 0), 'assertion failed'),
                 ('', 'rotl(left[first:last],1) ^ rotr(right[first:rlast],1 / zero)', (4, 4), 'division by zero'))
        for update, operand, reads, error in cases:
            source = ('bit[4] left = "1101"; bit[4] right = "1010"; uint[4] packed = 6; '
                      f'int first = 0; int last = 3; int rlast = 3; int zero = 0; uint huge = {maximum}; '
                      + update + f'packed = uint[4]({operand});')
            program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source)
            converter = self.module.QASM2QCXConverter(program)
            converter.visit(program)
            self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
            names = [converter._QASM2QCXConverter__capitalize_variable_name(name) for name in ('left', 'right')]
            destination = converter._QASM2QCXConverter__capitalize_variable_name('packed')
            instrumented = []
            for line in converter:
                for name, marker in zip(names, (22222, 33333)):
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
            self.assertEqual(result.stdout.split(), ['22222'] * reads[0] + ['33333'] * reads[1])
            type(self).checked += 1

    def test_measurement_feedback_and_quantum_parameters_with_fusion(self):
        for _ in range(16):
            self.check_exact('include "stdgates.inc"; qubit[2] q; reset q; h q[0]; '
                'bit[2] measured = measure q; int first = 0; int last = 1; int distance = 1; '
                'uint[2] packed = uint[2](rotl(measured[first:last],distance)); '
                'bool same = bit[2](packed) == measured[{1,0}]; if (bool(packed)) { x q[1]; } '
                'bit[2] observed = measure q; bool feedback = observed[1] == measured[0] && observed[0] == measured[0];',
                (('same', None), ('feedback', None)), (1, 1))
        import cmath
        import math
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = cmath.exp(3j * math.pi / 8)
        for fused in (False, True):
            for gate in ('p(angle[4](bit[4](uint[4](rotl(flags[first:last],distance))))) q;',
                         'pow(uint[4](rotl(flags[first:last],distance))) @ p(pi / 8) q;'):
                source = ('include "stdgates.inc"; qubit[2] q; bit[4] flags = "1001"; '
                          'int first = 0; int last = 3; int distance = 1; h q; ' + gate)
                BackendAuditTests.check(self, self.program(source, fused=fused), (0.5, phase / 2, phase / 2, phase * phase / 2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime range rotation UINT cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
