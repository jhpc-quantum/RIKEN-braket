#!/usr/bin/env python3
"""Audit angle rotl/rotr/popcount with independent bit-string results (#408).

Run with --bra /path/to/bra (including a UBSan build).
"""

import argparse
import cmath
import math
import pathlib
import re
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
    check_phase = arithmetic.BackendTests.check_phase
    check_zero_qubit = arithmetic.BackendTests.check_zero_qubit

    def test_documented_angle_bit_function_examples(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.md').read_text()
        examples = {}
        for marker in ('angle-bit-function-example', 'angle-bit-function-phase-example'):
            matches = re.findall(r'<!-- ' + marker + r' -->\s*```qasm\n(.*?)\n```', document, flags=re.DOTALL)
            self.assertEqual(len(matches), 1)
            examples[marker] = matches[0]
        self.exact(examples['angle-bit-function-example'],
                   ('a', 'distance', 'left', 'right', 'count', 'same', 'turned_count', 'skipped'),
                   (6, -1, 12, 3, 2, 1, 4, 0))
        for fused in (False, True):
            lines = self.program(examples['angle-bit-function-phase-example'])
            if fused:
                lines.insert(1, 'BEGIN FUSION')
                lines.insert(len(lines) - 1, 'END FUSION')
            result = self.execute(lines)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = {int(index, 2): complex(float(real), float(imaginary))
                      for index, real, imaginary in re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
            self.assertEqual(set(actual), set(range(8)), result.stdout)
            for index in range(8):
                phase = (3 * math.pi / 4 if index & 1 else 0)
                phase += 3 * math.pi / 2 if index & 2 else 0
                phase -= 3 * math.pi / 8 if not index & 4 else 0
                self.assertTrue(cmath.isclose(actual[index], cmath.exp(1j * phase) / math.sqrt(8),
                                             rel_tol=1e-5, abs_tol=2e-6), (index, actual[index]))
            type(self).checked += 1

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

    def test_exhaustive_four_bit_runtime_rotations(self):
        for bits in range(16):
            for distance in range(-8, 9):
                self.exact(f'''angle[4] a; int distance = {distance};
                    angle[4] left = rotl(a, distance); angle[4] right = rotr(a, distance);
                    uint count = popcount(a);''', ('a', 'left', 'right', 'count'),
                    (bits, rotate(bits, 4, distance, 'rotl'), rotate(bits, 4, distance, 'rotr'), bin(bits).count('1')),
                    raw=(('a', bits),))

    def test_self_assignment_counts_and_integer_promotion(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in sorted({1, 4, 8, native}):
            for bits in sorted({0, 1, (1 << width) - 1, ((1 << width) - 1) // 3}):
                count = bin(bits).count('1')
                self.exact(f'''angle[{width}] a;
                    angle[{width}] self = a; self = rotl(self, int(popcount(self)));
                    angle[{width}] back = rotr(self, int(popcount(self)));
                    uint count = popcount(a); count += popcount(self);
                    array[uint, 2] counts = {{0, 0}}; int index = 1; counts[index] = popcount(a);
                    uint stored = counts[index];''', ('a', 'self', 'back', 'count', 'stored'),
                    (bits, rotate(bits, width, count, 'rotl'), bits, count * 2, count), raw=(('a', bits),))
        self.exact('''angle[4] a; uint[4] small = 0; int negative = -1;
            angle[4] result = rotl(a, small + negative);''', ('a', 'result'), (9, 12), raw=(('a', 9),))
        result = self.execute(self.program('angle[4] a; uint native = 0; int negative = -1; '
                                          'angle[4] result = rotl(a, native + negative);'))
        self.assertNotEqual(result.returncode, 0)
        type(self).checked += 1

    def test_iterator_shadowing_and_repeated_calls(self):
        self.exact('''const angle[4] A = pi; angle[4] a; uint count = 0; uint[8] unsigned = 0;
            array[float, 3] radians = {pi / 2, pi, 3 * pi / 2};
            for float A in radians { a = rotl(angle[4](A), 1); count += popcount(a); }
            array[uint[8], 2] integers = {1, 3};
            for uint[8] A in integers { unsigned = rotl(A, 1); count += popcount(A); }
            bool restored = rotr(A, 1) == angle[4](pi / 2);
            int iteration = 0; while (iteration < 3) { a = rotr(a, 1); iteration += 1; }
            for int i in [0:3] { if (i == 1) { continue; } a = rotl(a, 1); if (i == 2) { break; } }''',
            ('a', 'count', 'unsigned', 'restored'), (12, 7, 6, 1))

    def test_measurement_controlled_rotations(self):
        for measured in (0, 1):
            self.exact('''include "stdgates.inc"; qubit q; bit outcome; angle[4] a;
                ''' + ('x q; ' if measured else '') + '''outcome = measure q;
                if (outcome) { a = rotl(a, 1); } else { a = rotr(a, 1); }
                uint count = popcount(a); bool order = a < angle[4](pi);''',
                ('a', 'outcome', 'count', 'order'), (3 if measured else 12, measured, 2, measured), raw=(('a', 9),))

    def test_phase_matrices_population_count_exponents_and_fusion(self):
        prefix = 'angle[4] a = 9 * pi / 8; uint count = popcount(a); a = rotl(a, int(count)); '
        for fused in (False, True):
            for gate in ('p', 'phase', 'u1', 'cp', 'cphase'):
                controlled = gate in ('cp', 'cphase')
                wires, operands = ((2, 0), 'q[2], q[0]') if controlled else ((0,), 'q[0]')
                for basis in (*range(8), None):
                    self.check_phase(prefix + f'{gate}(a) {operands};', 4, 6, (wires,),
                                     fused=fused, basis=basis, preserved=(('a', 6), ('count', 2)))
            self.check_phase(prefix + 'negctrl @ inv @ pow(0.5) @ p(a) q[2], q[0];',
                             5, 26, (((2, 0), 0),), fused=fused)
            self.check_phase(prefix + 'pow(popcount(a)) @ p(a) q[0];', 4, 12, ((0,),), fused=fused)
            self.check_phase(prefix + 'p(angle[4](float(popcount(a)))) q[0];', 4, 5, ((0,),), fused=fused)
            self.check_zero_qubit(prefix + 'pow(0.5) @ gphase(a);', 5, 6, fused=fused)


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
