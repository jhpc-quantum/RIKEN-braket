#!/usr/bin/env python3
"""Audit one-dimensional angle-array storage (#418).

Pass --bra /path/to/bra for the default-double non-MPI backend audit.
Inspect exact UINT storage without requiring source-level element access.
"""

import argparse
from fractions import Fraction
import math
import pathlib
import resource
import unittest

import qasm2qcx_math_function_numerical as functions


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    execute = functions.BackendTests.execute

    def check_initializers(self, source, arrays, expected, scalars=(), scalar_expected=()):
        program = self.module.openqasm3.parser.parse('OPENQASM 3; ' + source)
        converter = self.module.QASM2QCXConverter(program)
        converter.visit(program)
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        lines = list(converter)
        for name, length in arrays:
            operand = converter._QASM2QCXConverter__capitalize_variable_name(name)
            for index in range(length):
                lines.append(f'PRINTLN {operand}:{index}')
        for name in scalars:
            operand = converter._QASM2QCXConverter__capitalize_variable_name(name)
            lines.append(f'PRINTLN {operand}')
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + '\n'.join(lines))
        self.assertNotIn('runtime error:', result.stderr)
        self.assertEqual(result.stdout.split(), [str(value) for value in (*expected, *scalar_expected)])
        type(self).checked += 1

    def test_exact_initializers_at_every_width_preserve_sources(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            for bits in sorted({0, 1, modulus // 2, modulus - 1, 0xAAAAAAAA & (modulus - 1)}):
                self.check_initializers(f'const angle[{width}] SEED = angle[{width}]("{bits:0{width}b}"); '
                                        f'angle[{width}] live = SEED; '
                                        f'array[angle[{width}], 4] a = {{SEED, angle[{width}]("{bits:0{width}b}"), live, -live}};',
                                        (('a', 4),), (bits, bits, bits, -bits % modulus), ('live',), (bits,))

    def test_angle_width_conversion_and_unsized_elements(self):
        for width in (1, 4, 8, self.module.QASM2QCXConverter.QCX_UINT_WIDTH):
            expected = [bits << (width - 4) if width >= 4 else bits >> (4 - width) for bits in (3, 13)]
            self.check_initializers('const angle[4] SEED = angle[4]("0011"); angle[4] live = -SEED; '
                                    f'array[angle[{width}], 2] a = {{SEED, live}};', (('a', 2),), expected,
                                    ('live',), (13,))
        width = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        self.check_initializers('const angle[4] SEED = angle[4]("0011"); array[angle, 1] a = {SEED};',
                                (('a', 1),), (3 << (width - 4),))

    def test_float_inputs_quantize_with_scalar_angle_rules(self):
        for width in (1, 4, 8, self.module.QASM2QCXConverter.QCX_UINT_WIDTH):
            tie = math.ldexp(math.tau, -width - 1)
            for value in (0.0, -0.0, math.pi, -math.pi, math.tau, 0.1, -0.1, 1e308,
                          tie, -tie, math.nextafter(tie, 0), math.nextafter(tie, math.inf)):
                expected = round(Fraction(value) / Fraction(math.tau) * (1 << width)) % (1 << width)
                self.check_initializers(f'float x = {value!r}; array[angle[{width}], 1] a = {{x}};',
                                        (('a', 1),), (expected,))

    def test_initializer_order_and_dimension_constant_shadowing(self):
        self.check_initializers('const int W = 4; const int N = 2; '
                                'angle[4] seed = angle[4]("0011"); '
                                'array[angle[W], N] a = {seed, seed + angle[4]("0001")}; '
                                'for int W in [0:1] {} for int N in [0:1] {}',
                                (('a', 2),), (3, 4), ('seed',), (3,))
        self.check_initializers('array[float, 2] radians = {0.0, pi}; int i = 1; '
                                'array[angle[4], 2] a = {radians[i], radians[0]};',
                                (('a', 2),), (8, 0), ('i',), (1,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle-array programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
