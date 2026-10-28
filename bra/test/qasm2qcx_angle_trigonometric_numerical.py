#!/usr/bin/env python3
"""Audit angle trigonometry against a high-precision full-angle series (#410).

Run without arguments to verify converter constant evaluation. The reference
uses independent Machin-formula pi and Decimal Taylor series, without binary
libm calls or the converter's quadrant reduction.
"""

import decimal
import functools
import math
import random
import unittest

from qasm2qcx_runtime_for_numerical import qasm2qcx
from qasm2qcx_trigonometric_numerical import pi


@functools.lru_cache(maxsize=None)
def full_angle_series(bits, width):
    with decimal.localcontext() as context:
        context.prec = 110
        angle = decimal.Decimal(bits) * 2 * pi() / (1 << width)
        squared = -angle * angle
        sine, cosine = angle, decimal.Decimal(1)
        sine_term, cosine_term = sine, cosine
        index = 0
        while True:
            index += 1
            sine_term *= squared / ((2 * index) * (2 * index + 1))
            cosine_term *= squared / ((2 * index - 1) * (2 * index))
            next_sine, next_cosine = sine + sine_term, cosine + cosine_term
            if next_sine == sine and next_cosine == cosine:
                return sine, cosine
            sine, cosine = next_sine, next_cosine


def reference(name, bits, width):
    modulus = 1 << width
    axes = {0: (0.0, 1.0, 0.0), modulus // 2: (0.0, -1.0, 0.0)}
    if width >= 2:
        axes[modulus // 4] = (1.0, 0.0, None)
        axes[3 * modulus // 4] = (-1.0, 0.0, None)
    if bits in axes:
        result = axes[bits][('sin', 'cos', 'tan').index(name)]
        if result is None:
            raise ValueError('exact tangent pole')
        return result
    sine, cosine = full_angle_series(bits, width)
    with decimal.localcontext() as context:
        context.prec = 100
        return float(sine if name == 'sin' else cosine if name == 'cos' else sine / cosine)


class ConstantTests(unittest.TestCase):
    checked = 0

    def setUp(self):
        program = qasm2qcx.openqasm3.parser.parse('OPENQASM 3;')
        self.converter = qasm2qcx.QASM2QCXConverter(program)
        self.converter.visit(program)
        self.expressions = {name: qasm2qcx.ast.FunctionCall(qasm2qcx.ast.Identifier(name), [qasm2qcx.ast.Identifier('A')])
                            for name in ('sin', 'cos', 'tan')}
        self.converter._QASM2QCXConverter__declared_constant_variables.add('A')

    def check(self, bits, width):
        self.converter._QASM2QCXConverter__const_angle_values['A'] = qasm2qcx._ConstantAngle(qasm2qcx._AngleType(width, True), bits)
        self.converter._QASM2QCXConverter__expression_kind = qasm2qcx.ExpressionKind.CONST_ARITHMETIC
        for name, expression in self.expressions.items():
            try:
                expected = reference(name, bits, width)
            except ValueError:
                with self.assertRaises(qasm2qcx.InvalidConstantFunctionException):
                    self.converter.visit(expression)
            else:
                self.converter.visit(expression)
                result = self.converter._QASM2QCXConverter__value
                self.assertEqual(self.converter._QASM2QCXConverter__value_type, qasm2qcx.ValueType.FLOAT)
                self.assertTrue(math.isfinite(result))
                if (4 * bits) % (1 << width) == 0:
                    self.assertEqual(result, expected, (name, width, bits))
                    if expected == 0.0:
                        self.assertEqual(math.copysign(1.0, result), 1.0)
                else:
                    self.assertLessEqual(abs(result - expected), 8 * math.ulp(expected), (name, width, bits, result, expected))
            type(self).checked += 1
        self.assertEqual(self.converter._QASM2QCXConverter__qcx_lines, ['QUBITS 0'])
        self.assertEqual(self.converter._QASM2QCXConverter__used_temporary_variables, set())

    def test_exhaustive_four_bit_angles(self):
        for bits in range(16):
            self.check(bits, 4)

    def test_all_widths_axes_neighbors_octants_and_random_patterns(self):
        random_source = random.Random(410)
        for width in range(1, qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            patterns = {random_source.randrange(modulus) for _ in range(8)}
            for axis in (0, modulus // 4, modulus // 2, 3 * modulus // 4):
                patterns.update((axis + distance) % modulus for distance in (-2, -1, 0, 1, 2))
            patterns.update((octant * modulus // 8) % modulus for octant in range(8))
            for bits in sorted(patterns):
                self.check(bits, width)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ConstantTests))
    print(f'angle trigonometric constant calls checked: {ConstantTests.checked}')
    raise SystemExit(not result.wasSuccessful())
