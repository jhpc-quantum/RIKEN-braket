#!/usr/bin/env python3
"""Audit angle trigonometry against a high-precision full-angle series (#410).

Run without arguments to verify converter constant evaluation, or pass
--bra /path/to/bra to audit the default-double non-MPI backend. The reference
uses independent Machin-formula pi and Decimal Taylor series, without binary
libm calls or the converter's quadrant reduction.
"""

import argparse
import cmath
import decimal
import functools
import math
import random
import pathlib
import resource
import unittest

import qasm2qcx_math_function_numerical as functions
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


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    execute = functions.BackendTests.execute
    program = functions.BackendTests.program
    check = functions.BackendTests.check

    def raw_program(self, bits, width, names):
        # Inject exact UINT storage after declaring the angle, independently
        # of float-to-angle quantization and any source-language casts.
        source = f'OPENQASM 3; angle[{width}] a; '
        for name in names:
            expected = reference(name, bits, width)
            scale = abs(expected) if expected else 1.0
            source += f'float {name}_out = {name}(a); {name}_out /= {scale!r}; '
        program = qasm2qcx.openqasm3.parser.parse(source)
        converter = qasm2qcx.QASM2QCXConverter(program)
        for index, statement in enumerate(program.statements):
            converter.visit(statement)
            if index == 0:
                operand = converter._QASM2QCXConverter__capitalize_variable_name('a')
                converter._QASM2QCXConverter__qcx_lines.append(f'LET {operand} := {bits}')
        self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables, set())
        lines = list(converter)
        for name in names:
            operand = converter._QASM2QCXConverter__capitalize_variable_name(f'{name}_out')
            lines.append(f'PRINTLN {operand}')
        lines.append(f'PRINTLN {converter._QASM2QCXConverter__capitalize_variable_name("a")}')
        return lines

    def test_all_widths_axes_neighbors_octants_and_source_preservation(self):
        random_source = random.Random(410)
        for width in range(1, qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            patterns = set(range(modulus)) if width <= 6 else {random_source.randrange(modulus) for _ in range(8)}
            for axis in (0, modulus // 4, modulus // 2, 3 * modulus // 4):
                patterns.update((axis + distance) % modulus for distance in (-2, -1, 0, 1, 2))
            patterns.update(octant * modulus // 8 for octant in range(8))
            for bits in sorted(patterns):
                names = ('sin', 'cos') if width >= 2 and bits in (modulus // 4, 3 * modulus // 4) else ('sin', 'cos', 'tan')
                result = self.execute(self.raw_program(bits, width, names))
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = list(map(float, result.stdout.split()))
                self.assertEqual(len(actual), len(names) + 1)
                self.assertEqual(actual[-1], bits)
                for name, value in zip(names, actual):
                    expected = reference(name, bits, width)
                    normalized = math.copysign(1.0, expected) if expected else 0.0
                    self.assertAlmostEqual(value, normalized, delta=2e-6, msg=(name, width, bits))
                    if expected == 0:
                        self.assertEqual(math.copysign(1.0, value), 1.0)
                type(self).checked += 1

    def test_reached_poles_fail_and_skipped_poles_do_not_execute(self):
        for width in (2, 4, qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH):
            for angle in ('pi / 2', '3 * pi / 2'):
                source = f'angle[{width}] a = {angle}; float out = tan(a);'
                result = self.execute(self.program(source))
                self.assertNotEqual(result.returncode, 0)
                type(self).checked += 1
                self.check(f'angle[{width}] a = {angle}; bool out = false && bool(tan(a));',
                           (('out', False),), (0,))
                self.check(f'angle[{width}] a = {angle}; bool out = true || bool(tan(a));',
                           (('out', False),), (1,))

    def test_executable_literals_constants_and_compound_angle_arguments(self):
        for name in ('sin', 'cos', 'tan'):
            for expression in ('angle[4](3 * pi / 8)', 'A', 'a + angle[4](pi / 8)'):
                expected = reference(name, 3, 4)
                self.check('const angle[4] A = 3 * pi / 8; angle[4] a = pi / 4; '
                           f'float out = {name}({expression});', (('out', False), ('a', False)),
                           (expected, 2))
            self.check(f'angle a = pi / 2; float out = {name}(a / uint(2));',
                       (('out', False),), (reference(name, 1, 3),))

    def test_nested_calls_destinations_and_angle_operations(self):
        sine, cosine, tangent = (reference(name, 3, 4) for name in ('sin', 'cos', 'tan'))
        total = sine + cosine + tangent
        exponential = cmath.exp(complex(sine, cosine))
        self.check('angle[4] a = 3 * pi / 8; '
                   'float out = sin(a) + cos(a) + tan(a); '
                   'float nested = sin(cos(a)); complex z = exp(sin(a) + cos(a) * 1im);',
                   (('out', False), ('nested', False), ('z', True), ('a', False)),
                   (total, math.sin(cosine), exponential.real, exponential.imag, 3))
        self.check('angle[4] a = 3 * pi / 8; int i = 1; array[float, 2] values = {7, 0}; '
                   'values[i] = sin(a) + cos(a); values[i] += tan(a); '
                   'float out = values[i]; float saved = values[0];',
                   (('out', False), ('saved', False), ('i', False), ('a', False)), (total, 7, 1, 3))
        # Resizing and angle-valued functions must keep their width metadata;
        # float-valued trigonometric results must select the float overload.
        self.check('angle[8] a = 3 * pi / 8; '
                   'float resized = sin(angle[4](a)); '
                   'float rotated = cos(rotr(angle[4](a), 1)); '
                   'float shifted = tan(angle[4](a) << 1);',
                   (('resized', False), ('rotated', False), ('shifted', False), ('a', False)),
                   (sine, reference('cos', 9, 4), reference('tan', 6, 4), 48))

    def test_conditions_runtime_loops_and_iterator_shadowing(self):
        self.check('angle[4] a = 0.0; int count = 0; '
                   'while (cos(a) > 0) { a += angle[4](pi / 4); count += 1; } '
                   'if (sin(a) == 1) { count += 10; } else { count = -1; }',
                   (('count', False), ('a', False)), (12, 4))
        self.check('angle[4] a = 0.0; int total = 0; '
                   'for int i in [int(sin(a)):int(cos(a)):3] { '
                   'if (i == 1) { continue; } total += i; if (i == 2) { break; } } '
                   'for int j in {int(tan(a)), int(cos(a))} { total += j; }',
                   (('total', False),), (3,))
        self.check('angle[4] a = pi / 2; array[float, 2] values = {0.5, 1}; float total = 0; '
                   'for float a in values { total += sin(a); } total += sin(a);',
                   (('total', False), ('a', False)), (math.sin(0.5) + math.sin(1.0) + 1, 4))

    def test_skipped_branch_and_loop_bodies_do_not_evaluate_domains(self):
        for call in ('out = tan(a);', 'out = sin(a / zero);', 'out = cos(angle[4](1e309));'):
            for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}',
                            'for int i in [first:last] {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                self.check('angle[4] a = pi / 2; uint[4] zero = 0; bool flag = false; '
                           'int first = 2; int last = 1; float out = 7; ' + wrapper.format(call),
                           (('out', False),), (7,))
        for condition, expected in (('false && bool(sin(a / zero))', 0),
                                    ('true || bool(cos(angle[4](1e309)))', 1)):
            self.check('angle[4] a = pi / 2; uint[4] zero = 0; bool out = ' + condition + ';',
                       (('out', False),), (expected,))

    def test_gate_parameters_with_and_without_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for name in ('sin', 'cos', 'tan'):
            theta, phi = reference(name, 3, 4), reference(name, 15, 4)
            expected = (cmath.exp(-0.5j * phi) * math.cos(theta / 2),
                        -1j * cmath.exp(0.5j * phi) * math.sin(theta / 2))
            for fused in (False, True):
                lines = self.program('include "stdgates.inc"; qubit q; angle[4] a = 3 * pi / 8; '
                                     f'rx({name}(a)) q; rz({name}(angle[4](-pi / 8))) q;', fused=fused)
                BackendAuditTests.check(self, lines, expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
                               for case in (ConstantTests, BackendTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f'angle trigonometric constant calls checked: {ConstantTests.checked}')
    print(f'angle trigonometric programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
