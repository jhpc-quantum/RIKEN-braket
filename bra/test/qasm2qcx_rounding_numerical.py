#!/usr/bin/env python3

"""Audit ceiling/floor against exact integer-ratio references.

Run without arguments for standard-library reference tests, or pass
--bra /path/to/bra for the default-double, non-MPI backend audit.
Expected values use integer division of each binary float's exact ratio,
not math.ceil/math.floor or native floating-point rounding routines.
"""

import argparse
import cmath
import math
import pathlib
import resource
import unittest

import qasm2qcx_math_function_numerical as functions


def rounded(name, value):
    if name not in ('ceiling', 'floor') or not math.isfinite(value):
        raise ValueError('unknown function or nonfinite input')
    numerator, denominator = value.as_integer_ratio()
    integer = -(-numerator // denominator) if name == 'ceiling' else numerator // denominator
    result = float(integer)
    return math.copysign(result, value) if result == 0 else result


class ReferenceTests(unittest.TestCase):
    def test_known_values_and_zero_signs(self):
        for value, ceiling, floor in ((0.25, 1, 0), (-0.25, -0.0, -1), (1.5, 2, 1),
                                       (-1.5, -1, -2), (2.0, 2, 2), (0.0, 0.0, 0.0), (-0.0, -0.0, -0.0)):
            for name, expected in (('ceiling', ceiling), ('floor', floor)):
                actual = rounded(name, value)
                self.assertEqual(actual, expected)
                self.assertIsInstance(actual, float)
                self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))

    def test_neighbors_subnormals_extremes_and_invalid_inputs(self):
        for value in (5e-324, 1e-310):
            self.assertEqual(rounded('ceiling', value), 1)
            self.assertEqual(rounded('floor', -value), -1)
            self.assertEqual(math.copysign(1, rounded('ceiling', -value)), -1)
            self.assertEqual(math.copysign(1, rounded('floor', value)), 1)
        self.assertEqual(rounded('ceiling', math.nextafter(1.0, math.inf)), 2)
        self.assertEqual(rounded('floor', math.nextafter(1.0, 0)), 0)
        self.assertEqual(rounded('ceiling', math.nextafter(-1.0, 0)), -0.0)
        self.assertEqual(rounded('floor', math.nextafter(-1.0, -math.inf)), -2)
        for name in ('ceiling', 'floor'):
            for value in (1e20, -1e20, 1e308, -1e308, float.fromhex('0x1.fffffffffffffp1023')):
                self.assertEqual(rounded(name, value), value)
            for value in (math.inf, -math.inf, math.nan):
                with self.assertRaises(ValueError):
                    rounded(name, value)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute
    check = functions.BackendTests.check

    def test_constants_literals_variables_boundaries_and_extremes(self):
        values = [0.0, -0.0, 0.25, -0.25, 1.5, -1.5, 2.0, -2.0, 2147483648.5, -2147483648.5,
                  1e20, -1e20, 1e308, -1e308, float.fromhex('0x1.fffffffffffffp1023'),
                  -float.fromhex('0x1.fffffffffffffp1023'), float.fromhex('0x1p-1022'),
                  -float.fromhex('0x1p-1022'), 5e-324, -5e-324, 1e-310, -1e-310]
        values += [math.nextafter(boundary, direction) for boundary in (-2.0, -1.0, 0.0, 1.0, 2.0)
                   for direction in (-math.inf, math.inf)]
        for name in ('ceiling', 'floor'):
            for value in values:
                expected = rounded(name, value)
                for mode in ('constant', 'literal', 'variable'):
                    with self.subTest(name=name, value=value, mode=mode):
                        source = f'float x = {value!r}; '
                        if mode == 'constant':
                            source += f'const float C = {name}({value!r}); '
                        expression = 'C' if mode == 'constant' else f'{name}(x)' if mode == 'variable' else f'{name}({value!r})'
                        self.check(source + f'float out = {expression};', (('out', False), ('x', False)),
                                   (expected, value), signed_zero=True)

    def test_stored_subnormals_are_not_treated_as_exact_zero(self):
        for name in ('ceiling', 'floor'):
            for value in (5e-324, -5e-324, 1e-310, -1e-310,
                          math.nextafter(float.fromhex('0x1p-1022'), 0),
                          -math.nextafter(float.fromhex('0x1p-1022'), 0)):
                result = self.execute(self.program(f'float x = {value!r}; float out = {name}(x);',
                                                   (('out', False), ('x', False))))
                self.assertEqual(result.returncode, 0, result.stderr)
                actual, saved = map(float, result.stdout.split())
                # Stored inputs must survive, but PRINTLN's default precision
                # can round the largest subnormal to a nearby normal value.
                self.assertNotEqual(saved, 0)
                self.assertEqual(math.copysign(1, saved), math.copysign(1, value))
                if abs(value) < 1e-309:
                    self.assertEqual(saved, value)
                expected = rounded(name, value)
                self.assertEqual(actual, expected)
                self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))
                type(self).checked += 1

    def test_promotions_and_explicit_component_extraction(self):
        for name in ('ceiling', 'floor'):
            for declaration, value in (('int x = -2;', -2.0), ('uint x = 4294967295;', 4294967295.0),
                                       ('bool x = true;', 1.0), ('bool x = false;', 0.0)):
                self.check(declaration + f' complex out = {name}(x);', (('out', True), ('x', False)),
                           (rounded(name, value), 0, value))
            self.check(f'bit b = 1; float out = {name}(float(b));', (('out', False),), (1,))
            self.check(f'complex z = -0.25 + 2im; float out = {name}(real(z));',
                       (('out', False), ('z', True)), (rounded(name, -0.25), -0.25, 2), signed_zero=True)

    def test_nesting_arrays_assignments_and_source_preservation(self):
        self.check('float x = 1.25; float out = ceiling(floor(x)); x = floor(x);',
                   (('out', False), ('x', False)), (1, 1))
        self.check('float x = -0.25; x = ceiling(x); float out = floor(x);',
                   (('out', False), ('x', False)), (-0.0, -0.0), signed_zero=True)
        self.check('array[float, 2] values = {0.25, 1}; array[complex, 2] zs = {0.5 + 1im, 2}; int i = 0; '
                   'values[i] = ceiling(values[i]) + floor(real(zs[i])); values[i] += floor(values[i] + 0.5); '
                   'float out = values[i]; complex saved = zs[i];',
                   (('out', False), ('saved', True), ('i', False)), (2, 0.5, 1, 0))
        self.check('array[float, 2] values = {ceiling(0.5), floor(-0.5)}; float out = values[0] + values[1];',
                   (('out', False),), (0,))

    def test_control_flow_and_runtime_iteration_expressions(self):
        self.check('float x = 2.5; int count = 0; while (floor(x) > 0) { x -= 1; count += 1; }',
                   (('x', False), ('count', False)), (0.5, 2))
        self.check('float x = 0.5; int total = 0; for int i in [int(floor(x)):int(ceiling(x)):3] { '
                   'if (i == 1) { continue; } total += i; if (i == 2) { break; } } '
                   'for int j in {int(floor(x)), int(ceiling(x))} { total += j; }', (('total', False),), (3,))
        self.check('array[float, 2] values = {0.25, 1.25}; float total = 0; '
                   'for float x in values { total += ceiling(x); }', (('total', False),), (3,))

    def test_gate_parameters_with_and_without_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for name in ('ceiling', 'floor'):
            theta, phi = rounded(name, 0.5), rounded(name, -0.5)
            expected = (cmath.exp(-0.5j * phi) * math.cos(theta / 2),
                        -1j * cmath.exp(0.5j * phi) * math.sin(theta / 2))
            for fused in (False, True):
                lines = self.program(f'include "stdgates.inc"; qubit q; float x = 0.5; '
                                     f'rx({name}(x)) q; rz({name}(-0.5)) q;', fused=fused)
                BackendAuditTests.check(self, lines, expected)

    def test_skipped_calls_and_reached_errors(self):
        for name in ('ceiling', 'floor'):
            for expression, expected in ((f'true || bool({name}(1e309))', 1),
                                         (f'false && bool({name}(0.0 / 0.0))', 0)):
                self.check(f'bool out = {expression};', (('out', False),), (expected,))
            for call in (f'out = {name}(1e309);', f'out = {name}(values[i]);'):
                for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}',
                                'for int j in [first:last] {{ {} }}', 'for int j in [1:0] {{ {} }}'):
                    self.check('bool flag = false; int first = 2; int last = 1; float out = 7; '
                               'array[float, 1] values = {1}; int i = 2; ' + wrapper.format(call),
                               (('out', False),), (7,))
            for argument in ('1e309', '-1e309', '0.0 / 0.0', 'x', 'real(1.0 / 0.0 + 1im)'):
                result = self.execute(self.program(f'float x = 1.0 / 0.0; float out = {name}({argument});',
                                                   (('out', False),)))
                self.assertNotEqual(result.returncode, 0, (name, argument))
                self.assertIn(name.upper(), result.stderr)
                self.assertIn('nonfinite input', result.stderr)
                self.assertEqual(result.stdout, '')
                type(self).checked += 1
            result = self.execute(self.program(f'float x = 1.0 / 0.0; for int i in [0:int({name}(x)):3] {{}}'))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(name.upper(), result.stderr)
            self.assertIn('nonfinite input', result.stderr)
            type(self).checked += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve() if args.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if BackendTests.checked:
        print(f'Rounding programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
