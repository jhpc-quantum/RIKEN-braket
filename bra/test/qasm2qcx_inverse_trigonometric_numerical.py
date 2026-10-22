#!/usr/bin/env python3

"""Audit arcsin/arccos/arctan against independent Decimal references.

Run without arguments for standard-library reference tests, or use
--bra /path/to/bra for the default-double, non-MPI backend audit.
References use Decimal square roots, half-angle reduction, and an arctangent
series, not binary libm inverse-trigonometric functions.
"""

import argparse
import cmath
import decimal
import functools
import math
import pathlib
import resource
import unittest

import qasm2qcx_math_function_numerical as functions
from qasm2qcx_trigonometric_numerical import pi


def decimal_arctan(value):
    """Evaluate atan in the caller's Decimal context."""
    negative = value < 0
    value = abs(value)
    reciprocal = value > 1
    if reciprocal:
        value = 1 / value
    factor = 1
    while value > decimal.Decimal('0.125'):
        value /= 1 + (1 + value * value).sqrt()
        factor *= 2
    term, total, n = value, value, 1
    while True:
        term *= -value * value
        n += 2
        updated = total + term / n
        if updated == total:
            break
        total = updated
    result = factor * total
    if reciprocal:
        result = pi() / 2 - result
    return -result if negative else result


@functools.lru_cache(maxsize=None)
def nonzero_inverse(name, value):
    with decimal.localcontext() as context:
        context.prec = 100
        x = decimal.Decimal.from_float(value)
        if name == 'arctan':
            return float(decimal_arctan(x))
        if abs(x) > 1:
            raise ValueError('real input outside [-1, 1]')
        if name == 'arcsin':
            if abs(x) == 1:
                return float(pi() / 2 if x > 0 else -pi() / 2)
            return float(decimal_arctan(x / ((1 - x) * (1 + x)).sqrt()))
        if name == 'arccos':
            if x == -1:
                return float(pi())
            # This form avoids cancellation of pi/2 - asin(x) near x=1.
            return float(2 * decimal_arctan(((1 - x) / (1 + x)).sqrt()))
        raise ValueError('unknown inverse trigonometric function')


def inverse_trigonometric(name, value):
    if name not in ('arcsin', 'arccos', 'arctan') or not math.isfinite(value):
        raise ValueError('unknown function or nonfinite input')
    # Keep both signs of zero out of a cache whose float keys compare equal.
    if value == 0:
        if name == 'arccos':
            with decimal.localcontext() as context:
                context.prec = 100
                return float(pi() / 2)
        return value
    return nonzero_inverse(name, value)


class ReferenceTests(unittest.TestCase):
    def test_known_values_and_signed_zeros(self):
        for name, value, expected in (('arcsin', 0.5, math.pi / 6), ('arccos', 0.5, math.pi / 3),
                                      ('arctan', 1.0, math.pi / 4), ('arcsin', -1.0, -math.pi / 2),
                                      ('arcsin', 1.0, math.pi / 2), ('arccos', -1.0, math.pi),
                                      ('arccos', 1.0, 0)):
            self.assertAlmostEqual(inverse_trigonometric(name, value), expected)
        for name in ('arcsin', 'arctan'):
            for value in (0.0, -0.0):
                result = inverse_trigonometric(name, value)
                self.assertEqual(result, 0)
                self.assertEqual(math.copysign(1, result), math.copysign(1, value))
        self.assertEqual(math.copysign(1, inverse_trigonometric('arccos', 1.0)), 1)

    def test_endpoint_neighbors_tiny_values_and_large_arctangents(self):
        for name in ('arcsin', 'arccos', 'arctan'):
            for value in (math.inf, -math.inf, math.nan):
                with self.assertRaises(ValueError):
                    inverse_trigonometric(name, value)
        inside = math.nextafter(1.0, 0)
        self.assertGreater(inverse_trigonometric('arccos', inside), 0)
        self.assertLess(inverse_trigonometric('arcsin', inside), math.pi / 2)
        for name in ('arcsin', 'arccos'):
            for value in (math.nextafter(1.0, math.inf), math.nextafter(-1.0, -math.inf)):
                with self.assertRaises(ValueError):
                    inverse_trigonometric(name, value)
        for name in ('arcsin', 'arctan'):
            for value in (5e-324, -5e-324, 1e-300, -1e-300):
                self.assertEqual(inverse_trigonometric(name, value), value)
        for value in (1e20, 1e308, float.fromhex('0x1.fffffffffffffp1023')):
            self.assertEqual(inverse_trigonometric('arctan', value), math.pi / 2)
            self.assertEqual(inverse_trigonometric('arctan', -value), -math.pi / 2)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute
    check = functions.BackendTests.check

    def test_constants_literals_variables_and_domain_endpoints(self):
        for name in ('arcsin', 'arccos', 'arctan'):
            values = [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1e-300, -1e-300, 1e-12, -1e-12,
                      float.fromhex('0x1p-1022'), -float.fromhex('0x1p-1022'),
                      math.nextafter(1.0, 0), math.nextafter(-1.0, 0)]
            if name == 'arctan':
                values.extend((2.0, -2.0, 1e20, -1e20, 1e308, -1e308, float.fromhex('0x1.fffffffffffffp1023')))
            for value in values:
                expected = inverse_trigonometric(name, value)
                for mode in ('constant', 'literal', 'variable'):
                    with self.subTest(name=name, value=value, mode=mode):
                        source = f'float x = {value!r}; '
                        if mode == 'constant':
                            source += f'const float C = {name}({value!r}); '
                        expression = 'C' if mode == 'constant' else f'{name}(x)' if mode == 'variable' else f'{name}({value!r})'
                        # Normalize small results to expose zero snapping and
                        # relative errors near ARCCOS's upper input endpoint.
                        scale = abs(expected) if expected else 1.0
                        self.check(source + f'float out = {expression}; out /= {scale!r};',
                                   (('out', False), ('x', False)), (expected / scale, value), signed_zero=True)

    def test_promotions_and_explicit_component_extraction(self):
        for name in ('arcsin', 'arccos', 'arctan'):
            for declaration, value in (('int x = -1;', -1.0), ('uint x = 1;', 1.0),
                                       ('bool x = true;', 1.0), ('bool x = false;', 0.0)):
                self.check(declaration + f' complex out = {name}(x);', (('out', True), ('x', False)),
                           (inverse_trigonometric(name, value), 0, value))
            self.check(f'bit b = 1; float out = {name}(float(b));', (('out', False),),
                       (inverse_trigonometric(name, 1.0),))
            self.check(f'complex z = 0.5 + 2im; float out = {name}(real(z));',
                       (('out', False), ('z', True)), (inverse_trigonometric(name, 0.5), 0.5, 2))
        value = 4294967295.0
        self.check('uint x = 4294967295; float out = arctan(x);', (('out', False),),
                   (inverse_trigonometric('arctan', value),))

    def test_subnormal_arithmetic_follows_backend_environment(self):
        for name in ('arcsin', 'arccos', 'arctan'):
            for value in (5e-324, -5e-324, 1e-310, -1e-310):
                result = self.execute(self.program(f'float x = {value!r}; float out = {name}(x);',
                                                   (('out', False), ('x', False))))
                self.assertEqual(result.returncode, 0, result.stderr)
                actual, saved = map(float, result.stdout.split())
                self.assertEqual(saved, value)
                if name == 'arccos':
                    self.assertTrue(math.isclose(actual, inverse_trigonometric(name, value), rel_tol=1e-5))
                else:
                    self.assertIn(actual, (value, 0.0))
                    self.assertEqual(math.copysign(1, actual), math.copysign(1, value))
                type(self).checked += 1

    def test_nesting_arrays_and_source_preservation(self):
        nested = inverse_trigonometric('arcsin', inverse_trigonometric('arccos', inverse_trigonometric('arctan', 1.0)))
        self.check('float x = 1; float out = arcsin(arccos(arctan(x))); x = arcsin(x);',
                   (('out', False), ('x', False)), (nested, math.pi / 2))
        first = inverse_trigonometric('arcsin', 0.5) + inverse_trigonometric('arccos', 0.5)
        self.check('array[float, 2] values = {0.5, 1}; array[complex, 2] zs = {0.5 + 1im, 2}; int i = 0; '
                   'values[i] = arcsin(values[i]) + arccos(real(zs[i])); values[i] += arctan(values[i]); '
                   'float out = values[i]; complex saved = zs[i];',
                   (('out', False), ('saved', True), ('i', False)),
                   (first + inverse_trigonometric('arctan', first), 0.5, 1, 0))
        self.check('array[float, 3] values = {arcsin(0.5), arccos(0.5), arctan(0.5)}; '
                   'float out = values[0] + values[1] + values[2];', (('out', False),),
                   (sum(inverse_trigonometric(name, 0.5) for name in ('arcsin', 'arccos', 'arctan')),))

    def test_control_flow_and_runtime_iteration_expressions(self):
        self.check('float x = 1; int count = 0; while (arcsin(x) > 0) { x -= 0.5; count += 1; }',
                   (('x', False), ('count', False)), (0, 2))
        self.check('float x = 0; int total = 0; for int i in [int(arcsin(x)):int(arccos(x)):3] { '
                   'if (i == 1) { continue; } total += i; if (i == 2) { break; } } '
                   'for int j in {int(arctan(x)), int(arccos(x))} { total += j; }', (('total', False),), (3,))
        self.check('array[float, 2] values = {0.5, 1}; float total = 0; '
                   'for float x in values { total += arcsin(x); }', (('total', False),),
                   (inverse_trigonometric('arcsin', 0.5) + math.pi / 2,))

    def test_gate_parameters_with_and_without_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for name in ('arcsin', 'arccos', 'arctan'):
            theta, phi = inverse_trigonometric(name, 0.5), inverse_trigonometric(name, -0.5)
            expected = (cmath.exp(-0.5j * phi) * math.cos(theta / 2),
                        -1j * cmath.exp(0.5j * phi) * math.sin(theta / 2))
            for fused in (False, True):
                lines = self.program(f'include "stdgates.inc"; qubit q; float x = 0.5; '
                                     f'rx({name}(x)) q; rz({name}(-0.5)) q;', fused=fused)
                BackendAuditTests.check(self, lines, expected)

    def test_skipped_calls_and_reached_errors(self):
        for name in ('arcsin', 'arccos', 'arctan'):
            for argument in ('2', '1e309'):
                self.check(f'bool out = true || bool({name}({argument}));', (('out', False),), (1,))
            self.check(f'bool out = false && bool({name}(0.0 / 0.0));', (('out', False),), (0,))
            for call in (f'out = {name}(1e309);', f'out = {name}(values[i]);'):
                for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}',
                                'for int j in [first:last] {{ {} }}', 'for int j in [1:0] {{ {} }}'):
                    self.check('bool flag = false; int first = 2; int last = 1; float out = 7; '
                               'array[float, 1] values = {2}; int i = 2; ' + wrapper.format(call),
                               (('out', False),), (7,))
            failures = [('1e309', 'nonfinite input'), ('-1e309', 'nonfinite input'),
                        ('0.0 / 0.0', 'nonfinite input'), ('real(1.0 / 0.0 + 1im)', 'nonfinite input')]
            if name != 'arctan':
                failures.extend((argument, 'real input outside [-1, 1]') for argument in
                                ('2', '-2', repr(math.nextafter(1.0, math.inf)),
                                 repr(math.nextafter(-1.0, -math.inf)), 'x', 'values[i]'))
            for argument, reason in failures:
                result = self.execute(self.program('float x = 2; array[float, 1] values = {2}; int i = 0; '
                                                   f'float out = {name}({argument});', (('out', False),)))
                self.assertNotEqual(result.returncode, 0, (name, argument))
                self.assertIn(name.upper(), result.stderr)
                self.assertIn(reason, result.stderr)
                self.assertEqual(result.stdout, '')
                type(self).checked += 1
            if name != 'arctan':
                result = self.execute(self.program(f'float x = 2; for int i in [0:int({name}(x)):3] {{}}'))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name.upper(), result.stderr)
                self.assertIn('real input outside [-1, 1]', result.stderr)
                type(self).checked += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve() if args.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if BackendTests.checked:
        print(f'Inverse-trigonometric programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
