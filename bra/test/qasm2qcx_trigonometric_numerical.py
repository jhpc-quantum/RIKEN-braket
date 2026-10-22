#!/usr/bin/env python3

"""Audit OpenQASM sin/cos/tan using independent Decimal series references.

Run without arguments for standard-library reference tests; pass
--bra /path/to/bra for the default-double, non-MPI backend audit.
The reference retains over 100 reduction digits even for maximum doubles.
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


@functools.lru_cache(maxsize=None)
def pi():
    """Machin's formula, evaluated independently of binary libm constants."""
    with decimal.localcontext() as context:
        context.prec = 430

        def arctangent_inverse(denominator):
            x = decimal.Decimal(1) / denominator
            term, total, n = x, x, 1
            while True:
                term *= -x * x
                n += 2
                updated = total + term / n
                if updated == total:
                    return total
                total = updated

        return 16 * arctangent_inverse(5) - 4 * arctangent_inverse(239)


@functools.lru_cache(maxsize=None)
def sine_cosine(value):
    """Reduce the exact binary float, then evaluate Taylor series in Decimal."""
    with decimal.localcontext() as context:
        context.prec = 420
        angle = decimal.Decimal.from_float(value) % (2 * pi())
        if angle > pi():
            angle -= 2 * pi()
        elif angle < -pi():
            angle += 2 * pi()
        squared = -angle * angle
        sine, cosine = angle, decimal.Decimal(1)
        sine_term, cosine_term = sine, cosine
        n = 0
        while True:
            n += 1
            sine_term *= squared / ((2 * n) * (2 * n + 1))
            cosine_term *= squared / ((2 * n - 1) * (2 * n))
            next_sine, next_cosine = sine + sine_term, cosine + cosine_term
            if next_sine == sine and next_cosine == cosine:
                return sine, cosine
            sine, cosine = next_sine, next_cosine


def trigonometric(name, value):
    # Handle zero before the cache: +0 and -0 compare equal as cache keys.
    if value == 0:
        return 1.0 if name == 'cos' else value
    sine, cosine = sine_cosine(value)
    with decimal.localcontext() as context:
        context.prec = 100
        return float(sine if name == 'sin' else cosine if name == 'cos' else sine / cosine)


class ReferenceTests(unittest.TestCase):
    def test_known_values_and_signed_zeros(self):
        for name, expected in (('sin', 0.479425538604203), ('cos', 0.8775825618903728),
                                ('tan', 0.5463024898437905)):
            self.assertAlmostEqual(trigonometric(name, 0.5), expected)
            for value in (0.0, -0.0):
                result = trigonometric(name, value)
                self.assertEqual(result, 1 if name == 'cos' else 0)
                self.assertEqual(math.copysign(1, result), 1 if name == 'cos' else math.copysign(1, value))

    def test_large_argument_reference_and_pole_neighbors(self):
        self.assertAlmostEqual(trigonometric('sin', 1e20), -0.6452512852657808)
        self.assertAlmostEqual(trigonometric('cos', 1e20), 0.7639704044417283)
        self.assertGreater(trigonometric('tan', math.nextafter(math.pi / 2, 0)), 0)
        self.assertLess(trigonometric('tan', math.nextafter(math.pi / 2, math.inf)), 0)
        for value in (1e308, float.fromhex('0x1.fffffffffffffp1023')):
            sine, cosine = sine_cosine(value)
            self.assertAlmostEqual(float(sine * sine + cosine * cosine), 1)
            for name in ('sin', 'cos', 'tan'):
                positive, negative = trigonometric(name, value), trigonometric(name, -value)
                self.assertEqual(negative, positive if name == 'cos' else -positive)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute
    check = functions.BackendTests.check

    def test_constants_literals_variables_and_extreme_arguments(self):
        for name in ('sin', 'cos', 'tan'):
            for value in (0.0, -0.0, 0.5, -1.0, 1e-300, -1e-300, 1e-12, -1e-12,
                          math.pi, math.nextafter(math.pi, 0), math.nextafter(math.pi, math.inf),
                          math.pi / 2, math.nextafter(math.pi / 2, 0), math.nextafter(math.pi / 2, math.inf),
                          1e20, -1e20, 1e100, -1e100, 1e308, -1e308,
                          float.fromhex('0x1.fffffffffffffp1023')):
                expected = trigonometric(name, value)
                for mode in ('constant', 'literal', 'variable'):
                    with self.subTest(name=name, value=value, mode=mode):
                        source = f'float x = {value!r}; '
                        if mode == 'constant':
                            source += f'const float C = {name}({value!r}); '
                        expression = 'C' if mode == 'constant' else f'{name}(x)' if mode == 'variable' else f'{name}({value!r})'
                        # Normalize small answers so absolute output tolerances
                        # cannot hide zero snapping or inaccurate reduction.
                        scale = abs(expected) if expected else 1.0
                        source += f'float out = {expression}; out /= {scale!r};'
                        self.check(source, (('out', False), ('x', False)), (expected / scale, value), signed_zero=True)

    def test_promotions_and_explicit_component_extraction(self):
        for name in ('sin', 'cos', 'tan'):
            for declaration, value in (('int x = -1;', -1.0), ('uint x = 4294967295;', 4294967295.0),
                                       ('bool x = true;', 1.0), ('bool x = false;', 0.0)):
                self.check(declaration + f' complex out = {name}(x);', (('out', True), ('x', False)),
                           (trigonometric(name, value), 0, value))
            self.check(f'bit b = 1; float out = {name}(float(b));', (('out', False),), (trigonometric(name, 1.0),))
            self.check(f'complex z = 1 + 2im; float out = {name}(real(z));',
                       (('out', False), ('z', True)), (trigonometric(name, 1.0), 1, 2))

    def test_subnormal_arithmetic_follows_backend_environment(self):
        for name in ('sin', 'cos', 'tan'):
            for value in (5e-324, -5e-324, 1e-310, -1e-310):
                result = self.execute(self.program(f'float x = {value!r}; float out = {name}(x);',
                                                   (('out', False), ('x', False))))
                self.assertEqual(result.returncode, 0, result.stderr)
                actual, saved = map(float, result.stdout.split())
                self.assertEqual(saved, value)
                # SIN/TAN round to the operand at this scale. DAZ/FTZ may
                # instead produce signed zero; COS rounds to one either way.
                self.assertIn(actual, (1.0,) if name == 'cos' else (value, 0.0))
                self.assertEqual(math.copysign(1, actual), 1 if name == 'cos' else math.copysign(1, value))
                type(self).checked += 1

    def test_nesting_arrays_compound_assignment_and_source_preservation(self):
        nested = trigonometric('sin', trigonometric('cos', trigonometric('tan', 0.5)))
        self.check('float x = 0.5; float out = sin(cos(tan(x))); x = sin(x);',
                   (('out', False), ('x', False)), (nested, trigonometric('sin', 0.5)))
        first = trigonometric('sin', 0.5) + trigonometric('cos', 0.5)
        self.check('array[float, 2] values = {0.5, 1}; array[complex, 2] zs = {0.5 + 1im, 2}; int i = 0; '
                   'values[i] = sin(values[i]) + cos(real(zs[i])); values[i] += tan(values[i]); '
                   'float out = values[i]; complex saved = zs[i];',
                   (('out', False), ('saved', True), ('i', False)),
                   (first + trigonometric('tan', first), 0.5, 1, 0))
        self.check('array[float, 3] values = {sin(0.5), cos(0.5), tan(0.5)}; '
                   'float out = values[0] + values[1] + values[2];', (('out', False),),
                   (sum(trigonometric(name, 0.5) for name in ('sin', 'cos', 'tan')),))

    def test_control_flow_and_runtime_iteration_expressions(self):
        self.check('float x = 0; int count = 0; while (cos(x) > 0) { x += 1; count += 1; }',
                   (('x', False), ('count', False)), (2, 2))
        self.check('float x = 0; int total = 0; for int i in [int(sin(x)):int(cos(x)):3] { '
                   'if (i == 1) { continue; } total += i; if (i == 2) { break; } } '
                   'for int j in {int(tan(x)), int(cos(x))} { total += j; }', (('total', False),), (3,))
        self.check('array[float, 2] values = {0.5, 1}; float total = 0; '
                   'for float x in values { total += sin(x); }', (('total', False),),
                   (trigonometric('sin', 0.5) + trigonometric('sin', 1.0),))

    def test_gate_parameters_with_and_without_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        for name in ('sin', 'cos', 'tan'):
            theta, phi = trigonometric(name, 0.5), trigonometric(name, -0.5)
            expected = (cmath.exp(-0.5j * phi) * math.cos(theta / 2),
                        -1j * cmath.exp(0.5j * phi) * math.sin(theta / 2))
            for fused in (False, True):
                lines = self.program(f'include "stdgates.inc"; qubit q; float x = 0.5; '
                                     f'rx({name}(x)) q; rz({name}(-0.5)) q;', fused=fused)
                BackendAuditTests.check(self, lines, expected)

    def test_skipped_calls_and_reached_errors(self):
        for name in ('sin', 'cos', 'tan'):
            for expression, expected in ((f'false && bool({name}(1e309))', 0),
                                         (f'true || ({name}(0.0 / 0.0) > 0)', 1)):
                self.check('bool out = ' + expression + ';', (('out', False),), (expected,))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve() if args.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if BackendTests.checked:
        print(f'Trigonometric programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
