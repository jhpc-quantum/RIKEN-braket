#!/usr/bin/env python3

"""Audit constant and runtime OpenQASM sqrt calls.

  python3 bra/test/qasm2qcx_sqrt_numerical.py
  python3 bra/test/qasm2qcx_sqrt_numerical.py --bra /path/to/bra

Reference-only tests use the standard library. The optional default-double,
non-MPI backend audit checks promotions, principal branches, signed zeros,
arrays, control flow, failures, and gate parameters with and without fusion.
Expected complex roots use a scaled Cartesian formula, not cmath.sqrt.
Subnormal arithmetic inherits the backend floating-point environment.
"""

import argparse
import cmath
import math
import pathlib
import re
import resource
import unittest

import qasm2qcx_math_function_numerical as functions


def principal_root(value):
    """Scaled Cartesian principal root for finite complex reference inputs."""
    x, y = value.real, value.imag
    scale = max(abs(x), abs(y))
    if scale == 0:
        return complex(0, y)
    radius = math.hypot(x / scale, y / scale)
    large = math.sqrt(scale) * math.sqrt((radius + abs(x / scale)) / 2)
    if x >= 0:
        return complex(large, y / (2 * large))
    return complex(abs(y) / (2 * large), math.copysign(large, y))


class ReferenceTests(unittest.TestCase):
    def test_known_roots_and_branch_endpoints(self):
        for value, expected in ((3+4j, 2+1j), (-3+4j, 1+2j), (-3-4j, 1-2j),
                                (complex(-4, 0), 2j), (complex(-4, -0.0), -2j)):
            self.assertEqual(principal_root(value), expected)
        for x in (0.0, -0.0):
            for y in (0.0, -0.0):
                root = principal_root(complex(x, y))
                self.assertEqual(root, 0j)
                self.assertEqual(math.copysign(1, root.real), 1)
                self.assertEqual(math.copysign(1, root.imag), math.copysign(1, y))

    def test_scaled_roots_and_maximum_components(self):
        for scale in (1e-300, 1.0, 1e307):
            root = principal_root(complex(-3 * scale, -4 * scale))
            self.assertAlmostEqual(root.real / math.sqrt(scale), 1)
            self.assertAlmostEqual(root.imag / math.sqrt(scale), -2)
        root = principal_root(complex(1e308, -1e308))
        self.assertAlmostEqual(root.real / 1e154, math.sqrt((math.sqrt(2) + 1) / 2))
        self.assertAlmostEqual(root.imag / 1e154, -math.sqrt((math.sqrt(2) - 1) / 2))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute
    check = functions.BackendTests.check

    def test_real_constants_literals_and_variables(self):
        for value in (0.0, -0.0, 0.25, 2.0, 4.0, 1e-300, 1e308):
            scale = math.sqrt(value) if value != 0 else 1
            for mode in ('constant', 'literal', 'variable'):
                prefix = f'float x = {value!r}; '
                if mode == 'constant':
                    prefix += f'const float C = sqrt({value!r}); '
                expression = 'C' if mode == 'constant' else f'sqrt({value!r})' if mode == 'literal' else 'sqrt(x)'
                self.check(prefix + f'float out = {expression}; out /= {scale!r};',
                           (('out', False), ('x', False)), (math.sqrt(value) / scale, value), signed_zero=True)

    def test_complex_constants_literals_and_variables(self):
        profiles = [('3 + 4im', 3+4j), ('-3 - 4im', -3-4j), ('complex(-4)', complex(-4, 0)),
                    ('-complex(4)', complex(-4, -0.0)), ('1im', 1j),
                    ('1e308 + 1e308im', complex(1e308, 1e308)),
                    ('-1e308 - 1e308im', complex(-1e308, -1e308)),
                    ('1e-300 - 1e-300im', complex(1e-300, -1e-300))]
        for argument, value in profiles:
            expected = principal_root(value)
            scale = math.sqrt(max(abs(value.real), abs(value.imag)))
            for mode in ('constant', 'literal', 'variable'):
                prefix = f'complex z = {argument}; '
                if mode == 'constant':
                    prefix += f'const complex C = sqrt({argument}); '
                expression = 'C' if mode == 'constant' else f'sqrt({argument})' if mode == 'literal' else 'sqrt(z)'
                self.check(prefix + f'complex out = {expression}; out /= complex({scale!r});',
                           (('out', True), ('z', True)),
                           (expected.real / scale, expected.imag / scale, value.real, value.imag))

    def test_signed_zero_roots_in_literals_variables_constants_and_arrays(self):
        for expression, expected in (('sqrt(-0.0)', (-0.0,)), ('sqrt(-0.0im)', (0.0, -0.0)),
                                     ('sqrt(complex(-4))', (0.0, 2.0)), ('sqrt(-complex(4))', (0.0, -2.0)),
                                     ('sqrt(-complex(-4))', (2.0, -0.0))):
            is_complex = len(expected) == 2
            result_type = 'complex' if is_complex else 'float'
            with self.subTest(expression=expression):
                self.check(f'{result_type} out = {expression};', (('out', is_complex),), expected, signed_zero=True)
        for declaration, expression in (
                ('complex z = -complex(4);', 'sqrt(z)'),
                ('const complex Z = -complex(4);', 'sqrt(Z)'),
                ('array[complex, 1] zs = {-complex(4)}; int i = 0;', 'sqrt(zs[i])')):
            self.check(declaration + ' complex out = ' + expression + ';', (('out', True),), (0, -2), signed_zero=True)
        for expression, re_part, im_part in (('complex(0.0)', 0.0, 0.0), ('complex(-0.0)', -0.0, 0.0),
                                             ('-complex(-0.0)', 0.0, -0.0), ('-0.0im', -0.0, -0.0),
                                             ('-1im', -0.0, -1.0)):
            with self.subTest(stored=expression):
                expected = principal_root(complex(re_part, im_part))
                self.check(f'complex z = {expression}; complex out = sqrt(z);', (('z', True), ('out', True)),
                           (re_part, im_part, expected.real, expected.imag), signed_zero=True)
                negated = complex(-re_part, -im_part)
                expected = principal_root(negated)
                self.check(f'complex z = {expression}; complex n = -z; complex out = sqrt(n);',
                           (('n', True), ('out', True)),
                           (negated.real, negated.imag, expected.real, expected.imag), signed_zero=True)
        self.check('complex z = 4; complex out = sqrt(-z);', (('out', True),), (0, -2), signed_zero=True)

    def test_branch_cut_neighbors_and_input_promotions(self):
        for imaginary in (1e-12, -1e-12, 1e-300, -1e-300):
            expected = principal_root(complex(-1, imaginary))
            self.check(f'complex out = sqrt(-1 + ({imaginary!r}) * 1im); '
                       f'float small = real(out) / {expected.real!r}; float side = imag(out);',
                       (('small', False), ('side', False)), (1, expected.imag))
        for declaration, expected in (('int x = 4;', 2), ('uint x = 9;', 3),
                                      ('bool x = true;', 1), ('bool x = false;', 0)):
            self.check(declaration + ' complex out = sqrt(x);', (('out', True),), (expected, 0))
        self.check('bit b = 1; float out = sqrt(float(b));', (('out', False),), (1,))

    def test_constructor_preserves_subnormal_component_and_branch_side(self):
        tiny = float.fromhex('0x0.0000000000001p-1022')
        for imaginary in (tiny, -tiny):
            lines = self.program(f'complex z = -1 + ({imaginary!r}) * 1im; complex out = sqrt(z);',
                                 (('z', True), ('out', True)))
            result = self.execute(lines)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = list(map(float, result.stdout.split()))
            self.assertEqual(len(actual), 4)
            self.assertEqual(actual[:2], [-1, imaginary])
            # Root arithmetic may flush a subnormal result, but construction
            # must retain the stored input and hence the side of the cut.
            self.assertGreaterEqual(actual[2], 0)
            self.assertAlmostEqual(actual[3], math.copysign(1, imaginary))
            type(self).checked += 1

    def test_nesting_self_assignment_arrays_and_source_preservation(self):
        first = principal_root(-3+4j)
        second = principal_root(first)
        self.check('complex z = -3 + 4im; float x = sqrt(real(sqrt(z))); z = sqrt(z); z = sqrt(z);',
                   (('x', False), ('z', True)), (1, second.real, second.imag))
        self.check('array[float, 2] values = {4, 9}; array[complex, 2] zs = {-3 + 4im, -complex(4)}; int i = 0; '
                   'values[i] = sqrt(values[i]) + real(sqrt(zs[i])); values[i] += sqrt(values[i]); '
                   'float out = values[i]; complex saved = zs[i]; i = 1; complex lower = sqrt(zs[i]);',
                   (('out', False), ('saved', True), ('lower', True), ('i', False)), (3 + math.sqrt(3), -3, 4, 0, -2, 1))

    def test_functions_in_conditions_loops_and_gate_parameters(self):
        self.check('int n = 3; float total = 0; while (sqrt(float(n)) > 0) { total += sqrt(float(n * n)); n -= 1; }',
                   (('total', False), ('n', False)), (6, 0))
        self.check('int stop = 3; float total = 0; for int i in [0:stop] { '
                   'if (i == 1) { continue; } total += sqrt(float(i * i)); if (i == 2) { break; } }',
                   (('total', False),), (2,))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        theta, phi = 0.7, 1.0
        expected = (cmath.exp(-0.5j * phi) * math.cos(theta / 2),
                    -1j * cmath.exp(0.5j * phi) * math.sin(theta / 2))
        for fused in (False, True):
            lines = self.program('include "stdgates.inc"; qubit q; float x = 0.49; '
                                 'rx(sqrt(x)) q; rz(real(sqrt(-3 + 4im))) q;', fused=fused)
            BackendAuditTests.check(self, lines, expected)

    def test_documented_square_root_example(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- sqrt-function-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        theta, phi = math.sqrt(0.49), 2.0
        expected = (cmath.exp(-0.5j * phi) * math.cos(theta / 2),
                    -1j * cmath.exp(0.5j * phi) * math.sin(theta / 2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(examples[0], fused=fused), expected)

    def test_skipped_calls_and_reached_failures(self):
        for expression, expected in (('false && bool(sqrt(-1))', 0), ('true || (sqrt(-1) > 0)', 1),
                                     ('true || (real(sqrt(1e309im)) > 0)', 1)):
            self.check('bool out = ' + expression + ';', (('out', False),), (expected,))
        for call in ('out = sqrt(-1);', 'out = real(sqrt(1e309im));', 'out = sqrt(values[i]);'):
            for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}', 'for int j in [first:last] {{ {} }}',
                            'for int j in [1:0] {{ {} }}'):
                self.check('bool flag = false; int first = 2; int last = 1; float out = 7; '
                           'array[float, 1] values = {4}; int i = 2; ' + wrapper.format(call), (('out', False),), (7,))
        for expression, result_type, reason in (('sqrt(-1)', 'float', 'negative real input'),
                                               ('sqrt(1e309)', 'float', 'nonfinite input'),
                                               ('sqrt(-1e309)', 'float', 'nonfinite input'),
                                               ('sqrt(1e309im)', 'complex', 'nonfinite input'),
                                               ('sqrt(0.0 / 0.0)', 'float', 'nonfinite input'),
                                               ('sqrt(complex(0.0 / 0.0))', 'complex', 'nonfinite input'),
                                               ('sqrt(-complex(1.0 / 0.0))', 'complex', 'nonfinite input'),
                                               ('sqrt(real(sqrt(-complex(4))) - 1)', 'float', 'negative real input')):
            result = self.execute(self.program(result_type + ' out = ' + expression + ';',
                                               (('out', result_type == 'complex'),)))
            self.assertNotEqual(result.returncode, 0, expression)
            self.assertIn('SQRT', result.stderr)
            self.assertIn(reason, result.stderr)
            self.assertEqual(result.stdout, '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    options = parser.parse_args()
    if options.bra is not None:
        BackendTests.bra = options.bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if BackendTests.checked:
        print(f'SQRT programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
