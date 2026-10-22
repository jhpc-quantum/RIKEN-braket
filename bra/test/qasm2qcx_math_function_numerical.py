#!/usr/bin/env python3

"""Audit exp/real/imag calls in qasm2qcx (issue #362).

  python3 bra/test/qasm2qcx_math_function_numerical.py
  python3 bra/test/qasm2qcx_math_function_numerical.py --bra /path/to/bra

Reference-only tests use the standard library. The optional default-double,
non-MPI audit checks constants, executable literal calls, live variables,
arrays, nesting, short-circuiting, loops, failures, and quantum amplitudes.
Complex exponential references use exp(real)*(cos(imag)+i*sin(imag)), not
converter folding or emitted QCX operands. Ordinary native underflow is allowed.
"""

import argparse
import cmath
import math
import pathlib
import resource
import subprocess
import unittest


def complex_exp_reference(value):
    return math.exp(value.real) * complex(math.cos(value.imag), math.sin(value.imag))


class ReferenceTests(unittest.TestCase):
    def test_known_complex_exponentials(self):
        for value, expected in ((0j, 1), (complex(0, math.pi / 2), 1j),
                                (complex(0, math.pi), -1), (complex(1, 0), math.e)):
            self.assertAlmostEqual(complex_exp_reference(value), expected)

    def test_finite_large_imaginary_arguments_and_underflow(self):
        for imaginary in (1e20, 1e308, -1e308):
            self.assertAlmostEqual(abs(complex_exp_reference(complex(0, imaginary))), 1)
            self.assertEqual(complex_exp_reference(complex(-1000, imaginary)), 0)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit converted mathematical functions')
        from qasm2qcx_runtime_for_numerical import qasm2qcx
        self.module = qasm2qcx

    def program(self, source, outputs=(), *, fused=False):
        source = source if source.startswith('OPENQASM ') else 'OPENQASM 3; ' + source
        program = self.module.openqasm3.parser.parse(source)
        converter = self.module.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        if fused:
            lines.insert(1, 'BEGIN FUSION')
            lines.append('END FUSION')
        for name, is_complex in outputs:
            operand = converter._QASM2QCXConverter__capitalize_variable_name(name)
            lines.extend([f'PRINTLN :REAL:{operand}', f'PRINTLN :IMAG:{operand}'] if is_complex else [f'PRINTLN {operand}'])
        return lines

    def execute(self, lines):
        return subprocess.run([str(self.bra)], input='\n'.join(lines) + '\n', text=True, capture_output=True, timeout=30)

    def check(self, source, outputs, expected, *, fused=False, signed_zero=False):
        lines = self.program(source, outputs, fused=fused)
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + '\n'.join(lines))
        actual = [float(value) for value in result.stdout.split()]
        self.assertEqual(len(actual), len(expected), result.stdout)
        for value, wanted in zip(actual, expected):
            if math.isnan(wanted):
                self.assertTrue(math.isnan(value), (value, wanted))
            else:
                self.assertTrue(math.isclose(value, wanted, rel_tol=1e-5, abs_tol=2e-6), (source, value, wanted))
            if signed_zero and wanted == 0:
                self.assertEqual(math.copysign(1, value), math.copysign(1, wanted))
        type(self).checked += 1

    def test_real_exp_constants_literals_and_variables(self):
        for value in (-1e308, -1000.0, -745.0, -2.0, 0.0, 0.7, 10.0, 700.0):
            for mode in ('constant', 'literal', 'variable'):
                declarations = f'float x = {value!r}; '
                if mode == 'constant':
                    declarations += f'const float C = exp({value!r}); '
                expression = 'C' if mode == 'constant' else f'exp({value!r})' if mode == 'literal' else 'exp(x)'
                with self.subTest(value=value, mode=mode):
                    self.check(declarations + f'float out = {expression};',
                               (('out', False), ('x', False)), (math.exp(value), value))

    def test_complex_exp_constants_literals_and_variables(self):
        for value in (0j, complex(-0.5, 2), complex(0, math.pi), complex(1, -math.pi),
                      complex(0, 1e20), complex(0, 1e308), complex(-1000, 1e308), complex(700, 0.4)):
            argument = f'({value.real!r} + ({value.imag!r}) * 1im)'
            expected = complex_exp_reference(value)
            for mode in ('constant', 'literal', 'variable'):
                declarations = f'complex z = {argument}; '
                if mode == 'constant':
                    declarations += f'const complex C = exp({argument}); '
                expression = 'C' if mode == 'constant' else f'exp({argument})' if mode == 'literal' else 'exp(z)'
                with self.subTest(value=value, mode=mode):
                    self.check(declarations + f'complex out = {expression};', (('out', True), ('z', True)),
                               (expected.real, expected.imag, value.real, value.imag))

    def test_promotions_and_component_extraction(self):
        for declaration, value in (('int x = -2;', -2), ('uint x = 2;', 2), ('bool x = true;', 1),
                                    ('bool x = false;', 0), ('float x = 0.7;', 0.7)):
            self.check(declaration + ' complex e = exp(x); float r = real(x); float i = imag(x);',
                       (('e', True), ('r', False), ('i', False), ('x', False)), (math.exp(value), 0, value, 0, value))
        self.check('uint x = uint(-1); float r = real(x); float i = imag(x);', (('r', False), ('i', False)),
                   (self.module.QASM2QCXConverter.QCX_UINT_MAX, 0))
        self.check('float a = real(-0.0); float b = imag(-0.0im);', (('a', False), ('b', False)), (-0.0, -0.0), signed_zero=True)
        self.check('float a = real(1e309 + 2im); float b = imag(1e309 + 2im);',
                   (('a', False), ('b', False)), (math.inf, 2))

    def test_nesting_self_assignment_arrays_and_source_preservation(self):
        first = complex_exp_reference(complex(0.1, 0.2))
        second = complex_exp_reference(first)
        self.check('complex z = 0.1 + 0.2im; float r = real(exp(z)); z = exp(z); z = exp(z);',
                   (('r', False), ('z', True)), (first.real, second.real, second.imag))
        self.check('array[float, 2] values = {0.5, 1.0}; array[complex, 2] zs = {1im, 2im}; int index = 0; '
            'values[index] = exp(values[index]) + real(exp(zs[index])); '
            'values[index] += exp(values[index]); float out = values[index]; '
            'float a = real(zs[index]); float b = imag(zs[index]);',
            (('out', False), ('a', False), ('b', False), ('index', False)),
            ((math.exp(0.5) + math.cos(1)) + math.exp(math.exp(0.5) + math.cos(1)), 0, 1, 0))
        self.check('array[float, 2] values = {exp(0), exp(1)}; array[complex, 2] zs = {exp(1im), exp(-1im)}; '
            'int index = 1; float out = exp(real(zs[index])); float saved = values[index];',
            (('out', False), ('saved', False), ('index', False)), (math.exp(math.cos(1)), math.e, 1))

    def test_short_circuiting_branches_and_empty_loops_skip_numerical_failures(self):
        for expression, expected in (('false && bool(exp(1000))', 0), ('true || (exp(1000) > 0)', 1),
                                     ('true || bool(real(exp(1000 + 1im)))', 1)):
            self.check('bool out = ' + expression + ';', (('out', False),), (expected,))
        for call in ('out = exp(1000);', 'out = exp(1e309);', 'out = real(exp(1000 + 1im));',
                     'out = exp(values[index]);'):
            for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}',
                            'for int i in [first:last] {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                self.check('bool flag = false; float out = 7; int first = 2; int last = 1; '
                    'array[float, 1] values = {0.5}; int index = 2; ' + wrapper.format(call), (('out', False),), (7,))

    def test_functions_in_conditions_runtime_loops_and_integer_casts(self):
        self.check('int count = 0; float total = 0; '
            'while (real(exp(float(count) * 1im)) > 0) { total += exp(-float(count)); count += 1; }',
            (('total', False), ('count', False)), (1 + math.exp(-1), 2))
        self.check('int first = 1; int last = 3; float total = 0; '
            'for int i in [first:last] { if (i == 2) { continue; } total += exp(-float(i)); if (i == 3) { break; } }',
            (('total', False),), (math.exp(-1) + math.exp(-3),))
        self.check('float x = 0.7; int out = int(exp(x)); bool ok = (real(exp(1im)) > 0) && (imag(exp(1im)) > 0);',
            (('out', False), ('ok', False), ('x', False)), (2, 1, 0.7))

    def test_reached_exp_failures_identify_native_function_and_produce_no_result(self):
        for expression, reason, is_complex in (('exp(1000)', 'overflow', False), ('exp(1000 + 1im)', 'overflow', True),
                                    ('exp(1e309)', 'nonfinite input', False), ('exp(-1e309)', 'nonfinite input', False),
                                    ('exp(1e309im)', 'nonfinite input', True), ('exp(0.0 / 0.0)', 'nonfinite input', False),
                                    ('real(exp(1000 + 1im))', 'overflow', False)):
            result_type = 'complex' if is_complex else 'float'
            result = self.execute(self.program(result_type + ' out = ' + expression + ';', (('out', is_complex),)))
            self.assertNotEqual(result.returncode, 0, expression)
            self.assertIn('EXP', result.stderr)
            self.assertIn(reason, result.stderr)
            self.assertEqual(result.stdout, '')
        result = self.execute(self.program('array[float, 1] values = {0.5}; int index = 2; float out = exp(values[index]);'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ASSERT', result.stderr)

    def test_function_gate_parameters_with_and_without_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        theta, phi = math.exp(-0.7), math.cos(0.7)
        expected = (cmath.exp(-0.5j * phi) * math.cos(theta / 2),
                    -1j * cmath.exp(0.5j * phi) * math.sin(theta / 2))
        for fused in (False, True):
            lines = self.program('include "stdgates.inc"; qubit q; '
                'rx(exp(-0.7)) q; rz(real(exp(0.7im))) q;', fused=fused)
            BackendAuditTests.check(self, lines, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    options = parser.parse_args()
    if options.bra is not None:
        BackendTests.bra = options.bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if BackendTests.checked:
        print(f'Mathematical-function programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
