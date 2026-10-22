#!/usr/bin/env python3

"""Audit OpenQASM log calls against independent Decimal references.

Run without arguments for standard-library reference tests, or pass
--bra /path/to/bra for the default-double, non-MPI backend audit.
"""

import argparse
import cmath
import decimal
import math
import pathlib
import re
import resource
import unittest

import qasm2qcx_math_function_numerical as functions


def logarithm(value):
    """Natural logarithm of an exactly represented binary float."""
    with decimal.localcontext() as context:
        context.prec = 100
        return float(decimal.Decimal.from_float(value).ln())


class ReferenceTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(logarithm(1.0), 0)
        self.assertAlmostEqual(logarithm(2.0), 0.6931471805599453)
        self.assertEqual(logarithm(0.5), -logarithm(2.0))

    def test_extremes_and_near_one(self):
        self.assertAlmostEqual(logarithm(float.fromhex('0x0.0000000000001p-1022')),
                               -1074 * logarithm(2.0))
        for value in (math.nextafter(1.0, 0.0), math.nextafter(1.0, 2.0)):
            self.assertAlmostEqual(logarithm(value) / (value - 1), 1)


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute
    check = functions.BackendTests.check

    def test_constants_literals_variables_and_extremes(self):
        values = (0.25, 0.5, 1.0, 2.0, 10.0, 1e-310, 1e-300,
                  float.fromhex('0x0.0000000000001p-1022'),
                  float.fromhex('0x0.fffffffffffffp-1022'),
                  float.fromhex('0x1p-1022'), float.fromhex('0x1.fffffffffffffp1023'),
                  math.nextafter(1.0, 0.0), math.nextafter(1.0, 2.0))
        for value in values:
            expected = logarithm(value)
            for mode in ('constant', 'literal', 'variable'):
                with self.subTest(value=value, mode=mode):
                    source = f'float x = {value!r}; '
                    if mode == 'constant':
                        source += f'const float C = log({value!r}); '
                    expression = 'C' if mode == 'constant' else 'log(x)' if mode == 'variable' else f'log({value!r})'
                    # Normalize tiny near-one answers so a zero cannot pass
                    # the general CLI audit's absolute output tolerance.
                    scale = expected if expected else 1.0
                    source += f'float out = {expression}; out /= {scale!r};'
                    self.check(source, (('out', False), ('x', False)),
                               (expected / scale, value), signed_zero=True)

    def test_promotions_nesting_arrays_and_source_preservation(self):
        for declaration, value in (('int x = 2;', 2.0), ('uint x = 4294967295;', 4294967295.0),
                                   ('bool x = true;', 1.0)):
            self.check(declaration + ' complex out = log(x);', (('out', True), ('x', False)),
                       (logarithm(value), 0, value))
        self.check('bit b = 1; float out = log(float(b));', (('out', False),), (0,))
        self.check('float x = 2; float out = log(sqrt(exp(x))); x = log(x);',
                   (('out', False), ('x', False)), (1, logarithm(2.0)))
        self.check('array[float, 2] values = {2, 3}; array[complex, 2] zs = {2 + 1im, 3}; int i = 0; '
                   'values[i] = log(values[i]) + log(real(zs[i])); values[i] += log(values[i]); '
                   'float out = values[i]; complex saved = zs[i];',
                   (('out', False), ('saved', True), ('i', False)),
                   (2 * logarithm(2.0) + logarithm(2 * logarithm(2.0)), 2, 1, 0))

    def test_loops_and_runtime_iteration_expressions(self):
        self.check('float x = 3; float total = 0; while (log(x) > 0) { total += log(x); x -= 1; }',
                   (('total', False), ('x', False)), (logarithm(3.0) + logarithm(2.0), 1))
        self.check('float x = 8; int total = 0; for int i in [0:int(log(x)):4] { total += i; } '
                   'for int j in {int(log(x)), 3} { total += j; }', (('total', False),), (11,))
        self.check('array[float, 2] values = {2, 3}; float total = 0; '
                   'for float x in values { total += log(x); }', (('total', False),),
                   (logarithm(2.0) + logarithm(3.0),))

    def test_gate_parameters_with_and_without_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        theta, phi = logarithm(2.0), logarithm(0.5)
        expected = (cmath.exp(-0.5j * phi) * math.cos(theta / 2),
                    -1j * cmath.exp(0.5j * phi) * math.sin(theta / 2))
        for fused in (False, True):
            lines = self.program('include "stdgates.inc"; qubit q; float x = 2; '
                                 'rx(log(x)) q; rz(log(0.5)) q;', fused=fused)
            BackendAuditTests.check(self, lines, expected)

    def test_documented_logarithm_example(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- log-function-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        theta, phi = logarithm(2.0), logarithm(0.5)
        expected = (cmath.exp(-0.5j * phi) * math.cos(theta / 2),
                    -1j * cmath.exp(0.5j * phi) * math.sin(theta / 2))
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(examples[0], fused=fused), expected)

    def test_skipped_calls_and_reached_errors(self):
        for expression, expected in (('false && bool(log(0))', 0), ('true || (log(-1) > 0)', 1),
                                     ('true || (log(1e309) > 0)', 1)):
            self.check('bool out = ' + expression + ';', (('out', False),), (expected,))
        for call in ('out = log(0);', 'out = log(1e309);', 'out = log(values[i]);'):
            for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}',
                            'for int j in [first:last] {{ {} }}', 'for int j in [1:0] {{ {} }}'):
                self.check('bool flag = false; int first = 2; int last = 1; float out = 7; '
                           'array[float, 1] values = {2}; int i = 2; ' + wrapper.format(call),
                           (('out', False),), (7,))
        for argument in ('0', '-0.0', '-1', '-5e-324', 'false', '1e309', '-1e309', '0.0 / 0.0',
                         'log(1)', 'real(1.0 / 0.0 + 1im)'):
            result = self.execute(self.program(f'float out = log({argument});', (('out', False),)))
            self.assertNotEqual(result.returncode, 0, argument)
            self.assertIn('LOG', result.stderr)
            self.assertIn('nonfinite input' if argument in ('1e309', '-1e309', '0.0 / 0.0',
                                                          'real(1.0 / 0.0 + 1im)')
                          else 'nonpositive real input', result.stderr)
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
        print(f'LOG programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
