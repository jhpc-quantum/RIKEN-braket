#!/usr/bin/env python3

"""Audit integer/real mod calls against exact integer and binary-ratio references.

Run without arguments for reference tests, or pass --bra /path/to/bra for
the default-double, non-MPI audit. No math.fmod is used by the oracle.
"""

import argparse
from fractions import Fraction
import math
import pathlib
import re
import resource
import unittest

import qasm2qcx_math_function_numerical as functions


def remainder(left, right):
    if right == 0:
        raise ValueError('zero divisor')
    if isinstance(left, int) and isinstance(right, int):
        quotient = abs(left) // abs(right)
        return left - (-quotient if (left < 0) != (right < 0) else quotient) * right
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError('nonfinite input')
    dividend, divisor = Fraction(left), Fraction(right)
    result = float(dividend - int(dividend / divisor) * divisor)
    return math.copysign(result, left) if result == 0 else result


class ReferenceTests(unittest.TestCase):
    def test_integer_signs_and_endpoints(self):
        for left, right, expected in ((-7, 3, -1), (7, -3, 1), (-(2 ** 31), -1, 0), (2 ** 100 + 17, 7, 5)):
            self.assertEqual(remainder(left, right), expected)

    def test_real_signs_zero_and_extreme_ratios(self):
        for left, right, expected in ((-7.5, 3.0, -1.5), (7.5, -3.0, 1.5),
                                      (-0.0, 3.0, -0.0), (-6.0, 3.0, -0.0), (1.0, 5e-324, 0.0)):
            actual = remainder(left, right)
            self.assertEqual(actual, expected)
            self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = functions.BackendTests.setUp
    program = functions.BackendTests.program
    execute = functions.BackendTests.execute

    def check(self, source, outputs, expected, *, fused=False, allow_underflow=False):
        result = self.execute(self.program(source, tuple((name, False) for name in outputs), fused=fused))
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + source)
        actual = [float(value) for value in result.stdout.split()]
        self.assertEqual(len(actual), len(expected), result.stdout)
        for value, wanted in zip(actual, expected):
            flushed = allow_underflow and value == 0 and 0 < abs(wanted) < float.fromhex('0x1p-1022')
            self.assertTrue(flushed or math.isclose(value, wanted, rel_tol=1e-5, abs_tol=0), (source, value, wanted))
            if value == 0 or wanted == 0:
                self.assertEqual(math.copysign(1, value), math.copysign(1, wanted))
        type(self).checked += 1

    def test_integer_constants_literals_variables_and_endpoints(self):
        minimum = self.module.QASM2QCXConverter.QCX_INT_MIN
        maximum = self.module.QASM2QCXConverter.QCX_INT_MAX
        for left in (minimum, -7, -1, 0, 1, 7, maximum):
            for right in (minimum, -3, -1, 1, 3):
                for mode in ('constant', 'literal', 'variable'):
                    prefix = f'int x = {left}; int y = {right}; '
                    if mode == 'constant':
                        prefix += f'const int C = mod({left}, {right}); '
                    call = 'C' if mode == 'constant' else f'mod({left}, {right})' if mode == 'literal' else 'mod(x, y)'
                    self.check(prefix + f'int out = {call};', ('out', 'x', 'y'), (remainder(left, right), left, right))

    def test_real_constants_literals_variables_signs_and_extremes(self):
        for left in (-7.5, -6.0, -0.0, 0.0, 6.0, 7.5, 1e308, -1e308, 1e-300):
            for right in (-3.0, 3.0, 1e-300, 1e308):
                for mode in ('constant', 'literal', 'variable'):
                    prefix = f'float x = {left!r}; float y = {right!r}; '
                    if mode == 'constant':
                        prefix += f'const float C = mod({left!r}, {right!r}); '
                    call = 'C' if mode == 'constant' else f'mod({left!r}, {right!r})' if mode == 'literal' else 'mod(x, y)'
                    self.check(prefix + f'float out = {call};', ('out', 'x', 'y'), (remainder(left, right), left, right),
                               allow_underflow=True)

    def test_promotions_nesting_and_self_references(self):
        for source, expected in (
                ('int a = 7; uint b = 3; int out = mod(a, b);', 1),
                ('uint a = 7; bool b = true; int out = mod(a, b);', 0),
                ('uint[8] a = 255; int b = -7; int out = mod(a, b);', 3),
                ('int a = 7; float b = 3.5; float out = mod(a, b);', 0),
                ('bool a = true; float b = 0.75; float out = mod(a, b);', 0.25),
                ('int a = 17; int b = 7; int out = mod(mod(a, b), mod(b, 3) + 1);', 1),
                ('float a = 7.5; float b = 3.0; float out = mod(sqrt(a + 1.5), floor(b - 0.5));', 1)):
            self.check(source, ('out',), (expected,))
        self.check('int a = -7; a = mod(a, a);', ('a',), (0,))
        self.check('float a = -7.5; a = mod(a, 3); a += mod(a, a - 1);', ('a',), (-3,))

    def test_arrays_and_destination_index_capture(self):
        self.check('array[int, 3] a = {7, 3, 2}; int i = 0; '
                   'a[mod(i, 2)] = mod(a[i] + 1, a[a[2] - 1]); int out = a[0];', ('out',), (2,))
        self.check('array[float, 2] a = {7.5, 3}; int i = 0; '
                   'a[i] += mod(a[i], a[i + 1]); float out = a[i];', ('out', 'i'), (9, 0))
        self.check('array[uint[8], 2] a = {255, 7}; int i = 0; '
                   'a[i] = mod(a[i], a[i + 1]); uint out = a[i];', ('out',), (3,))

    def test_conditions_and_runtime_loop_expressions(self):
        self.check('int n = 5; int total = 0; while (mod(n, 3) != 0) { n -= 1; total += 1; }',
                   ('n', 'total'), (3, 2))
        self.check('int first = 4; int last = 7; int total = 0; '
                   'for int i in [mod(first, 3):mod(last, 4)] { total += i; }', ('total',), (6,))
        self.check('int step = 5; int total = 0; for int i in [0:mod(step, 3):4] { total += i; } '
                   'for int j in {mod(step, 3), mod(step, 4)} { total += j; }', ('total',), (9,))

    def test_skipped_invalid_calls_remain_skipped(self):
        for call in ('mod(7, 0)', 'mod(7.5, -0.0)', 'mod(7.5, 1e309)', 'mod(uint(-1), 3)', 'mod(a[i], 3.0)'):
            self.check('array[float, 1] a = {7.5}; int i = 2; bool flag = false; '
                       f'bool out = flag && bool({call});', ('out',), (0,))
            for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}',
                            'for int j in [first:last] {{ {} }}', 'for int j in [1:0] {{ {} }}'):
                self.check('array[float, 1] a = {7.5}; int i = 2; bool flag = false; '
                           'int first = 2; int last = 1; float out = 7; '
                           + wrapper.format(f'out = {call};'), ('out',), (7,))

    def test_reached_failures_name_the_native_instruction(self):
        for call, diagnostic in (
                ('mod(7, 0)', 'integer remainder by zero in LET'),
                ('mod(7.5, 0.0)', 'MOD domain error: zero divisor'),
                ('mod(7.5, -0.0)', 'MOD domain error: zero divisor'),
                ('mod(1e309, 3.0)', 'MOD domain error: nonfinite input'),
                ('mod(7.5, 1e309)', 'MOD domain error: nonfinite input'),
                ('mod(0.0 / 0.0, 3.0)', 'MOD domain error: nonfinite input'),
                ('mod(uint(-1), 3)', 'UINT to INT conversion out of range'),
                ('mod(7, uint(-1))', 'UINT to INT conversion out of range')):
            result = self.execute(self.program(f'float out = {call};', (('out', False),)))
            self.assertNotEqual(result.returncode, 0, call)
            self.assertIn(diagnostic, result.stderr)
            self.assertEqual(result.stdout, '')
            type(self).checked += 1

    def test_stored_subnormals_follow_backend_underflow_mode(self):
        for left, right in ((5e-324, 1.0), (-5e-324, 1.0), (1.0, 5e-324),
                            (1e-310, 7e-311), (-1e-310, -7e-311)):
            self.check(f'float x = {left!r}; float y = {right!r}; float out = mod(x, y);',
                       ('out',), (remainder(left, right),), allow_underflow=True)

    def test_gate_parameters_with_and_without_fusion(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        theta = remainder(7.5, 3.0)
        expected = (complex(math.cos(theta / 2)), -1j * math.sin(theta / 2))
        source = 'include "stdgates.inc"; qubit q; float x = 7.5; float y = 3; rx(mod(x, y)) q;'
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(source, fused=fused), expected)

    def test_documented_mod_example(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- mod-function-example -->\s*```qasm\n(.*?)\n```', document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        expected = (complex(math.cos(0.75)), 0j, -1j * math.sin(0.75), 0j)
        for fused in (False, True):
            BackendAuditTests.check(self, self.program(examples[0], fused=fused), expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve() if args.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if BackendTests.checked:
        print(f'mod programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
