#!/usr/bin/env python3
"""Check exact runtime angle multiplication and division (#400).

Run with --bra /path/to/bra. Raw fixtures avoid angle quantization dependence.
"""

import argparse
import cmath
import math
import pathlib
import re
import resource
import unittest

import qasm2qcx_angle_arithmetic_numerical as arithmetic


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = arithmetic.BackendTests.setUp
    program = arithmetic.BackendTests.program
    execute = arithmetic.BackendTests.execute
    exact = arithmetic.BackendTests.exact
    check_phase = arithmetic.BackendTests.check_phase
    check_zero_qubit = arithmetic.BackendTests.check_zero_qubit

    def test_documented_multiplicative_examples(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.md').read_text()
        examples = {}
        for marker in ('angle-multiplicative-example', 'angle-multiplicative-phase-example'):
            matches = re.findall(r'<!-- ' + marker + r' -->\s*```qasm\n(.*?)\n```', document, flags=re.DOTALL)
            self.assertEqual(len(matches), 1)
            examples[marker] = matches[0]
        self.exact(examples['angle-multiplicative-example'], ('a', 'factor', 'product', 'divided', 'quotient'),
                   (2, 2, 14, 3, 7))
        for fused in (False, True):
            lines = self.program(examples['angle-multiplicative-phase-example'])
            if fused:
                lines.insert(1, 'BEGIN FUSION')
                lines.insert(len(lines) - 1, 'END FUSION')
            result = self.execute(lines)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = {int(index, 2): complex(float(real), float(imaginary))
                      for index, real, imaginary in re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
            self.assertEqual(set(actual), set(range(8)), result.stdout)
            for index in range(8):
                phase = (7 * math.pi / 8 if index & 1 else 0)
                phase += 7 * math.pi / 8 if index & 6 == 6 else 0
                phase += 7 * math.pi / 16 if not index & 4 else 0
                self.assertTrue(cmath.isclose(actual[index], cmath.exp(1j * phase) / math.sqrt(8),
                                             rel_tol=1e-5, abs_tol=2e-6), (index, actual[index]))
            type(self).checked += 1

    def test_scaling_and_ratios_at_every_width(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            for bits in sorted({0, 1, modulus // 2, modulus - 1}):
                for factor in sorted({1, min(3, modulus - 1), modulus - 1}):
                    source = f'''angle[{width}] a; angle[{width}] b;
                        uint[{width}] u = {factor};
                        angle[{width}] product = a * u;
                        angle[{width}] reverse = u * a;
                        angle[{width}] quotient = a / u;
                        uint[{width}] ratio = a / b;'''
                    self.exact(source, ('a', 'b', 'u', 'product', 'reverse', 'quotient', 'ratio'),
                               (bits, factor, factor, bits * factor % modulus,
                                bits * factor % modulus, bits // factor, bits // factor),
                               raw=(('a', bits), ('b', factor)))

    def test_ratio_promotion_and_uint_composition(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for left_width in sorted({1, 4, 8, native}):
            for right_width in sorted({1, 4, 8, native}):
                width = max(left_width, right_width)
                left_bits, right_bits = (1 << left_width) - 1, 1
                ratio = (left_bits << (width - left_width)) // (right_bits << (width - right_width))
                self.exact(f'''angle[{left_width}] a; angle[{right_width}] b;
                    uint[{width}] ratio = a / b;
                    uint[{width}] product = (a / b) * uint[{width}](3);
                    uint count = popcount(a / b);''', ('a', 'b', 'ratio', 'product', 'count'),
                    (left_bits, right_bits, ratio, ratio * 3 % (1 << width), ratio.bit_count()),
                    raw=(('a', left_bits), ('b', right_bits)))

    def test_reentry_and_array_factor(self):
        self.exact('''angle[4] a; array[uint[4], 2] factors = {2, 3};
            int index = 0; angle[4] result;
            while (index < 2) { result = a * factors[index]; index += 1; }''',
            ('a', 'result'), (7, 5), raw=(('a', 7),))

    def test_zero_division_only_fails_when_reached(self):
        for expression, declaration in (
                ('a / uint[4](0)', 'angle[4] result'),
                ('angle[4](pi) / uint[4](0)', 'angle[4] result'),
                ('a / angle[4](0.0)', 'uint[4] result'),
                ('angle[4](pi) / angle[4](0.0)', 'uint[4] result'),
                ('A / Z', 'angle[4] result'),
                ('A / AZ', 'uint[4] result'),
                ('(a / uint[4](0)) * uint[4](0)', 'angle[4] result')):
            statement = f'{declaration} = {expression};'
            constants = 'const angle[4] A = pi; const uint[4] Z = 0; const angle[4] AZ = 0.0; '
            result = self.execute(self.program(constants + 'angle[4] a = pi; ' + statement))
            self.assertNotEqual(result.returncode, 0, expression)
            type(self).checked += 1
            for wrapper in ('if (false) { %s }', 'while (false) { %s }',
                            'for int i in [1:0] { %s }'):
                self.exact(constants + f'angle[4] a; {declaration}; ' + wrapper % f'result = {expression};',
                           ('a',), (7,), raw=(('a', 7),))
        self.exact('''angle[4] a; uint[4] zero = 0;
            bool lhs = false && bool(a / zero);
            bool rhs = true || bool(a / zero);''', ('a', 'lhs', 'rhs'), (7, 0, 1), raw=(('a', 7),))

    def test_scaled_phase_consumers(self):
        for fused in (False, True):
            for gate in ('p', 'phase', 'u1'):
                self.check_phase(f'angle[4] a = pi; uint[4] factor = 3; {gate}(a * factor) q[0];',
                                 4, 8, ((0,),), fused=fused)
                self.check_phase(f'angle[4] a = pi; uint[4] factor = 2; negctrl @ {gate}(a / factor) q[1], q[0];',
                                 4, 4, (((1, 0), 0),), fused=fused)
            self.check_zero_qubit('angle[4] a = pi; uint[4] factor = 2; gphase(a / factor);',
                                  4, 4, fused=fused)

    def test_compound_results_at_every_width_and_aliasing(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            for bits in sorted({0, 1, modulus // 2, modulus - 1}):
                factor = min(3, modulus - 1)
                self.exact(f'''angle[{width}] a; uint[{width}] u = {factor};
                    angle[{width}] product = a; product *= u;
                    angle[{width}] quotient = a; quotient /= u;
                    angle[{width}] nested = a; nested *= a / angle[{width}](pi);
                    angle[{width}] self = a; self /= uint[{width}](1);''',
                    ('a', 'u', 'product', 'quotient', 'nested', 'self'),
                    (bits, factor, bits * factor % modulus, bits // factor,
                     bits * (bits // (modulus // 2)) % modulus, bits), raw=(('a', bits),))

    def test_compound_reentry_break_continue_and_array_reads(self):
        self.exact('''angle[4] a; array[uint[4], 3] factors = {2, 3, 5}; int index = 0;
            while (index < 3) { a *= factors[index]; index += 1; }
            for uint[4] u in factors { a /= u; }''', ('a',), (0,), raw=(('a', 7),))
        self.exact('''angle[4] a; uint[4] factor = 3;
            for int i in [0:4] { if (i == 1) { continue; } a *= factor; if (i == 3) { break; } }
            a /= uint[4](2);''', ('a', 'factor'), (6, 3), raw=(('a', 7),))
        for operator in ('*=', '/='):
            self.exact(f'''angle[4] a; const uint[4] U = 2;
                array[uint[4], 2] factors = {{2, 3}};
                for uint[4] U in factors {{ a {operator} U; }}''',
                ('a',), (10 if operator == '*=' else 1,), raw=(('a', 7),))
        for wrapper in ('if (false) { %s }', 'while (false) { %s }', 'for int i in [1:0] { %s }'):
            self.exact('angle[4] a; ' + wrapper % 'a /= uint[4](0);', ('a',), (7,), raw=(('a', 7),))
        result = self.execute(self.program('angle[4] a = pi; a /= uint[4](0);'))
        self.assertNotEqual(result.returncode, 0)
        type(self).checked += 1

    def test_compound_phase_matrices_and_power_modifiers(self):
        prefix = 'angle[4] a = 5 * pi / 8; uint[4] u = 3; a *= u; a /= uint[4](2); '
        for fused in (False, True):
            for gate in ('p', 'phase', 'u1', 'cp', 'cphase'):
                controlled = gate in ('cp', 'cphase')
                wires, operands = ((2, 0), 'q[2], q[0]') if controlled else ((0,), 'q[0]')
                for basis in (*range(8), None):
                    self.check_phase(prefix + f'{gate}(a) {operands};', 4, 7, (wires,),
                                     fused=fused, basis=basis, preserved=(('a', 7), ('u', 3)))
            self.check_phase(prefix + 'negctrl @ inv @ pow(0.5) @ p(a) q[2], q[0];',
                             5, 25, (((2, 0), 0),), fused=fused)
            self.check_phase(prefix + 'uint[4] exponent = a / angle[4](pi / 4); pow(exponent) @ p(a) q[0];',
                             4, 5, ((0,),), fused=fused)
            self.check_phase(prefix + 'p(a) q;', 4, 7, ((0,), (1,), (2,)), fused=fused)
            self.check_phase(prefix + 'gphase(a);', 4, 7, ((),), fused=fused)
            self.check_phase(prefix + 'negctrl @ gphase(a) q[1];',
                             4, 7, (((1, 0),),), fused=fused)
            self.check_zero_qubit(prefix + 'gphase(a);', 4, 7, fused=fused, preserved=(('a', 7), ('u', 3)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle multiplicative programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
