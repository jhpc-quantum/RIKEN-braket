#!/usr/bin/env python3
"""Check exact runtime angle multiplication and division (#400).

Run with --bra /path/to/bra. Raw fixtures avoid angle quantization dependence.
"""

import argparse
import pathlib
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
