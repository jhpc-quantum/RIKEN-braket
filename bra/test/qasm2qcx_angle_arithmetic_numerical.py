#!/usr/bin/env python3
"""Check exact scalar angle arithmetic and resulting phase gates (#398).

  python3 bra/test/qasm2qcx_angle_arithmetic_numerical.py --bra /path/to/bra

Raw UINT fixtures isolate arithmetic from floating-point angle quantization.
Expected results use independent integer alignment and modular arithmetic.
"""

import argparse
import pathlib
import resource
import unittest

import qasm2qcx_angle_global_phase_numerical as angles


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = angles.BackendTests.setUp
    program = angles.BackendTests.program
    execute = angles.BackendTests.execute
    check_phase = angles.BackendTests.check_phase
    check_zero_qubit = angles.BackendTests.check_zero_qubit

    def exact(self, source, outputs, expected, *, raw=()):
        lines = self.program(source, tuple((name, False) for name in outputs))
        # The name mapping is used only to seed declared storage, never to
        # calculate expected values or reproduce converter arithmetic.
        empty = self.module.QASM2QCXConverter(self.module.openqasm3.parser.parse('OPENQASM 3;'))
        for name, bits in raw:
            operand = empty._QASM2QCXConverter__capitalize_variable_name(name)
            declaration = lines.index(f'VAR {operand} UINT')
            lines.insert(declaration + 1, f'LET {operand} := {bits}')
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + source)
        self.assertEqual(result.stdout.split(), list(map(str, expected)), source)
        type(self).checked += 1

    def test_exact_runtime_results_at_every_width(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            modulus = 1 << width
            values = sorted({0, 1, modulus // 2 - 1, modulus // 2, modulus - 1})
            for lhs in values:
                for rhs in values:
                    source = f'''angle[{width}] a; angle[{width}] b;
                        angle[{width}] sum = a + b; angle[{width}] difference = a - b;
                        angle[{width}] negative = -a; angle[{width}] nested = -(a - (b + a));
                        angle[{width}] self = a; self = self + b; self = self - b;
                        bool zero = bool(a - a);'''
                    self.exact(source, ('a', 'b', 'sum', 'difference', 'negative', 'nested', 'self', 'zero'),
                               (lhs, rhs, (lhs + rhs) % modulus, (lhs - rhs) % modulus,
                                (-lhs) % modulus, rhs, lhs, 0), raw=(('a', lhs), ('b', rhs)))

    def test_mixed_width_promotion_and_destination_narrowing(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for left_width in sorted({1, 4, 8, native}):
            for right_width in sorted({1, 4, 8, native}):
                width, small = max(left_width, right_width), min(left_width, right_width)
                for lhs, rhs in ((0, 1), (1, 1), ((1 << left_width) - 1, 1),
                                  (1, (1 << right_width) - 1)):
                    left = lhs << (width - left_width)
                    right = rhs << (width - right_width)
                    total = (left + right) % (1 << width)
                    difference = (left - right) % (1 << width)
                    source = f'''angle[{left_width}] a; angle[{right_width}] b;
                        angle[{width}] sum = a + b; angle[{width}] difference = a - b;
                        angle[{small}] narrow_sum = a + b; angle[{small}] narrow_difference = a - b;
                        angle[{width}] negative_sum = -(a + b);'''
                    self.exact(source, ('a', 'b', 'sum', 'difference', 'narrow_sum', 'narrow_difference', 'negative_sum'),
                               (lhs, rhs, total, difference, total >> (width - small), difference >> (width - small),
                                (-total) % (1 << width)), raw=(('a', lhs), ('b', rhs)))

    def test_constants_unsized_angles_and_executable_array_casts(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        self.exact('''const angle[4] A = 15 * pi / 8; angle[4] a = A;
            const angle[8] B = pi / 128; angle[4] result = a - B;
            angle[8] widened = a + B; angle[4] negative = -a;''',
                   ('a', 'result', 'widened', 'negative'), (15, 14, 241, 1))
        self.exact('''angle a; const angle[4] B = pi / 8;
            angle sum = a + B; angle difference = a - B; angle negative = -a;''',
                   ('a', 'sum', 'difference', 'negative'),
                   (1, 1 + (1 << (native - 4)), (1 - (1 << (native - 4))) % (1 << native), (1 << native) - 1),
                   raw=(('a', 1),))
        self.exact('''array[float, 2] data = {pi, pi / 2}; int index = 1;
            angle[4] a; angle[8] sum = a + angle[8](data[index]);
            angle[4] difference = angle[8](data[index]) - a;
            angle[8] negative = -(angle[4](data[index]) + angle[8](data[index - 1]));''',
                   ('a', 'sum', 'difference', 'negative'), (15, 48, 5, 64), raw=(('a', 15),))

    def test_computed_phase_parameters_preserve_full_matrices_and_sources(self):
        declaration = 'const angle[4] A = pi; const angle[8] B = pi / 128; angle[4] a = A; angle[8] b = B; '
        for gate in ('p', 'phase', 'u1', 'cp', 'cphase', 'gphase'):
            wires = (2, 0) if gate in ('cp', 'cphase') else () if gate == 'gphase' else (1,)
            operands = ', '.join(f'q[{wire}]' for wire in wires)
            for modifier, width, bits in (('', 8, 127), ('inv @ ', 8, 129), ('pow(0.5) @ ', 9, 127)):
                for fused in (False, True):
                    self.check_phase(declaration + f'{modifier}{gate}(-(a + b)) {operands};',
                                     width, bits, (wires,), fused=fused, preserved=(('a', 8), ('b', 1)))
        for fused in (False, True):
            self.check_phase(declaration + 'negctrl @ pow(0.5) @ p(-(a + b)) q[2], q[0:1];',
                             9, 127, (((2, 0), 0), ((2, 0), 1)), fused=fused)
            self.check_zero_qubit(declaration + 'pow(0.5) @ gphase(-(a + b));', 9, 127,
                                  fused=fused, preserved=(('a', 8), ('b', 1)))

    def test_skipped_calls_and_reached_conversion_and_access_errors(self):
        for wrapper in ('if (flag) {{ {} }}', 'for int i in [1:0] {{ {} }}', 'while (flag) {{ {} }}'):
            self.exact('angle[4] a; angle[8] out; bool flag = false; array[float, 1] data = {0}; int index = 2; '
                       + wrapper.format('out = -(a + angle[8](1e309)); out = angle[8](data[index]) - a;'),
                       ('a', 'out'), (5, 0), raw=(('a', 5),))
        for expression in ('a + angle[8](1e309)', 'angle[8](data[index]) - a', '-angle[8](1e309)'):
            source = 'angle[4] a; angle[8] out; array[float, 1] data = {0}; int index = 2; '
            result = self.execute(self.program(source + f'out = {expression};') + ['DO AMPLITUDES'])
            self.assertNotEqual(result.returncode, 0, expression)
            self.assertNotIn(' => ', result.stdout)
            type(self).checked += 1
        self.exact('''angle[4] a; bool flag = false;
            bool and_result = flag && bool(a + angle[8](1e309));
            bool or_result = !flag || bool(-angle[8](1e309));''',
                   ('and_result', 'or_result'), (0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'angle arithmetic programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
