#!/usr/bin/env python3

"""Check absolute and controlled angle-valued global phases (#394).

  python3 bra/test/qasm2qcx_angle_global_phase_numerical.py --bra /path/to/bra
"""

import argparse
import cmath
import math
import pathlib
import re
import resource
import unittest
from fractions import Fraction

import qasm2qcx_angle_phase_numerical as phases


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = phases.BackendTests.setUp
    program = phases.BackendTests.program
    execute = phases.BackendTests.execute
    check_phase = phases.BackendTests.check_phase

    def check_zero_qubit(self, source, width, bits, *, fused=False, preserved=()):
        lines = self.program(source, tuple((name, False) for name, _ in preserved), fused=fused) + ['DO AMPLITUDES']
        self.assertEqual(lines[0], 'QUBITS 0')
        result = self.execute(lines)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + source)
        if preserved:
            self.assertEqual(re.findall(r'^([0-9]+)$', result.stdout, re.MULTILINE),
                             [str(value) for _, value in preserved])
        # The sole zero-qubit basis state has an empty bit-string label.
        actual = re.findall(r'^ => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)
        self.assertEqual(len(actual), 1, result.stdout)
        signed = bits if bits <= (1 << (width - 1)) else bits - (1 << width)
        expected = cmath.exp(1j * math.ldexp(math.tau * signed, -width))
        self.assertTrue(cmath.isclose(complex(*map(float, actual[0])), expected,
                                     rel_tol=1e-5, abs_tol=2e-6), (source, actual, expected))
        type(self).checked += 1

    def test_zero_qubit_constants_widths_patterns_and_fusion(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in sorted({1, 4, 8, native}):
            for bits in sorted({0, 1, 1 << (width - 1), (1 << width) // 3, (1 << width) - 1}):
                radians = math.ldexp(math.tau * bits, -width)
                for fused in (False, True):
                    with self.subTest(width=width, bits=bits, fused=fused):
                        self.check_zero_qubit(f'const angle[{width}] A = {radians!r}; gphase(A);',
                                              width, bits, fused=fused)

    def test_every_width_single_bit_on_zero_qubits(self):
        for width in range(1, self.module.QASM2QCXConverter.QCX_UINT_WIDTH + 1):
            # Every exponent is checked on a state with no qubits. Include
            # both the least-significant and most-significant contributions.
            for bit in sorted({0, width - 1}):
                radians = math.ldexp(math.tau, bit - width)
                self.check_zero_qubit(f'const angle[{width}] A = {radians!r}; gphase(A);',
                                      width, 1 << bit)

    def test_global_and_mixed_controlled_diagonal_phases(self):
        controls = (('', (), ''), ('ctrl @', (2,), 'q[2]'),
                    ('negctrl @', ((1, 0),), 'q[1]'),
                    ('ctrl @ negctrl @', (2, (0, 0)), 'q[2], q[0]'),
                    ('negctrl(3) @', ((2, 0), (0, 0), (1, 0)), 'q[2], q[0], q[1]'),
                    ('negctrl @ ctrl(2) @', ((2, 0), 0, 1), 'q[2], q[0], q[1]'))
        for width in sorted({1, 4, self.module.QASM2QCXConverter.QCX_UINT_WIDTH}):
            for bits in sorted({0, 1, (1 << width) // 3, (1 << width) - 1}):
                radians = math.ldexp(math.tau * bits, -width)
                for modifier, wires, operands in controls:
                    source = f'const angle[{width}] A = {radians!r}; {modifier} gphase(A) {operands};'
                    for fused in (False, True):
                        self.check_phase(source, width, bits, (wires,), fused=fused)
                    for basis in range(8):
                        self.check_phase(source, width, bits, (wires,), basis=basis)

    def test_width_casts_wrapped_inputs_and_repeated_global_phases(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for fused in (False, True):
            for expression, width, bits in (('angle[8](A)', 8, 80), ('angle[2](A)', 2, 1),
                                             ('angle(A)', native, 5 << (native - 4))):
                self.check_zero_qubit('const angle[4] A = 5 * pi / 8; gphase(' + expression + ');',
                                      width, bits, fused=fused)
            self.check_zero_qubit('const angle[4] A = -pi / 2; gphase(A);', 4, 12, fused=fused)
            self.check_zero_qubit('const angle A = pi; gphase(A);', native, 1 << (native - 1), fused=fused)
            self.check_zero_qubit('const angle[4] A = tau; gphase(A);', 4, 0, fused=fused)
            self.check_zero_qubit('const angle[4] A = 5 * pi / 8; gphase(A); gphase(A);',
                                  4, 10, fused=fused)

    def test_controlled_broadcasting_and_classical_control_flow(self):
        for fused in (False, True):
            self.check_phase('const angle[4] A = 5 * pi / 8; ctrl @ gphase(A) q;',
                             4, 5, ((0,), (1,), (2,)), fused=fused)
            self.check_phase('const angle[4] A = 5 * pi / 8; negctrl @ ctrl @ gphase(A) q[2], q[0:1];',
                             4, 5, (((2, 0), 0), ((2, 0), 1)), fused=fused)
            self.check_phase('''const angle[4] A = 5 * pi / 8; bool flag = true;
                if (flag) { gphase(A); } else { ctrl @ gphase(A) q; }
                for int i in [1:0] { gphase(A); }
                while (false) { gphase(A); }''', 4, 5, ((),), fused=fused)
            self.check_phase('const angle[4] A = 5 * pi / 8; for int i in [0:1] { ctrl @ gphase(A) q[i]; }',
                             4, 5, ((0,), (1,)), fused=fused)
            self.check_phase('const angle[4] A = 5 * pi / 8; int count = 0; '
                             'while (count < 2) { gphase(A); count += 1; }',
                             4, 10, ((),), fused=fused)

    def test_runtime_patterns_zero_qubits_and_mixed_controls(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in sorted({1, 4, 8, native}):
            for bits in sorted({0, 1, 1 << (width - 1), (1 << width) // 3, (1 << width) - 1}):
                radians = math.ldexp(math.tau * bits, -width)
                declaration = f'const angle[{width}] A = {radians!r}; angle[{width}] live = A; '
                for fused in (False, True):
                    self.check_zero_qubit(declaration + 'gphase(live);', width, bits,
                                          fused=fused, preserved=(('live', bits),))
                    self.check_phase(declaration + 'gphase(live);', width, bits, ((),),
                                     fused=fused, preserved=(('live', bits),))
                    self.check_phase(declaration + 'negctrl @ gphase(live) q[1];',
                                     width, bits, (((1, 0),),), fused=fused, preserved=(('live', bits),))
                    self.check_phase(declaration + 'negctrl @ ctrl @ gphase(live) q[2], q[0];',
                                     width, bits, (((2, 0), 0),), fused=fused, preserved=(('live', bits),))

    def test_runtime_float_casts_use_native_quantization(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for width in (1, 4, 8, native):
            tie = math.ldexp(math.tau, -width - 1)
            for value in (0.0, -math.pi / 2, 0.1, 1e308, tie, math.nextafter(tie, math.inf)):
                bits = round(Fraction(value) / Fraction(math.tau) * (1 << width)) % (1 << width)
                for expression in (repr(value), 'value'):
                    self.check_zero_qubit(f'float value = {value!r}; gphase(angle[{width}]({expression}));',
                                          width, bits)

    def test_runtime_width_casts_arrays_broadcasting_and_reentry(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for fused in (False, True):
            for expression, width, bits in (('angle[8](live)', 8, 80), ('angle[2](live)', 2, 1),
                                             ('angle(live)', native, 5 << (native - 4))):
                self.check_zero_qubit('const angle[4] A = 5 * pi / 8; angle[4] live = A; '
                                      + 'gphase(' + expression + ');', width, bits,
                                      fused=fused, preserved=(('live', 5),))
            self.check_phase('const angle[4] A = 5 * pi / 8; angle[4] live = A; ctrl @ gphase(live) q;',
                             4, 5, ((0,), (1,), (2,)), fused=fused, preserved=(('live', 5),))
            self.check_phase('const angle[4] A = 5 * pi / 8; angle[4] live = A; '
                             'negctrl @ ctrl @ gphase(live) q[2], q[0:1];',
                             4, 5, (((2, 0), 0), ((2, 0), 1)), fused=fused, preserved=(('live', 5),))
            self.check_zero_qubit('''array[float, 2] values = {0.0, 5 * pi / 8}; int index = 1;
                gphase(angle[8](angle[4](values[index])));''', 4, 5, fused=fused)
            self.check_zero_qubit('''angle[4] live = pi / 2; int count = 0;
                while (count < 2) { gphase(live); live = pi; count += 1; }''',
                                  4, 12, fused=fused, preserved=(('live', 8),))
            self.check_zero_qubit('''array[float, 2] values = {pi / 2, pi};
                for float item in values { gphase(angle[4](item)); }''', 4, 12, fused=fused)
            self.check_zero_qubit('''const angle[4] A = pi / 2; array[float, 1] values = {pi};
                for float A in values { gphase(angle[4](A)); } gphase(A);''', 4, 12, fused=fused)

    def test_skipped_and_reached_runtime_conversion_and_access_errors(self):
        for fused in (False, True):
            self.check_zero_qubit('''bool flag = false; angle[4] live = pi / 2;
                if (flag) { gphase(angle[4](1e309)); }
                while (flag) { gphase(angle[4](1.0 / 0.0)); }
                array[float, 1] values = {0.0}; int index = 2;
                for int i in [1:0] { gphase(angle[4](values[index])); }
                gphase(live);''', 4, 4, fused=fused, preserved=(('live', 4),))
            for body in ('gphase(angle[4](1e309));', 'gphase(angle[4](1.0 / 0.0));',
                         'gphase(angle[4](values[index]));', 'ctrl @ gphase(angle[4](1e309)) q;',
                         'x q; negctrl @ gphase(angle[4](1e309)) q;'):
                source = ('include "stdgates.inc"; qubit q; array[float, 1] values = {0.0}; int index = 2; ' + body)
                result = self.execute(self.program(source, fused=fused) + ['DO AMPLITUDES'])
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn(' => ', result.stdout)
                type(self).checked += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    BackendTests.bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    print(f'angle-global-phase programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
