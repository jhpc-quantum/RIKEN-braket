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

    def test_documented_global_phase_examples(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.md').read_text()
        for marker, qubits in (('zero', 0), ('controlled', 3)):
            examples = re.findall(r'<!-- angle-global-phase-' + marker + r'-example -->\s*```qasm\n(.*?)\n```',
                                  document, flags=re.DOTALL)
            self.assertEqual(len(examples), 1)
            for fused in (False, True):
                lines = self.program(examples[0])
                self.assertEqual(lines[0], f'QUBITS {qubits}')
                if fused:
                    lines.insert(1, 'BEGIN FUSION')
                    lines.insert(len(lines) - 1, 'END FUSION')  # Flush before pragma-requested output.
                result = self.execute(lines)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = {int(index, 2) if index else 0: complex(float(real), float(imaginary))
                          for index, real, imaginary in
                          re.findall(r'^([01]*) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
                self.assertEqual(set(actual), set(range(1 << qubits)))
                for index in range(1 << qubits):
                    if not qubits:
                        expected = -1j
                    else:
                        eighths = 5 if not index & 4 else 0
                        eighths -= 5 if index & 1 and index & 2 else 0
                        expected = cmath.exp(1j * eighths * math.pi / 8) / math.sqrt(8)
                    self.assertTrue(cmath.isclose(actual[index], expected, rel_tol=1e-5, abs_tol=2e-6))
                type(self).checked += 1

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

    def test_inverse_and_controlled_adjoint_combinations(self):
        for parameter in ('A', 'live'):
            declaration = 'const angle[4] A = 5 * pi / 8; angle[4] live = A; '
            for modifiers, power in (('inv @', -1), ('inv @ inv @', 1),
                                     ('pow(-3) @', -3), ('inv @ pow(-3) @', 3)):
                for fused in (False, True):
                    self.check_zero_qubit(declaration + modifiers + f' gphase({parameter});',
                                          4, (5 * power) % 16, fused=fused, preserved=(('live', 5),))
                    self.check_phase(declaration + f'negctrl @ {modifiers} gphase({parameter}) q[2];',
                                     4, (5 * power) % 16, (((2, 0),),), fused=fused, preserved=(('live', 5),))
            for fused in (False, True):
                self.check_phase(declaration + f'negctrl(2) @ ctrl @ inv @ gphase({parameter}) q[3], q[1], q[0];',
                                 4, 11, (((3, 0), (1, 0), 0),), fused=fused,
                                 preserved=(('live', 5),), qubits=4)

    def test_integer_power_boundaries_for_uncontrolled_and_controlled_phases(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        minimum = self.module.QASM2QCXConverter.QCX_INT_MIN
        signed_max = self.module.QASM2QCXConverter.QCX_INT_MAX
        maximum = self.module.QASM2QCXConverter.QCX_UINT_MAX
        for width in (1, 4, native):
            bits = (1 << width) - 1
            radians = math.ldexp(math.tau * bits, -width)
            for power in (-3, -1, 0, 1, 2, minimum, signed_max, maximum):
                exponent_type = 'uint' if power > signed_max else 'int'
                declaration = (f'const angle[{width}] A = {radians!r}; angle[{width}] live = A; '
                               + f'{exponent_type} exponent = {power}; ')
                expected = (bits * power) % (1 << width)
                for parameter in ('A', 'live'):
                    for exponent in (str(power), 'exponent'):
                        for fused in (False, True):
                            self.check_zero_qubit(declaration + f'pow({exponent}) @ gphase({parameter});',
                                                  width, expected, fused=fused, preserved=(('live', bits),))
                            self.check_phase(declaration + f'negctrl @ ctrl @ pow({exponent}) @ gphase({parameter}) q[2], q[0];',
                                             width, expected, (((2, 0), 0),), fused=fused, preserved=(('live', bits),))
            for fused in (False, True):
                huge = (1 << 100) + 1
                source = (f'const angle[{width}] A = {radians!r}; angle[{width}] live = A; '
                          + f'int a = {minimum + 1}; uint b = {maximum}; '
                          + f'inv @ pow(a) @ pow(b) @ pow({huge}) @ gphase(live);')
                expected = (-bits * (minimum + 1) * maximum * huge) % (1 << width)
                self.check_zero_qubit(source, width, expected, fused=fused, preserved=(('live', bits),))

    def test_exact_powered_bits_at_every_width(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        minimum = self.module.QASM2QCXConverter.QCX_INT_MIN
        maximum = self.module.QASM2QCXConverter.QCX_UINT_MAX
        huge = (1 << 100) + 1
        cases = (('int a = -1;', 'pow(a) @', -1, 1),
                 (f'int a = {minimum};', 'inv @ pow(a) @', minimum, -1),
                 (f'uint a = {maximum};', 'pow(a) @ pow(3) @', maximum, 3),
                 (f'int a = {minimum + 1}; uint b = {maximum};',
                  f'pow(a) @ pow(b) @ pow({huge}) @', (minimum + 1) * maximum, huge),
                 ('int a = -1;', 'pow(0) @ pow(a) @', -1, 0))
        for width in range(1, native + 1):
            for bits in sorted({1, (1 << width) - 1}):
                radians = math.ldexp(math.tau * bits, -width)
                for declarations, modifiers, runtime_power, constant_power in cases:
                    source = (f'const angle[{width}] A = {radians!r}; angle[{width}] live = A; '
                              + declarations + modifiers + ' gphase(live);')
                    lines = self.program(source, (('live', False),))
                    jump = next(index for index, line in enumerate(lines)
                                if line.startswith('JUMPIF ') and '_GLOBAL_ANGLE_1_SKIP ' in line)
                    # Observe the powered private UINT snapshot, not rounded
                    # complex output, including all native-width low bits.
                    snapshot = lines[jump - 2].split()[-1]
                    lines.insert(jump - 2, f'PRINTLN {snapshot}')
                    result = self.execute(lines)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = (bits * runtime_power * abs(constant_power)) % (1 << width)
                    self.assertEqual(result.stdout.split(), [str(expected), str(bits)], source)
                    type(self).checked += 1

    def test_power_reentry_array_iterators_and_required_input_checks(self):
        for fused in (False, True):
            self.check_zero_qubit('''const angle[4] A = 5 * pi / 8; angle[4] live = A;
                array[int, 3] exponents = {2, -3, 0};
                for int exponent in exponents { pow(exponent) @ gphase(live); }''',
                                  4, 11, fused=fused, preserved=(('live', 5),))
            self.check_zero_qubit('''const angle[4] A = 5 * pi / 8; angle[4] live = A;
                int exponent = -3; int count = 0;
                while (count < 2) { pow(exponent) @ gphase(live); exponent = 2; count += 1; }''',
                                  4, 11, fused=fused, preserved=(('live', 5),))
            self.check_phase('''const angle[4] A = 5 * pi / 8; angle[4] live = A;
                array[int, 2] exponents = {0, -3}; int index = 1;
                negctrl @ ctrl @ pow(exponents[index]) @ gphase(live) q[2], q[0];''',
                             4, 1, (((2, 0), 0),), fused=fused, preserved=(('live', 5),))
            self.check_zero_qubit('''bool flag = false; int zero = 0;
                if (flag) { pow(0) @ pow(1 / zero) @ gphase(angle[4](1e309)); }
                for int i in [1:0] { pow(0) @ gphase(angle[4](1e309)); }''', 4, 0, fused=fused)
            for body in ('pow(0) @ gphase(angle[4](1e309));',
                         'pow(0) @ pow(1 / zero) @ gphase(A);',
                         'ctrl @ pow(0) @ gphase(angle[4](1e309)) q;',
                         'x q; negctrl @ pow(0) @ gphase(angle[4](1e309)) q;'):
                source = 'include "stdgates.inc"; qubit q; int zero = 0; const angle[4] A = pi; ' + body
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
