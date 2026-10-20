#!/usr/bin/env python3

"""Audit U-family phase conventions against independent matrix definitions.

Run reference tests alone, or also execute native/converted circuits:
  python3 bra/test/qasm2qcx_u_phase_numerical.py
  python3 bra/test/qasm2qcx_u_phase_numerical.py --bra /path/to/bra

Convention audit:
* The published built-in U matrix is exp(i*theta/2) times conventional U3.
  https://openqasm.com/versions/3.0/language/gates.html#built-in-gates
* QCX U3 (ket::gate::phase_shift3) implements conventional U3, as does the
  converter's existing u3 alias. U2 is its theta=pi/2 specialization.
* The published cu mathematical definition has active block exp(i*gamma)*U.
  These regressions follow that definition, not the example library source.
  https://openqasm.com/language/standard_library.html#gate-cu
* The example stdgates.inc instead defines cu with p(gamma-theta/2), yielding
  exp(i*gamma) times conventional U3. Its u2/u3 phase corrections also disagree
  with the stated OpenQASM 2 compatibility when combined with the built-in U
  matrix. Preserve existing conventional u2/u3 behavior; do not use these
  inconsistent source relations to infer U's phase correction.
  https://github.com/openqasm/openqasm/blob/main/examples/stdgates.inc

The old inverse/power references deliberately preserve the converter's existing
phase convention. Do not import their expected matrices into this audit.
"""

import argparse
import cmath
import math
import pathlib
import re
import resource
import subprocess
import unittest

from qasm2qcx_runtime_for_numerical import qasm2qcx


def builtin_u_matrix(theta, phi, lam):
    """Transcribe the published matrix directly, without a QCX phase formula."""
    phase = cmath.exp(1j * theta)
    return [[(1 + phase) / 2, -1j * cmath.exp(1j * lam) * (1 - phase) / 2],
            [1j * cmath.exp(1j * phi) * (1 - phase) / 2,
             cmath.exp(1j * (phi + lam)) * (1 + phase) / 2]]


def conventional_u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return [[c, -cmath.exp(1j * lam) * s],
            [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c]]


def documented_cu_active_matrix(theta, phi, lam, gamma):
    return [[cmath.exp(1j * gamma) * value for value in row]
            for row in builtin_u_matrix(theta, phi, lam)]


def apply_matrix(state, matrix, target, controls=()):
    """Apply a 2x2 matrix with explicit (wire, activating value) controls."""
    result = state.copy()
    for zero in range(len(state)):
        if zero & (1 << target) or any((zero >> wire) & 1 != value for wire, value in controls):
            continue
        one = zero | (1 << target)
        result[zero] = matrix[0][0] * state[zero] + matrix[0][1] * state[one]
        result[one] = matrix[1][0] * state[zero] + matrix[1][1] * state[one]
    return result


class ReferenceTests(unittest.TestCase):
    def assert_matrix_close(self, actual, expected):
        for row, reference in zip(actual, expected):
            for value, wanted in zip(row, reference):
                self.assertAlmostEqual(value, wanted, places=13)

    def test_builtin_u_special_cases(self):
        self.assert_matrix_close(builtin_u_matrix(0, 0, math.pi), [[1, 0], [0, -1]])
        # A phi+lambda correction cannot pass both these special cases.
        self.assert_matrix_close(builtin_u_matrix(math.pi, 0, math.pi), [[0, 1j], [1j, 0]])

    def test_published_matrix_and_conventional_u3_relation(self):
        for angles in ((0.2, 0.4, 0.6), (-0.7, 1.1, -0.3), (2 * math.pi, 0.3, 0.8)):
            with self.subTest(angles=angles):
                conventional = conventional_u3_matrix(*angles)
                expected = [[cmath.exp(1j * angles[0] / 2) * value for value in row]
                            for row in conventional]
                self.assert_matrix_close(builtin_u_matrix(*angles), expected)

    def test_documented_cu_and_example_source_are_different(self):
        theta, phi, lam, gamma = 0.7, 0.4, -0.2, 0.3
        documented = documented_cu_active_matrix(theta, phi, lam, gamma)
        example = [[cmath.exp(1j * (gamma - theta / 2)) * value for value in row]
                   for row in builtin_u_matrix(theta, phi, lam)]
        conventional = [[cmath.exp(1j * gamma) * value for value in row]
                        for row in conventional_u3_matrix(theta, phi, lam)]
        self.assert_matrix_close(example, conventional)
        self.assertGreater(abs(documented[0][0] - example[0][0]), 0.1)


class BackendTests(unittest.TestCase):
    bra = None

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to execute amplitude regressions')

    def check_qcx(self, lines, expected):
        result = subprocess.run([str(self.bra)], input='\n'.join([*lines, 'DO AMPLITUDES']) + '\n',
                                capture_output=True, text=True, check=True, timeout=30)
        actual = {int(index, 2): complex(float(real), float(imag)) for index, real, imag in
                  re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
        self.assertEqual(set(actual), set(range(len(expected))))
        for index, wanted in enumerate(expected):
            self.assertTrue(cmath.isclose(actual[index], wanted, rel_tol=1e-5, abs_tol=2e-6),
                            f'amplitude {index}: actual={actual[index]}, expected={wanted}\n'
                            + '\n'.join(lines))

    def check_source(self, source, expected):
        self.check_qcx(qasm2qcx.convert('OPENQASM 3; include "stdgates.inc"; ' + source), expected)

    def test_native_u2_u3_and_converter_aliases_keep_conventional_matrices(self):
        # Reconstruct every column, rather than checking only one input state.
        for gate, angles in (('u2', (0.4, -0.7)), ('u3', (0.3, 0.4, -0.7))):
            matrix = conventional_u3_matrix(*(angles if gate == 'u3' else (math.pi / 2, *angles)))
            arguments = ', '.join(map(str, angles))
            for basis in (0, 1):
                with self.subTest(gate=gate, basis=basis):
                    expected = [matrix[row][basis] for row in range(2)]
                    preparation = ['X 0'] if basis else []
                    self.check_qcx(['QUBITS 1', *preparation,
                                    f'{gate.upper()} 0 ' + ' '.join(map(str, angles))], expected)
                    self.check_source('qubit q; ' + ('x q; ' if basis else '')
                                      + f'{gate}({arguments}) q;', expected)

    def test_builtin_u_reproducer_retains_absolute_phase(self):
        self.check_source('qubit q; U(0, 0, pi) q;', [1, 0])

    def test_builtin_u_all_matrix_columns_with_constant_and_runtime_parameters(self):
        angles = (0.3, 0.4, -0.7)
        matrix = builtin_u_matrix(*angles)
        for runtime in (False, True):
            declarations = 'float theta = 0.3; float phi = 0.4; float lam = -0.7; ' if runtime else ''
            arguments = 'theta, phi, lam' if runtime else '0.3, 0.4, -0.7'
            for basis in (0, 1):
                with self.subTest(runtime=runtime, basis=basis):
                    expected = [matrix[row][basis] for row in range(2)]
                    self.check_source('qubit q; ' + declarations + ('x q; ' if basis else '')
                                      + f'U({arguments}) q;', expected)

    def test_added_control_exposes_relative_phase(self):
        scale = 1 / math.sqrt(2)
        # Wire 0 is the control. With target |0>, controlled Z is identity.
        for modifier in ('ctrl', 'negctrl'):
            with self.subTest(modifier=modifier):
                self.check_source(f'qubit[2] q; h q[0]; {modifier} @ U(0, 0, pi) q[0], q[1];',
                                  [scale, scale, 0, 0])

    def test_cu_follows_published_mathematical_active_block(self):
        scale = 1 / math.sqrt(2)
        state = [scale, scale, 0, 0]
        angles = (0.7, 0.4, -0.2, 0.3)
        expected = apply_matrix(state, documented_cu_active_matrix(*angles), 1, ((0, 1),))
        for runtime in (False, True):
            with self.subTest(runtime=runtime):
                declarations = ('float theta = 0.7; float phi = 0.4; float lam = -0.2; '
                                'float gamma = 0.3; ') if runtime else ''
                arguments = 'theta, phi, lam, gamma' if runtime else '0.7, 0.4, -0.2, 0.3'
                self.check_source('qubit[2] q; ' + declarations + 'h q[0]; '
                                  + f'cu({arguments}) q[0], q[1];', expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    args = parser.parse_args()
    if args.bra is not None:
        BackendTests.bra = args.bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
