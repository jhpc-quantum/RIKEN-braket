#!/usr/bin/env python3

"""Regress ordinary/integer cu phase-sum loss and overflow (issue #360).

  python3 bra/test/qasm2qcx_cu_phase_numerical.py
  python3 bra/test/qasm2qcx_cu_phase_numerical.py --bra /path/to/bra

Reference tests use only Python's standard library. Expected matrices come
from the independent direct cu matrix and integer matrix products, not from
converter phase operands or floating-power spectral preparation. Compare
complex amplitudes without aligning away global or controlled relative phase.

Stage 1 marks the four known converter failures with expectedFailure. Remove
those markers when Stage 2 separates the two control-phase corrections. The
native audit already demonstrates the proposed factorization, including
inverse/integer sequences and fusion, without modifying the converter.
"""

import argparse
import math
import pathlib
import resource
import unittest

import qasm2qcx_cu_power_numerical as cref


LOSS = (0.7, 0.4, -0.2, 1e20)
OVERFLOW = (1.5e308, 0.4, -0.2, 1.5e308)


class ReferenceTests(unittest.TestCase):
    assert_matrix_close = cref.uref.ReferenceTests.assert_matrix_close

    def test_large_gamma_discards_theta_correction_and_changes_relative_phase(self):
        theta, phi, lam, gamma = LOSS
        self.assertEqual(gamma + theta / 2, gamma)
        correct = cref.direct_active_matrix(LOSS)
        factor = cref.uref.phase_factor(gamma + theta / 2)
        wrong = [[factor * value for value in row]
                 for row in cref.uref.direct_matrix('u3', (theta, phi, lam))]
        self.assertGreater(cref.uref.distance(correct, wrong), 0.3)
        # Entry magnitudes agree: probabilities alone would miss this bug.
        self.assert_matrix_close([[abs(value) for value in row] for row in correct],
                                 [[abs(value) for value in row] for row in wrong])
        correct, wrong = (cref.uref.condition_matrix(matrix, (1,)) for matrix in (correct, wrong))
        self.assertEqual(correct[0], wrong[0])  # Same inactive control block.
        # An active amplitude ratio is not the inactive block's identity phase.
        self.assertGreater(abs(correct[2][2] / wrong[2][2] - 1), 0.3)

    def test_finite_parameters_can_have_an_overflowing_sum_but_finite_unitary(self):
        for sign in (1, -1):
            parameters = tuple(sign * value for value in OVERFLOW)
            self.assertTrue(all(map(math.isfinite, parameters)))
            self.assertFalse(math.isfinite(parameters[3] + parameters[0] / 2))
            operation = cref.integer_reference(parameters, 1)
            self.assertTrue(all(math.isfinite(value.real) and math.isfinite(value.imag)
                                for row in operation for value in row))
            self.assert_matrix_close(cref.uref.multiply(cref.uref.adjoint(operation), operation),
                                     cref.uref.identity(4))

    def test_separate_phase_factors_match_direct_matrix_without_combined_angles(self):
        for parameters in (LOSS, OVERFLOW, tuple(-value for value in LOSS), tuple(-value for value in OVERFLOW)):
            theta, phi, lam, gamma = parameters
            factor = cref.uref.phase_factor(gamma) * cref.uref.phase_factor(theta / 2)
            factored = [[factor * value for value in row]
                        for row in cref.uref.direct_matrix('u3', (theta, phi, lam))]
            self.assert_matrix_close(factored, cref.direct_active_matrix(parameters))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    check = cref.BackendAuditTests.check
    apply = staticmethod(cref.BackendAuditTests.apply)
    inputs = staticmethod(cref.BackendAuditTests.inputs)
    check_source = cref.uref.ConversionAuditTests.check_source

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit cu phase-sum regressions')
        from qasm2qcx_runtime_for_numerical import qasm2qcx
        self.module = qasm2qcx

    def check_ordinary_conversion(self, parameters, *, runtime):
        declarations = ''.join(f'float p{index} = {value!r}; ' for index, value in enumerate(parameters))
        arguments = ', '.join(f'p{index}' if runtime else repr(value) for index, value in enumerate(parameters))
        _, initial = self.inputs(2, None)
        self.check_source('qubit[2] q; ' + declarations + 'h q; s q[0]; '
                          + f'cu({arguments}) q[1], q[0];',
                          self.apply(cref.integer_reference(parameters, 1), initial))

    @unittest.expectedFailure  # #360: remove when the phase sum is eliminated.
    def test_literal_cu_retains_small_theta_beside_large_gamma(self):
        self.check_ordinary_conversion(LOSS, runtime=False)

    @unittest.expectedFailure  # #360: remove when the phase sum is eliminated.
    def test_runtime_cu_retains_small_theta_beside_large_gamma(self):
        self.check_ordinary_conversion(LOSS, runtime=True)

    @unittest.expectedFailure  # #360: remove when the phase sum is eliminated.
    def test_literal_cu_accepts_finite_parameters_with_overflowing_sum(self):
        self.check_ordinary_conversion(OVERFLOW, runtime=False)

    @unittest.expectedFailure  # #360: remove when the phase sum is eliminated.
    def test_runtime_cu_accepts_finite_parameters_with_overflowing_sum(self):
        self.check_ordinary_conversion(OVERFLOW, runtime=True)

    def test_native_separate_factors_preserve_complete_matrix_and_integer_powers(self):
        for parameters in (LOSS, OVERFLOW, tuple(-value for value in OVERFLOW)):
            theta, phi, lam, gamma = parameters
            forward = [f'U1 1 {gamma!r}', f'U1 1 {theta / 2!r}',
                       f'CU3 1 0 {theta!r} {phi!r} {lam!r}']
            adjoint = [f'CU3+ 1 0 {theta!r} {phi!r} {lam!r}',
                       f'U1+ 1 {theta / 2!r}', f'U1+ 1 {gamma!r}']
            for exponent in (1, -1, 2, -2, 0):
                operation = cref.integer_reference(parameters, exponent)
                for fused in (False, True):
                    for basis in (*range(4), None):
                        preparation, initial = self.inputs(2, basis)
                        sequence = (adjoint if exponent < 0 else forward) * abs(exponent)
                        if fused:
                            sequence = ['BEGIN FUSION', *sequence, 'END FUSION']
                        with self.subTest(parameters=parameters, exponent=exponent, fused=fused, basis=basis):
                            self.check(['QUBITS 2', *preparation, *sequence], self.apply(operation, initial))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    options = parser.parse_args()
    if options.bra is not None:
        BackendTests.bra = options.bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if BackendTests.checked:
        print(f'Separate-factor cu circuits checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
