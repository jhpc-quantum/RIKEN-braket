#!/usr/bin/env python3
"""Check the experimental endpoint-to-binary64 spectral bridge."""

from fractions import Fraction
import unittest
from unittest.mock import patch

import qasm2qcx_exact_single_qubit_prototype as exact
import qasm2qcx_exact_spectral_bridge_prototype as bridge
import qasm2qcx_exact_spectral_numerical as numerical
import test_qasm2qcx_single_qubit_power_reference as reference
from test_qasm2qcx_single_qubit_spectral_prototype import direct_matrix_power


class ExactBridgeNumericalTests(unittest.TestCase):
    assertMatrixClose = reference.SingleQubitPowerReferenceTests.assertMatrixClose

    def test_exact_endpoint_power_matches_for_equivalent_gate_sequences(self):
        for names in (('t',) * 4, ('s', 's'), ('z',)):
            with self.subTest(sequence=names):
                self.assertMatrixClose(numerical.powered_matrix(exact.compose_sequence(names), 0.5),
                                       reference.GATES['s'], tolerance=1e-9)

    def test_full_scalar_phase_is_preserved(self):
        for value, expected in ((-exact.ONE, 1j), (exact.I, complex(2**-0.5, 2**-0.5))):
            matrix = ((value, exact.ZERO), (exact.ZERO, value))
            with self.subTest(value=value):
                self.assertMatrixClose(numerical.powered_matrix(matrix, 0.5),
                                       ((expected, 0), (0, expected)), tolerance=1e-9)

    def test_nonendpoint_compositions_match_independent_matrix_reference(self):
        for names in (('h', 's'), ('s', 'h'), ('h', 't') * 12):
            expected = direct_matrix_power(reference.sequence_matrix(names), 0.5)
            with self.subTest(sequence=names):
                self.assertMatrixClose(numerical.powered_matrix(exact.compose_sequence(names), 0.5),
                                       expected, tolerance=1e-8)

    def test_endpoint_and_other_complex_phase_are_not_combined(self):
        matrix = ((-exact.ONE, exact.ZERO), (exact.ZERO, exact.I))
        expected = reference.diagonal_power(-1 + 0j, 1j, 0.5)
        self.assertMatrixClose(numerical.powered_matrix(matrix, 0.5), expected, tolerance=1e-9)

    def test_rounded_zero_cannot_override_certified_side(self):
        tiny = Fraction(1, 2**1200)
        root = bridge.CertifiedEigenvalue((Fraction(-1), Fraction(-1)), (tiny, tiny), 'above')
        with self.assertRaisesRegex(ValueError, 'lost its certified side'):
            numerical.approximate_eigenvalue(root)

    def test_nearly_degenerate_float_basis_is_rejected_not_merged(self):
        root = bridge.CertifiedEigenvalue((Fraction(1), Fraction(1)), (Fraction(1, 10**9),) * 2, 'above')
        other = bridge.CertifiedEigenvalue((Fraction(1), Fraction(1)), (Fraction(-1, 10**9),) * 2, 'below')
        with patch.object(bridge, 'certify_eigenvalue_branches', return_value=(root, other)):
            with self.assertRaisesRegex(ValueError, 'separation too small'):
                numerical.approximate_powered_spectrum(exact.GATES['z'], 0.5)


if __name__ == '__main__':
    unittest.main()
