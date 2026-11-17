#!/usr/bin/env python3
"""Investigate composed one-qubit powers with an independent matrix oracle."""

import cmath
import math
import unittest

from qasm2qcx_single_qubit_composition_prototype import (
    Unitary, as_matrix, compose_sequence, native_principal_argument, spectral_power_matrix)
import test_qasm2qcx_single_qubit_power_reference as reference


def direct_matrix_power(matrix, exponent):
    """Apply a power using a 2x2 characteristic polynomial and projectors.

    This independent reference is restricted to well-separated eigenvalues.
    """
    a, b = matrix[0]
    c, d = matrix[1]
    delta = cmath.sqrt((a - d) ** 2 + 4 * b * c)
    if abs(delta) < 1e-8:
        raise ValueError('reference requires separated eigenvalues')
    eigenvalues = ((a + d + delta) / 2, (a + d - delta) / 2)
    powered = [cmath.exp(1j * exponent * reference.principal_argument(value)) for value in eigenvalues]
    return tuple(tuple(
        (powered[0] * (matrix[row][column] - eigenvalues[1] * (row == column))
         - powered[1] * (matrix[row][column] - eigenvalues[0] * (row == column))) / delta
        for column in range(2)) for row in range(2))


class SingleQubitSpectralPrototypeTests(unittest.TestCase):
    assertMatrixClose = reference.SingleQubitPowerReferenceTests.assertMatrixClose

    def test_noncommuting_sequences_match_independent_spectral_reference(self):
        for names in (('h', 's'), ('s', 'h'), ('x', 't'), ('s', 'h', 't')):
            for exponent in (-1.5, 0.5, 2.0):
                with self.subTest(gates=names, exponent=exponent):
                    actual = spectral_power_matrix(compose_sequence(names), (exponent,))
                    expected = direct_matrix_power(reference.sequence_matrix(names), exponent)
                    self.assertMatrixClose(actual, expected)

    def test_exact_scalar_and_identity_powers_retain_phase(self):
        self.assertMatrixClose(spectral_power_matrix(compose_sequence(('x', 'x')), (0.5,)),
                               reference.IDENTITY)
        self.assertMatrixClose(spectral_power_matrix(compose_sequence(('z', 'y', 'x')), (0.5,)),
                               reference.diagonal_power(1j, 1j, 0.5))

    def test_branch_endpoints_and_nearby_values(self):
        for sign in (-1, 1):
            epsilon = 1e-10
            value = complex(-math.cos(epsilon), sign * math.sin(epsilon))
            scalar = Unitary(value, 1.0, (0.0, 0.0, 0.0))
            with self.subTest(sign=sign):
                self.assertMatrixClose(spectral_power_matrix(scalar, (0.5,)),
                                       reference.diagonal_power(value, value, 0.5))
        negative_identity = Unitary(-1 + 0j, 1.0, (0.0, 0.0, 0.0))
        self.assertMatrixClose(spectral_power_matrix(negative_identity, (0.5,)), ((1j, 0), (0, 1j)))

    def test_native_rounding_at_cut_is_observed_without_snapping(self):
        # T^4 is mathematically Z. Its composed -1 eigenvalue may lie on
        # either side of the cut depending on host floating-point rounding.
        composed = compose_sequence(('t', 't', 't', 't'))
        length = math.hypot(*composed.vector)
        negative = composed.phase * complex(composed.scalar, length)
        native_root = cmath.exp(0.5j * native_principal_argument(negative))
        actual = spectral_power_matrix(composed, (0.5,))
        self.assertMatrixClose(actual, ((1, 0), (0, native_root)))
        self.assertMatrixClose(as_matrix(composed), reference.GATES['z'])

    def test_nested_powers_select_each_intermediate_principal_phase(self):
        # Native EXP of a rounded -pi can fall below the cut. Compare the
        # result with separately reselecting the computed intermediate phase.
        sdg = compose_sequence(('sdg',))
        initial = sdg.phase * complex(sdg.scalar, -math.hypot(*sdg.vector))
        intermediate = cmath.exp(2j * native_principal_argument(initial))
        root = cmath.exp(0.5j * native_principal_argument(intermediate))
        self.assertMatrixClose(spectral_power_matrix(sdg, (2.0, 0.5)), ((1, 0), (0, root)))
        self.assertMatrixClose(spectral_power_matrix(sdg, (0.5, 2.0)), reference.GATES['sdg'])


if __name__ == '__main__':
    unittest.main()
