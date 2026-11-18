#!/usr/bin/env python3
"""Tests for the symbolic exact endpoint spectral prototype."""

import unittest

import qasm2qcx_exact_single_qubit_prototype as exact
import qasm2qcx_exact_spectral_bridge_prototype as bridge


def scale_matrix(matrix, value):
    return tuple(tuple(entry * value for entry in row) for row in matrix)


class ExactEndpointSpectrumTests(unittest.TestCase):
    def test_exact_scalar_spectra_keep_the_full_phase(self):
        for value, multiplicity in ((exact.ONE, 0), (-exact.ONE, 2), (exact.I, 0), (-exact.I, 0)):
            matrix = ((value, exact.ZERO), (exact.ZERO, value))
            with self.subTest(value=value):
                spectrum = bridge.endpoint_spectrum(matrix)
                self.assertEqual(spectrum, bridge.ScalarSpectrum(value))
                self.assertEqual(spectrum.negative_one_multiplicity, multiplicity)

    def test_exact_gate_sequences_share_the_same_endpoint(self):
        z = bridge.endpoint_spectrum(exact.GATES['z'])
        for names in (('t',) * 4, ('s', 's')):
            with self.subTest(gates=names):
                self.assertEqual(bridge.endpoint_spectrum(exact.compose_sequence(names)), z)
        self.assertEqual(z.other_eigenvalue, exact.ONE)
        self.assertEqual(z.negative_one_multiplicity, 1)

    def test_endpoint_projectors_are_exact_without_dividing_in_the_ring(self):
        phases = ((-exact.ONE, exact.ZERO), (exact.ZERO, exact.I))
        for matrix in (exact.GATES['x'], exact.GATES['y'], exact.GATES['z'], exact.GATES['h'], phases):
            with self.subTest(matrix=matrix):
                spectrum = bridge.endpoint_spectrum(matrix)
                self.assertIsInstance(spectrum, bridge.EndpointSpectrum)
                numerator = spectrum.projector_numerator
                denominator = spectrum.projector_denominator
                self.assertNotEqual(denominator, exact.ZERO)
                # G P = -P and P**2 = P, multiplied by the exact
                # denominator so no ring division or rounding is needed.
                self.assertEqual(exact.multiply_matrices(matrix, numerator), scale_matrix(numerator, -exact.ONE))
                self.assertEqual(exact.multiply_matrices(numerator, numerator), scale_matrix(numerator, denominator))
                determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
                self.assertEqual(spectrum.other_eigenvalue, -determinant)

    def test_rejects_nonunitary_and_nonendpoint_inputs(self):
        with self.assertRaisesRegex(ValueError, 'exactly unitary'):
            bridge.endpoint_spectrum(((exact.ONE, exact.ONE), (exact.ZERO, exact.ONE)))
        with self.assertRaisesRegex(ValueError, 'no -1 eigenvalue'):
            bridge.endpoint_spectrum(exact.GATES['t'])
        with self.assertRaisesRegex(ValueError, '2x2'):
            bridge.endpoint_spectrum(((exact.ONE,),))


if __name__ == '__main__':
    unittest.main()
