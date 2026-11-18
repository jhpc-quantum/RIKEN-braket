#!/usr/bin/env python3
"""Tests for the symbolic exact endpoint spectral prototype."""

import unittest
import cmath
from fractions import Fraction

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


class CertifiedBranchTests(unittest.TestCase):
    def test_integer_surd_sign_is_exact_near_zero(self):
        for a, b, expected in ((0, 0, 0), (3, -2, 1), (-3, 2, -1),
                               (1, -1, -1), (-1, 1, 1), (0, -2, -1)):
            with self.subTest(a=a, b=b):
                self.assertEqual(bridge._sign_surd(a, b), expected)

    def test_scalar_and_both_endpoint_cases_are_identified_exactly(self):
        for matrix, branches in ((exact.GATES['id'], ('positive-real', 'positive-real')),
                                 (exact.GATES['z'], ('negative-endpoint', 'positive-real')),
                                 (exact.GATES['t'], ('positive-real', 'above')),
                                 (exact.GATES['tdg'], ('positive-real', 'below'))):
            with self.subTest(matrix=matrix):
                result = bridge.certify_eigenvalue_branches(matrix)
                self.assertEqual(tuple(root.branch for root in result), branches)
                self.assertTrue(all(root.exact_value is not None for root in result))
        minus_identity = ((-exact.ONE, exact.ZERO), (exact.ZERO, -exact.ONE))
        self.assertEqual(tuple(root.branch for root in bridge.certify_eigenvalue_branches(minus_identity)),
                         ('negative-endpoint', 'negative-endpoint'))

    def test_nonendpoint_roots_enclose_independent_numeric_eigenvalues(self):
        # Independent quadratic reference; skip cases whose roots are so close
        # that binary64 cannot serve as a useful numerical cross-check.
        sequences = (('h', 't'), ('t', 'h', 's'), ('h', 't') * 12,
                     ('h', 't', 'tdg', 's', 'h', 't', 'x'))
        for sequence in sequences:
            matrix = exact.compose_sequence(sequence)
            roots = bridge.certify_eigenvalue_branches(matrix)
            a, b = (value.to_complex() for value in matrix[0])
            c, d = (value.to_complex() for value in matrix[1])
            discriminant = cmath.sqrt((a - d) ** 2 + 4 * b * c)
            numeric = ((a + d + discriminant) / 2, (a + d - discriminant) / 2)
            # Principal square root may be labeled differently if a rounded
            # discriminant lands on its own cut, so compare both assignments.
            def contains(root, value):
                tolerance = Fraction(1, 10**12)
                real, imag = Fraction.from_float(value.real), Fraction.from_float(value.imag)
                return (root.real[0] - tolerance <= real <= root.real[1] + tolerance
                        and root.imaginary[0] - tolerance <= imag <= root.imaginary[1] + tolerance)
            with self.subTest(sequence=sequence):
                self.assertTrue((contains(roots[0], numeric[0]) and contains(roots[1], numeric[1]))
                                or (contains(roots[0], numeric[1]) and contains(roots[1], numeric[0])))
                for root in roots:
                    if root.branch == 'above':
                        self.assertGreater(root.imaginary[0], 0)
                    elif root.branch == 'below':
                        self.assertLess(root.imaginary[1], 0)

    def test_explicit_budgets_and_nonunitary_rejection(self):
        matrix = exact.compose_sequence(('h', 't') * 12)
        with self.assertRaisesRegex(bridge.CertificationLimit, 'coefficient budget'):
            bridge.certify_eigenvalue_branches(matrix, max_input_bits=1)
        with self.assertRaisesRegex(ValueError, 'invalid certification budget'):
            bridge.certify_eigenvalue_branches(matrix, max_bits=7)
        with self.assertRaisesRegex(ValueError, 'exactly unitary'):
            bridge.certify_eigenvalue_branches(((exact.ONE, exact.ONE), (exact.ZERO, exact.ONE)))

    def test_unresolved_enclosure_stops_at_precision_budget(self):
        from unittest.mock import patch
        matrix = exact.compose_sequence(('h', 't'))
        broad = (Fraction(-2), Fraction(2))
        with patch.object(bridge, '_sqrt_complex', return_value=(broad, broad)):
            with self.assertRaisesRegex(bridge.CertificationLimit, 'unresolved'):
                bridge.certify_eigenvalue_branches(matrix, max_bits=64)


if __name__ == '__main__':
    unittest.main()
