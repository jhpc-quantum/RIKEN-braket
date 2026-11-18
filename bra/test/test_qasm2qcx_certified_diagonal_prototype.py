#!/usr/bin/env python3
"""Checks for the restricted exact-diagonal bound connection."""

from fractions import Fraction
import unittest

import qasm2qcx_certified_diagonal_prototype as diagonal
import qasm2qcx_exact_single_qubit_prototype as exact
import qasm2qcx_exact_spectral_bridge_prototype as bridge
import qasm2qcx_spectral_target_prototype as target


PI_LOW = Fraction('3.14159265358979323846264338327950288419716939937510')
PI_HIGH = PI_LOW + Fraction(1, 10**50)


class CertifiedDiagonalTests(unittest.TestCase):
    def setUp(self):
        self.request = target.AccuracyRequest(target.BINARY64, Fraction(1, 10**12))

    def test_equivalent_z_sequences_certify_same_serialized_power(self):
        results = [diagonal.certify_diagonal_power(exact.compose_sequence(names), Fraction(1, 2), self.request)
                   for names in (('z',), ('s', 's'), ('t',) * 4)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        result = results[0]
        self.assertEqual(result.branches, ('positive-real', 'negative-endpoint'))
        self.assertTrue(result.assessment.meets_request)
        chosen = Fraction(result.angles[1])
        self.assertLessEqual(max(abs(chosen - PI_LOW / 2), abs(chosen - PI_HIGH / 2)), result.assessment.total)
        self.assertEqual(Fraction(result.angles[0]), 0)

    def test_scalar_phase_and_nonendpoint_branch_are_preserved(self):
        matrix = ((exact.I, exact.ZERO), (exact.ZERO, exact.I))
        result = diagonal.certify_diagonal_power(matrix, Fraction(1, 2), self.request)
        self.assertEqual(result.branches, ('above', 'above'))
        self.assertEqual(result.angles[0], result.angles[1])
        chosen = Fraction(result.angles[0])
        self.assertLessEqual(max(abs(chosen - PI_LOW / 4), abs(chosen - PI_HIGH / 4)), result.assessment.total)
        for value, expected in ((-exact.OMEGA, 'below'), (-exact.OMEGA.conjugate(), 'above')):
            matrix = ((exact.ONE, exact.ZERO), (exact.ZERO, value))
            with self.subTest(value=value):
                result = diagonal.certify_diagonal_power(matrix, Fraction(1, 2), self.request)
                self.assertEqual(result.branches[1], expected)
                self.assertEqual(Fraction(result.angles[1]) < 0, expected == 'below')

    def test_decimal_rounding_is_exact_and_ties_to_even(self):
        for value, expected in ((Fraction(1, 8), '0.12'), (Fraction(27, 200), '0.14'),
                                (Fraction(-1, 8), '-0.12'), (Fraction(-27, 200), '-0.14'),
                                (Fraction(-1, 1000), '0.00')):
            with self.subTest(value=value):
                text, rounded = diagonal._decimal_nearest_even(value, 2)
                self.assertEqual(text, expected)
                self.assertEqual(Fraction(text), rounded)

    def test_negative_exponent_retains_endpoint_decision(self):
        result = diagonal.certify_diagonal_power(exact.GATES['z'], Fraction(-1, 2), self.request)
        self.assertEqual(result.branches, ('positive-real', 'negative-endpoint'))
        chosen = Fraction(result.angles[1])
        self.assertLessEqual(max(abs(chosen + PI_LOW / 2), abs(chosen + PI_HIGH / 2)), result.assessment.total)

    def test_unmet_error_and_unsupported_matrix_are_rejected(self):
        strict = target.AccuracyRequest(target.BINARY64, Fraction(1, 10**30))
        with self.assertRaisesRegex(bridge.CertificationLimit, 'operator bound'):
            diagonal.certify_diagonal_power(exact.GATES['z'], Fraction(1, 2), strict, decimal_digits=6)
        with self.assertRaisesRegex(bridge.CertificationLimit, 'operator bound'):
            diagonal.certify_diagonal_power(exact.GATES['z'], Fraction(10**9), self.request, phase_bits=8)
        with self.assertRaisesRegex(ValueError, 'exactly diagonal'):
            diagonal.certify_diagonal_power(exact.GATES['h'], Fraction(1, 2), self.request)
        with self.assertRaisesRegex(bridge.CertificationLimit, 'exponent exceeds'):
            diagonal.certify_diagonal_power(exact.GATES['z'], Fraction(2**5000), self.request)


if __name__ == '__main__':
    unittest.main()
