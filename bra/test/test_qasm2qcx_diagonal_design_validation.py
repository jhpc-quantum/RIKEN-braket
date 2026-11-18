#!/usr/bin/env python3
"""Validate selected proposed #640 contract cases against existing prototypes."""

from fractions import Fraction as F
import unittest

import qasm2qcx_certified_diagonal_prototype as diagonal
import qasm2qcx_certified_phase_prototype as phase
import qasm2qcx_exact_single_qubit_prototype as exact
import qasm2qcx_exact_spectral_bridge_prototype as bridge
import qasm2qcx_spectral_target_prototype as target


PI_LOW = F('3.14159265358979323846264338327950288419716939937510')
PI_HIGH = PI_LOW + F(1, 10**50)


def pi_interval(multiplier):
    values = (multiplier * PI_LOW, multiplier * PI_HIGH)
    return min(values), max(values)


class DiagonalDesignValidationTests(unittest.TestCase):
    def request(self, descriptor=target.BINARY64, goal=F(1, 10**12)):
        return target.AccuracyRequest(descriptor, goal)

    def test_known_exact_phase_families_and_powers_stay_within_bound(self):
        scalar_i = exact.compose_sequence(('z', 'y', 'x'))
        cases = ((exact.GATES['id'], (F(0), F(0))),
                 (exact.GATES['z'], (F(0), F(1))),
                 (exact.compose_sequence(('t',) * 4), (F(0), F(1))),
                 (exact.GATES['t'], (F(0), F(1, 4))),
                 (exact.GATES['tdg'], (F(0), F(-1, 4))),
                 (scalar_i, (F(1, 2), F(1, 2))))
        for matrix, coefficients in cases:
            for exponent in (F(0), F(1, 2), F(-1, 2), F(2), F(10**8)):
                with self.subTest(matrix=matrix, exponent=exponent):
                    result = diagonal.certify_diagonal_power(
                        matrix, exponent, self.request(), phase_bits=120, decimal_digits=40)
                    self.assertTrue(result.assessment.meets_request)
                    for angle, coefficient in zip(result.angles, coefficients):
                        low, high = pi_interval(coefficient * exponent)
                        actual = F(angle)
                        worst = max(abs(actual - low), abs(actual - high))
                        self.assertLessEqual(worst, result.assessment.total)

    def test_exact_real_certificate_does_not_depend_on_target_descriptor(self):
        first = diagonal.certify_diagonal_power(exact.GATES['z'], F(1, 2), self.request(target.BINARY32))
        second = diagonal.certify_diagonal_power(exact.GATES['z'], F(1, 2), self.request(target.BINARY64))
        self.assertEqual(first, second)

    def test_synthetic_near_cut_phase_enclosures_do_not_snap(self):
        # Synthetic rectangles exercise the phase routine; they are not a
        # claim that these points are eligible fixed-gate eigenvalues.
        tiny = F(1, 2**120)
        pi = phase._pi_bounds(180, 512)
        for sign in (-1, 1):
            imaginary = sign * tiny
            root = bridge.CertifiedEigenvalue((F(-1), F(-1)), (imaginary, imaginary),
                                               'above' if sign > 0 else 'below')
            lower, upper = phase._phase_bounds(root, 180, 512, pi)
            with self.subTest(sign=sign):
                self.assertLessEqual(lower, upper)
                if sign > 0:
                    self.assertGreater(lower, 0)
                    self.assertLess(upper, pi[0])
                else:
                    self.assertLess(upper, 0)
                    self.assertGreater(lower, -pi[0])

    def test_prototype_precision_and_serialization_failures_are_explicit(self):
        matrix = exact.GATES['z']
        with self.assertRaisesRegex(bridge.CertificationLimit, 'precision budget'):
            diagonal.certify_diagonal_power(matrix, F(1, 2), self.request(), phase_bits=40, max_precision=8)
        with self.assertRaisesRegex(bridge.CertificationLimit, 'term budget'):
            diagonal.certify_diagonal_power(matrix, F(1, 2), self.request(), max_terms=1)
        with self.assertRaisesRegex(bridge.CertificationLimit, 'operator bound'):
            diagonal.certify_diagonal_power(matrix, F(1, 2), self.request(), decimal_digits=2)
        with self.assertRaisesRegex(ValueError, 'invalid diagonal-certification budget'):
            diagonal.certify_diagonal_power(matrix, F(1, 2), self.request(), decimal_digits=201)


if __name__ == '__main__':
    unittest.main()
