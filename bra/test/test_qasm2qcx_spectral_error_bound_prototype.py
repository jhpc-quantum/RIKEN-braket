#!/usr/bin/env python3
"""Tests for conditional spectral operation-error bounds."""

import cmath
from fractions import Fraction as F
import math
import unittest

import qasm2qcx_spectral_error_bound_prototype as bound
import qasm2qcx_spectral_target_prototype as target


def operation(beta, eta, phase0, phase1):
    c, s = math.cos(beta / 2), math.sin(beta / 2)
    e = cmath.exp(1j * eta)
    v = ((c, -s), (e * s, e * c))
    phases = (cmath.exp(1j * phase0), cmath.exp(1j * phase1))
    return tuple(tuple(sum(v[row][k] * phases[k] * v[column][k].conjugate() for k in range(2))
                       for column in range(2)) for row in range(2))


def difference_operator_norm(left, right):
    diff = [[left[r][c] - right[r][c] for c in range(2)] for r in range(2)]
    trace = sum(abs(value) ** 2 for row in diff for value in row)
    determinant = abs(diff[0][0] * diff[1][1] - diff[0][1] * diff[1][0]) ** 2
    return math.sqrt((trace + math.sqrt(max(0.0, trace * trace - 4 * determinant))) / 2)


class ConditionalSpectralBoundTests(unittest.TestCase):
    def assumptions(self, *, exponent=F(2), exponent_error=F(1, 10000),
                    beta_error=F(1, 1000), eta_error=F(1, 2000)):
        phases = (bound.PhaseErrorAssumption('first', 'above', F(1, 3000), F(1, 5000)),
                  bound.PhaseErrorAssumption('second', 'below', F(1, 4000), F(1, 6000)))
        return bound.SpectralErrorAssumptions(phases, ('first', 'second'), exponent,
                                              exponent_error, beta_error, eta_error)

    def test_exact_components_and_request_decision(self):
        assumptions = self.assumptions()
        result = bound.assess_bound(target.AccuracyRequest(target.BINARY64, F(1, 100)), assumptions)
        self.assertEqual(result.basis, F(1, 500))
        self.assertEqual(result.phases[0], 2 * F(1, 3000) + bound.PI_UPPER * F(1, 10000) + F(1, 5000))
        self.assertEqual(result.total, result.basis + max(result.phases))
        self.assertTrue(result.meets_request)
        stricter = bound.assess_bound(target.AccuracyRequest(target.BINARY64, F(1, 100000)), assumptions)
        self.assertFalse(stricter.meets_request)

    def test_large_exponent_magnifies_input_phase_uncertainty(self):
        result = bound.assess_bound(target.AccuracyRequest(target.BINARY64, F(1, 10)),
                                    self.assumptions(exponent=F(10**9)))
        self.assertGreater(result.total, 1)
        self.assertFalse(result.meets_request)

    def test_mismatched_labels_and_invalid_inputs_are_rejected(self):
        assumptions = self.assumptions()
        with self.assertRaisesRegex(ValueError, 'column order'):
            bound.SpectralErrorAssumptions(assumptions.phases, ('second', 'first'), F(1), F(0), F(0), F(0))
        with self.assertRaises(ValueError):
            bound.PhaseErrorAssumption('first', 'unknown', F(0), F(0))
        with self.assertRaises(ValueError):
            bound.PhaseErrorAssumption('first', 'above', F(-1), F(0))
        with self.assertRaises(TypeError):
            bound.PhaseErrorAssumption('first', 'above', 0.1, F(0))

    def test_perturbed_operations_respect_conditional_bound(self):
        assumptions = self.assumptions()
        result = bound.assess_bound(target.AccuracyRequest(target.BINARY64, F(1, 10)), assumptions)
        for theta0, theta1, beta, eta in ((0.2, -1.3, 1.1, -0.7),
                                         (math.pi - 0.1, -math.pi + 0.1, 0.01, 2.0),
                                         (0.0, 0.0, 2.9, -3.0)):
            with self.subTest(theta0=theta0, theta1=theta1):
                used_exponent = 2.0
                ideal_exponent = used_exponent - float(assumptions.exponent_error)
                ideal = operation(beta, eta, ideal_exponent * theta0, ideal_exponent * theta1)
                perturbed = operation(beta + float(assumptions.beta_error),
                                      eta - float(assumptions.eta_error),
                                      used_exponent * (theta0 + float(assumptions.phases[0].phase_error))
                                      + float(assumptions.phases[0].output_error),
                                      used_exponent * (theta1 - float(assumptions.phases[1].phase_error))
                                      - float(assumptions.phases[1].output_error))
                self.assertLessEqual(difference_operator_norm(ideal, perturbed), float(result.total) + 1e-12)


if __name__ == '__main__':
    unittest.main()
