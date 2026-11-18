#!/usr/bin/env python3
"""Checks for exact-rational principal-phase enclosures."""

import cmath
from fractions import Fraction
import math
import unittest

import qasm2qcx_certified_phase_prototype as phases
import qasm2qcx_exact_single_qubit_prototype as exact
import qasm2qcx_exact_spectral_bridge_prototype as bridge


# A published decimal prefix of pi, used as an independent narrow interval;
# the implementation obtains pi from a rational Machin series instead.
PI_LOW = Fraction('3.14159265358979323846264338327950288419716939937510')
PI_HIGH = PI_LOW + Fraction(1, 10**50)


class CertifiedPhaseTests(unittest.TestCase):
    def assertOverlaps(self, result, lower, upper):
        self.assertLessEqual(result.lower, upper)
        self.assertGreaterEqual(result.upper, lower)

    def test_pi_bounds_include_independent_decimal_prefix(self):
        lower, upper = phases._pi_bounds(120, 512)
        self.assertLessEqual(lower, PI_HIGH)
        self.assertGreaterEqual(upper, PI_LOW)
        self.assertLess(upper - lower, Fraction(1, 2**120))

    def test_exact_endpoint_and_scalar_phases(self):
        examples = ((exact.GATES['z'], ((PI_LOW, PI_HIGH), (Fraction(0), Fraction(0)))),
                    (exact.GATES['t'], ((Fraction(0), Fraction(0)), (PI_LOW / 4, PI_HIGH / 4))),
                    (exact.GATES['tdg'], ((Fraction(0), Fraction(0)), (-PI_HIGH / 4, -PI_LOW / 4))))
        for matrix, expected in examples:
            with self.subTest(matrix=matrix):
                actual = phases.certify_principal_phases(matrix, target_bits=40)
                for phase, (low, high) in zip(actual, expected):
                    self.assertOverlaps(phase, low, high)
                    self.assertLessEqual(phase.upper - phase.lower, Fraction(1, 2**40))
        for value, expected in ((-exact.ONE, (PI_LOW, PI_HIGH)),
                                (exact.I, (PI_LOW / 2, PI_HIGH / 2)),
                                (-exact.I, (-PI_HIGH / 2, -PI_LOW / 2))):
            matrix = ((value, exact.ZERO), (exact.ZERO, value))
            with self.subTest(value=value):
                for phase in phases.certify_principal_phases(matrix, target_bits=40):
                    self.assertOverlaps(phase, *expected)

    def test_equivalent_exact_endpoint_sequences_match(self):
        expected = phases.certify_principal_phases(exact.GATES['z'])
        self.assertEqual(phases.certify_principal_phases(exact.compose_sequence(('t',) * 4)), expected)
        self.assertEqual(phases.certify_principal_phases(exact.compose_sequence(('s', 's'))), expected)

    def test_general_nonendpoint_phases_match_numeric_crosscheck(self):
        for sequence in (('h', 't'), ('h', 't') * 12, ('s', 'h', 't')):
            matrix = exact.compose_sequence(sequence)
            actual = phases.certify_principal_phases(matrix, target_bits=32)
            a, b = (entry.to_complex() for entry in matrix[0])
            c, d = (entry.to_complex() for entry in matrix[1])
            root = cmath.sqrt((a - d) ** 2 + 4 * b * c)
            reference = (cmath.phase((a + d + root) / 2), cmath.phase((a + d - root) / 2))
            # Quadratic roots may be labeled differently when the rounded
            # discriminant lies on its own branch cut.
            def matches(phase, value):
                return float(phase.lower) - 1e-12 <= value <= float(phase.upper) + 1e-12
            with self.subTest(sequence=sequence):
                self.assertTrue((matches(actual[0], reference[0]) and matches(actual[1], reference[1]))
                                or (matches(actual[0], reference[1]) and matches(actual[1], reference[0])))
                self.assertEqual([phase.index for phase in actual], [0, 1])
                self.assertTrue(all(phase.upper - phase.lower <= Fraction(1, 2**32) for phase in actual))

    def test_atan_reduction_encloses_positive_and_negative_values(self):
        pi = phases._pi_bounds(100, 512)
        for value in (Fraction(-100), Fraction(-1), Fraction(-1, 3), Fraction(0),
                      Fraction(1, 3), Fraction(1), Fraction(100)):
            lo, hi = phases._atan(value, 80, 512, pi)
            with self.subTest(value=value):
                numeric = math.atan(float(value))
                self.assertLessEqual(float(lo) - 1e-15, numeric)
                self.assertGreaterEqual(float(hi) + 1e-15, numeric)

    def test_precision_and_term_budgets_are_enforced(self):
        matrix = exact.GATES['t']
        with self.assertRaisesRegex(bridge.CertificationLimit, 'term budget'):
            phases.certify_principal_phases(matrix, target_bits=40, max_terms=1)
        with self.assertRaisesRegex(bridge.CertificationLimit, 'precision budget'):
            phases.certify_principal_phases(matrix, target_bits=40, max_precision=8)
        with self.assertRaises(ValueError):
            phases.certify_principal_phases(matrix, target_bits=129)
        with self.assertRaisesRegex(bridge.CertificationLimit, 'coefficient budget'):
            phases.certify_principal_phases(exact.compose_sequence(('h', 't') * 12), max_input_bits=1)


if __name__ == '__main__':
    unittest.main()
