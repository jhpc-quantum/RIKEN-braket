#!/usr/bin/env python3
"""Independent checks for rational projector enclosures."""

import cmath
from fractions import Fraction
import unittest
from unittest.mock import patch

import qasm2qcx_certified_phase_prototype as phases
import qasm2qcx_certified_projector_prototype as projectors
import qasm2qcx_exact_single_qubit_prototype as exact
import qasm2qcx_exact_spectral_bridge_prototype as bridge


class CertifiedProjectorTests(unittest.TestCase):
    def assertContains(self, enclosure, value, tolerance=1e-12):
        real, imaginary = enclosure
        self.assertLessEqual(float(real[0]) - tolerance, value.real)
        self.assertGreaterEqual(float(real[1]) + tolerance, value.real)
        self.assertLessEqual(float(imaginary[0]) - tolerance, value.imag)
        self.assertGreaterEqual(float(imaginary[1]) + tolerance, value.imag)

    def test_scalar_path_avoids_arbitrary_basis(self):
        for value in (exact.ONE, -exact.ONE, exact.I):
            matrix = ((value, exact.ZERO), (exact.ZERO, value))
            with self.subTest(value=value):
                result = projectors.certify_projectors(matrix)
                self.assertIsInstance(result, projectors.ScalarProjectors)
                self.assertEqual(result.eigenvalue.exact_value, value)

    def test_exact_endpoint_projector_and_phase_pairing(self):
        for names in (('z',), ('t',) * 4, ('s', 's')):
            matrix = exact.compose_sequence(names)
            result = projectors.certify_projectors(matrix, target_bits=36)
            eigenphases = phases.certify_principal_phases(matrix, target_bits=36)
            with self.subTest(sequence=names):
                self.assertEqual([p.index for p in result], [p.index for p in eigenphases])
                self.assertEqual([p.eigenvalue.branch for p in result], [p.branch for p in eigenphases])
                self.assertEqual(result[0].eigenvalue.branch, 'negative-endpoint')
                for row in range(2):
                    for column in range(2):
                        self.assertContains(result[0].entries[row][column], complex(row == column == 1))
                        self.assertContains(result[1].entries[row][column], complex(row == column == 0))

    def test_noncommuting_projectors_match_independent_quadratic_reference(self):
        for sequence in (('h', 't'), ('s', 'h', 't'), ('h', 't') * 12):
            matrix = exact.compose_sequence(sequence)
            result = projectors.certify_projectors(matrix, target_bits=32)
            eigenphases = phases.certify_principal_phases(matrix, target_bits=32)
            values = [[entry.to_complex() for entry in row] for row in matrix]
            a, b = values[0]
            c, d = values[1]
            delta = cmath.sqrt((a - d) ** 2 + 4 * b * c)
            roots = ((a + d + delta) / 2, (a + d - delta) / 2)
            with self.subTest(sequence=sequence):
                self.assertEqual([p.eigenvalue.branch for p in result], [p.branch for p in eigenphases])
                for item in result:
                    midpoint = complex(float(sum(item.eigenvalue.real) / 2),
                                       float(sum(item.eigenvalue.imaginary) / 2))
                    index = min(range(2), key=lambda i: abs(roots[i] - midpoint))
                    own, other = roots[index], roots[1 - index]
                    for row in range(2):
                        for column in range(2):
                            expected = (values[row][column] - other * (row == column)) / (own - other)
                            self.assertContains(item.entries[row][column], expected)
                            for component in item.entries[row][column]:
                                self.assertLessEqual(component[1] - component[0], Fraction(1, 2**32))

    def test_zero_gap_and_budget_failures_do_not_guess(self):
        zero = (Fraction(-1), Fraction(1))
        with self.assertRaisesRegex(bridge.CertificationLimit, 'gap'):
            projectors._cdivide((zero, zero), (zero, zero))
        with self.assertRaisesRegex(bridge.CertificationLimit, 'precision budget'):
            projectors.certify_projectors(exact.GATES['h'], target_bits=40, max_precision=8)
        matrix = exact.compose_sequence(('h', 't') * 12)
        with self.assertRaisesRegex(bridge.CertificationLimit, 'coefficient budget'):
            projectors.certify_projectors(matrix, max_input_bits=1)
        with patch.object(projectors.bridge, 'certify_eigenvalue_branches',
                          side_effect=bridge.CertificationLimit('eigenvalue branch unresolved')):
            with self.assertRaisesRegex(bridge.CertificationLimit, 'precision budget'):
                projectors.certify_projectors(matrix, max_precision=64)


if __name__ == '__main__':
    unittest.main()
