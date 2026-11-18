#!/usr/bin/env python3
"""Tests for the exact arithmetic and fixed-gate prototype."""

import cmath
import itertools
import unittest

import qasm2qcx_exact_single_qubit_prototype as exact
import test_qasm2qcx_single_qubit_power_reference as reference


def matrix_product(left, right):
    return tuple(tuple(sum((left[row][k] * right[k][column] for k in range(2)), exact.ZERO)
                       for column in range(2)) for row in range(2))


class ExactArithmeticTests(unittest.TestCase):
    def test_normalization_and_invalid_inputs(self):
        self.assertEqual(exact.Exact(4, -2, 6, -8, 2), exact.Exact(2, -1, 3, -4, 1))
        self.assertEqual(exact.Exact(0, 0, 0, 0, 100), exact.ZERO)
        with self.assertRaises(ValueError):
            exact.Exact(k=-1)
        for value in (1.0, True):
            with self.subTest(value=value), self.assertRaises(TypeError):
                exact.Exact(a=value)

    def test_exact_ring_relations(self):
        self.assertEqual(exact.SQRT_HALF * exact.SQRT_HALF, exact.Exact(a=1, k=1))
        self.assertEqual(exact.I * exact.I, -exact.ONE)
        self.assertEqual(exact.OMEGA * exact.OMEGA, exact.I)
        self.assertEqual(exact.OMEGA * exact.OMEGA.conjugate(), exact.ONE)
        self.assertEqual(exact.HALF_PLUS_HALF_I + exact.HALF_MINUS_HALF_I, exact.ONE)
        self.assertEqual(exact.Exact(a=-3, k=3) - exact.Exact(a=1, k=1), exact.Exact(a=-7, k=3))

    def test_arithmetic_and_conjugation_match_complex_values(self):
        values = (exact.ZERO, exact.ONE, exact.I, exact.SQRT_HALF, exact.OMEGA,
                  exact.Exact(-3, 2, 5, -7, 3))
        for left, right in itertools.product(values, repeat=2):
            with self.subTest(left=left, right=right):
                self.assertTrue(cmath.isclose((left + right).to_complex(),
                                              left.to_complex() + right.to_complex(), abs_tol=1e-14))
                self.assertTrue(cmath.isclose((left * right).to_complex(),
                                              left.to_complex() * right.to_complex(), abs_tol=1e-14))
                self.assertEqual((left * right).conjugate(), left.conjugate() * right.conjugate())

    def test_exact_gate_matrices_match_reference_and_are_unitary(self):
        self.assertEqual(set(exact.GATES), set(reference.GATES))
        for name, matrix in exact.GATES.items():
            with self.subTest(gate=name):
                for row, column in itertools.product(range(2), repeat=2):
                    self.assertTrue(cmath.isclose(matrix[row][column].to_complex(),
                                                  reference.GATES[name][row][column], abs_tol=1e-14))
                adjoint = tuple(tuple(matrix[column][row].conjugate() for column in range(2))
                                for row in range(2))
                self.assertEqual(matrix_product(adjoint, matrix), exact.GATES['id'])


if __name__ == '__main__':
    unittest.main()
