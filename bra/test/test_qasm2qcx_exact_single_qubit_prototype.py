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


class ExactCompositionTests(unittest.TestCase):
    def test_exact_gate_identities_and_scalar_phase(self):
        identity = exact.GATES['id']
        for names in ((), ('h', 'h'), ('x', 'x'), ('s', 'sdg'), ('t',) * 8,
                      ('sx', 'sx', 'sx', 'sx')):
            with self.subTest(gates=names):
                self.assertEqual(exact.compose_sequence(names), identity)
        self.assertEqual(exact.compose_sequence(('t',) * 4), exact.GATES['z'])
        self.assertEqual(exact.compose_sequence(('sx', 'sx')), exact.GATES['x'])
        self.assertEqual(exact.compose_sequence(('z', 'y', 'x')),
                         ((exact.I, exact.ZERO), (exact.ZERO, exact.I)))

    def test_order_for_noncommuting_gates(self):
        hs = exact.compose_sequence(('h', 's'))
        sh = exact.compose_sequence(('s', 'h'))
        self.assertEqual(hs, exact.multiply_matrices(exact.GATES['s'], exact.GATES['h']))
        self.assertEqual(sh, exact.multiply_matrices(exact.GATES['h'], exact.GATES['s']))
        self.assertNotEqual(hs, sh)

    def test_fixed_sequences_match_independent_direct_matrix_reference(self):
        for length in range(4):
            for names in itertools.product(exact.GATES, repeat=length):
                actual = exact.compose_sequence(names)
                expected = reference.sequence_matrix(names)
                for row, column in itertools.product(range(2), repeat=2):
                    with self.subTest(gates=names, row=row, column=column):
                        self.assertTrue(cmath.isclose(actual[row][column].to_complex(), expected[row][column],
                                                      rel_tol=1e-14, abs_tol=1e-14))

    def test_unknown_gate_is_rejected(self):
        with self.assertRaises(KeyError):
            exact.compose_sequence(('h', 'unknown'))


if __name__ == '__main__':
    unittest.main()
