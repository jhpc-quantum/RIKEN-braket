#!/usr/bin/env python3
"""Independent matrix cases for the single-qubit power feasibility study.

This reference uses explicit matrices and multiplication; it does not use the
converter's scalar/vector representation or change the converter's behavior.
"""

import cmath
import math
import unittest


SQRT_HALF = math.sqrt(0.5)
IDENTITY = ((1, 0), (0, 1))
GATES = {
    'id': IDENTITY,
    'x': ((0, 1), (1, 0)),
    'y': ((0, -1j), (1j, 0)),
    'z': ((1, 0), (0, -1)),
    'h': ((SQRT_HALF, SQRT_HALF), (SQRT_HALF, -SQRT_HALF)),
    's': ((1, 0), (0, 1j)),
    'sdg': ((1, 0), (0, -1j)),
    't': ((1, 0), (0, cmath.exp(1j * math.pi / 4))),
    'tdg': ((1, 0), (0, cmath.exp(-1j * math.pi / 4))),
    'sx': (((1 + 1j) / 2, (1 - 1j) / 2),
           ((1 - 1j) / 2, (1 + 1j) / 2)),
}


def multiply(left, right):
    return tuple(tuple(sum(left[row][k] * right[k][column] for k in range(2))
                       for column in range(2)) for row in range(2))


def sequence_matrix(names):
    """Apply names left to right, matching gate-body execution order."""
    result = IDENTITY
    for name in names:
        result = multiply(GATES[name], result)
    return result


def principal_argument(value):
    """Use the converter's (-pi, pi] convention, including signed zeros."""
    if not (math.isfinite(value.real) and math.isfinite(value.imag)) or value == 0:
        raise ValueError('a finite, nonzero eigenvalue is required')
    if value.real < 0 and value.imag == 0:
        return math.pi
    return cmath.phase(value)


def diagonal_power(first, second, exponent):
    """Reference for diagonal cases with independently known eigenvalues."""
    return ((cmath.exp(1j * exponent * principal_argument(first)), 0),
            (0, cmath.exp(1j * exponent * principal_argument(second))))


class SingleQubitPowerReferenceTests(unittest.TestCase):
    def assertMatrixClose(self, actual, expected, *, tolerance=1e-14):
        for row in range(2):
            for column in range(2):
                with self.subTest(row=row, column=column):
                    self.assertTrue(cmath.isclose(actual[row][column], expected[row][column],
                                                  rel_tol=tolerance, abs_tol=tolerance),
                                    (actual, expected))

    def test_noncommuting_gates_use_execution_order(self):
        self.assertMatrixClose(sequence_matrix(('h', 's')),
                               ((SQRT_HALF, SQRT_HALF), (1j * SQRT_HALF, -1j * SQRT_HALF)))
        self.assertMatrixClose(sequence_matrix(('s', 'h')),
                               ((SQRT_HALF, 1j * SQRT_HALF), (SQRT_HALF, -1j * SQRT_HALF)))
        self.assertNotEqual(sequence_matrix(('h', 's')), sequence_matrix(('s', 'h')))

    def test_identity_and_nontrivial_scalar_phase(self):
        self.assertMatrixClose(sequence_matrix(('x', 'x')), IDENTITY)
        self.assertMatrixClose(sequence_matrix(('h', 'h')), IDENTITY)
        # XYZ = iI, so this case detects accidental removal of global phase.
        self.assertMatrixClose(sequence_matrix(('z', 'y', 'x')), ((1j, 0), (0, 1j)))

    def test_fixed_gate_matrices_are_unitary(self):
        for name, matrix in GATES.items():
            adjoint = tuple(tuple(matrix[column][row].conjugate() for column in range(2))
                            for row in range(2))
            with self.subTest(gate=name):
                self.assertMatrixClose(multiply(adjoint, matrix), IDENTITY)

    def test_negative_real_boundary_and_signed_zero(self):
        for imaginary in (0.0, -0.0):
            with self.subTest(imaginary=imaginary):
                value = complex(-1.0, imaginary)
                self.assertEqual(principal_argument(value), math.pi)
                self.assertMatrixClose(diagonal_power(value, value, 0.5), ((1j, 0), (0, 1j)))

    def test_distinct_sides_of_branch_cut_are_not_merged(self):
        epsilon = 1e-10
        above = complex(-math.cos(epsilon), math.sin(epsilon))
        below = complex(-math.cos(epsilon), -math.sin(epsilon))
        result = diagonal_power(above, below, 0.5)
        expected = ((cmath.exp(0.5j * (math.pi - epsilon)), 0),
                    (0, cmath.exp(-0.5j * (math.pi - epsilon))))
        self.assertMatrixClose(result, expected)
        self.assertGreater(abs(result[0][0] - result[1][1]), 1.99)

    def test_nearly_equal_eigenvalues_away_from_cut(self):
        epsilon = 1e-10
        result = diagonal_power(cmath.exp(1j * epsilon), cmath.exp(-1j * epsilon), 0.5)
        self.assertMatrixClose(result, ((cmath.exp(0.5j * epsilon), 0),
                                        (0, cmath.exp(-0.5j * epsilon))))
        self.assertNotEqual(result[0][0], result[1][1])


if __name__ == '__main__':
    unittest.main()
