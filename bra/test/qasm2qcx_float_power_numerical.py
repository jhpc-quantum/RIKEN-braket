#!/usr/bin/env python3

"""Independent spectral references for constant floating-point gate powers.

Stage 1 checks the reference semantics without a converter or backend:
  python3 bra/test/qasm2qcx_float_power_numerical.py

Fixed-family eigenphases are represented exactly as rational multiples of pi.
This avoids using a floating-point eigensolver (and its branch-cut tolerance)
as the oracle for eigenvalue -1. Projectors include all global phases. Modifier
tuples are in source order, with the innermost modifier applied first.
"""

import cmath
from fractions import Fraction
import math
import unittest


Matrix = list[list[complex]]
Spectrum = list[tuple[Fraction, Matrix]]


def principal_phase(phase: Fraction) -> Fraction:
    """Return an eigenphase in pi units in (-1, 1], with -1 mapped to +1."""
    result = (phase + 1) % 2 - 1
    return Fraction(1) if result == -1 else result


def projector(vector) -> Matrix:
    return [[left * right.conjugate() for right in vector] for left in vector]


def fixed_spectrum(gate: str) -> Spectrum:
    """Analytic spectral projectors, independent of converter gate mappings."""
    intrinsic_controls = {'cx': ('x', 1), 'CX': ('x', 1), 'cy': ('y', 1),
                          'cz': ('z', 1), 'ch': ('h', 1), 'ccx': ('x', 2),
                          'cswap': ('swap', 1)}
    if gate in intrinsic_controls:
        target, count = intrinsic_controls[gate]
        return controlled_spectrum(fixed_spectrum(target), (1,) * count)
    zero, one = (1 + 0j, 0j), (0j, 1 + 0j)
    scale = 1 / math.sqrt(2)
    plus, minus = (scale + 0j, scale + 0j), (scale + 0j, -scale + 0j)
    if gate in ('id', 'z', 's', 'sdg', 't', 'tdg'):
        phase = {'id': Fraction(0), 'z': Fraction(1), 's': Fraction(1, 2),
                 'sdg': Fraction(-1, 2), 't': Fraction(1, 4), 'tdg': Fraction(-1, 4)}[gate]
        return [(Fraction(0), projector(zero)), (phase, projector(one))]
    if gate in ('x', 'sx'):
        return [(Fraction(0), projector(plus)),
                (Fraction(1, 2) if gate == 'sx' else Fraction(1), projector(minus))]
    if gate == 'y':
        return [(Fraction(0), projector((scale + 0j, 1j * scale))),
                (Fraction(1), projector((scale + 0j, -1j * scale)))]
    if gate == 'h':
        c, s = math.cos(math.pi / 8), math.sin(math.pi / 8)
        return [(Fraction(0), projector((c + 0j, s + 0j))),
                (Fraction(1), projector((-s + 0j, c + 0j)))]
    if gate == 'swap':
        return [(Fraction(0), projector((1 + 0j, 0j, 0j, 0j))),
                (Fraction(0), projector((0j, scale + 0j, scale + 0j, 0j))),
                (Fraction(0), projector((0j, 0j, 0j, 1 + 0j))),
                (Fraction(1), projector((0j, scale + 0j, -scale + 0j, 0j)))]
    raise ValueError(f'No fixed-family spectral reference for {gate}')


def controlled_spectrum(spectrum: Spectrum, values: tuple[int, ...]) -> Spectrum:
    """Prepend controls as the most significant tensor factors."""
    if not values or any(value not in (0, 1) for value in values):
        raise ValueError('Control pattern must be nonempty and binary')
    target_size = len(spectrum[0][1])
    control_size = 1 << len(values)
    pattern = 0
    for value in values:
        pattern = (pattern << 1) | value
    size = target_size * control_size
    offset = pattern * target_size
    inactive = [[complex(row == column and not offset <= row < offset + target_size)
                 for column in range(size)] for row in range(size)]
    result = [(Fraction(0), inactive)]
    for phase, matrix in spectrum:
        active = [[0j] * size for _ in range(size)]
        for row in range(target_size):
            for column in range(target_size):
                active[offset + row][offset + column] = matrix[row][column]
        result.append((phase, active))
    return result


def modified_spectrum(gate: str, modifiers=()) -> Spectrum:
    spectrum = fixed_spectrum(gate)
    for kind, argument in reversed(modifiers):
        if kind == 'inv':
            # Inversion does not select a new branch. The next power does.
            spectrum = [(-phase, matrix) for phase, matrix in spectrum]
        elif kind == 'pow':
            factor = Fraction(argument)  # Preserve the exact supplied float.
            spectrum = [(principal_phase(phase) * factor, matrix) for phase, matrix in spectrum]
        elif kind in ('ctrl', 'negctrl'):
            spectrum = controlled_spectrum(spectrum, (1 if kind == 'ctrl' else 0,) * argument)
        else:
            raise ValueError(f'Unknown reference modifier {kind}')
    return spectrum


def spectral_matrix(spectrum: Spectrum) -> Matrix:
    size = len(spectrum[0][1])
    result = [[0j] * size for _ in range(size)]
    for phase, matrix in spectrum:
        coefficient = cmath.exp(1j * math.pi * float(principal_phase(phase)))
        for row in range(size):
            for column in range(size):
                result[row][column] += coefficient * matrix[row][column]
    return result


def reference_matrix(gate: str, modifiers=()) -> Matrix:
    return spectral_matrix(modified_spectrum(gate, modifiers))


class SpectralReferenceTests(unittest.TestCase):
    gates = ('id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'swap')

    def assert_matrix_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for row, reference in zip(actual, expected):
            self.assertEqual(len(row), len(reference))
            for value, wanted in zip(row, reference):
                self.assertAlmostEqual(value, wanted, places=12)

    def identity(self, size):
        return [[complex(row == column) for column in range(size)] for row in range(size)]

    def test_branch_endpoint_and_both_sides_are_exact(self):
        epsilon = Fraction(1, 2 ** 55)
        self.assertEqual(principal_phase(Fraction(-1)), 1)
        self.assertEqual(principal_phase(Fraction(1)), 1)
        self.assertEqual(principal_phase(1 - epsilon), 1 - epsilon)
        self.assertEqual(principal_phase(1 + epsilon), -1 + epsilon)
        self.assertEqual(principal_phase(-1 - epsilon), 1 - epsilon)
        self.assertEqual(principal_phase(-1 + epsilon), -1 + epsilon)

    def test_projectors_resolve_identity_and_are_orthogonal(self):
        for gate in self.gates:
            with self.subTest(gate=gate):
                spectrum = fixed_spectrum(gate)
                size = len(spectrum[0][1])
                total = [[sum(matrix[row][column] for _, matrix in spectrum)
                          for column in range(size)] for row in range(size)]
                self.assert_matrix_close(total, self.identity(size))
                for left_index, (_, left) in enumerate(spectrum):
                    for right_index, (_, right) in enumerate(spectrum):
                        product = [[sum(left[row][k] * right[k][column] for k in range(size))
                                    for column in range(size)] for row in range(size)]
                        expected = left if left_index == right_index else [[0j] * size for _ in range(size)]
                        self.assert_matrix_close(product, expected)

    def test_named_square_roots_keep_global_phase(self):
        self.assert_matrix_close(reference_matrix('z', (('pow', 0.5),)), [[1, 0], [0, 1j]])
        self.assert_matrix_close(reference_matrix('x', (('pow', 0.5),)),
                                 [[(1 + 1j) / 2, (1 - 1j) / 2],
                                  [(1 - 1j) / 2, (1 + 1j) / 2]])
        self.assert_matrix_close(reference_matrix('s', (('pow', 0.5),)),
                                 [[1, 0], [0, cmath.exp(1j * math.pi / 4)]])

    def test_base_matrices_match_direct_definitions(self):
        scale = 1 / math.sqrt(2)
        expected = {'id': self.identity(2), 'x': [[0, 1], [1, 0]],
                    'y': [[0, -1j], [1j, 0]], 'z': [[1, 0], [0, -1]],
                    'h': [[scale, scale], [scale, -scale]],
                    's': [[1, 0], [0, 1j]], 'sdg': [[1, 0], [0, -1j]],
                    't': [[1, 0], [0, complex(scale, scale)]],
                    'tdg': [[1, 0], [0, complex(scale, -scale)]],
                    'sx': [[(1 + 1j) / 2, (1 - 1j) / 2], [(1 - 1j) / 2, (1 + 1j) / 2]],
                    'swap': [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]}
        for gate in self.gates:
            with self.subTest(gate=gate):
                self.assert_matrix_close(reference_matrix(gate), expected[gate])

    def test_nested_powers_are_not_flattened(self):
        for gate in ('x', 'y', 'z', 'h', 'swap'):
            with self.subTest(gate=gate):
                actual = reference_matrix(gate, (('pow', 0.5), ('pow', 2)))
                self.assert_matrix_close(actual, self.identity(len(actual)))
                self.assertTrue(any(abs(a - b) > 0.1 for row, other in zip(actual, reference_matrix(gate))
                                    for a, b in zip(row, other)))

    def test_integer_outer_power_retains_the_inner_fractional_root(self):
        for gate in self.gates:
            with self.subTest(gate=gate):
                self.assert_matrix_close(reference_matrix(gate, (('pow', 2), ('pow', 0.5))),
                                         reference_matrix(gate))

    def test_inversion_order_at_negative_eigenvalue(self):
        self.assert_matrix_close(reference_matrix('z', (('pow', 0.5), ('inv', None))), [[1, 0], [0, 1j]])
        self.assert_matrix_close(reference_matrix('z', (('inv', None), ('pow', 0.5))), [[1, 0], [0, -1j]])
        # In particular, negating the exponent is not the same as moving inv.
        self.assert_matrix_close(reference_matrix('z', (('pow', -0.5),)), [[1, 0], [0, -1j]])

    def test_nested_powers_wrap_the_intermediate_phase(self):
        actual = modified_spectrum('t', (('pow', 0.5), ('pow', 6)))
        self.assertEqual([phase for phase, _ in actual], [0, Fraction(-1, 4)])
        self.assert_matrix_close(spectral_matrix(actual), [[1, 0], [0, cmath.exp(-1j * math.pi / 4)]])

    def test_zero_and_large_finite_powers(self):
        for gate in self.gates:
            with self.subTest(gate=gate):
                self.assert_matrix_close(reference_matrix(gate, (('pow', -0.0),)),
                                         self.identity(len(reference_matrix(gate))))
        self.assert_matrix_close(reference_matrix('x', (('pow', 1e308),)), self.identity(2))

    def test_positive_negative_and_mixed_controls(self):
        for values in ((1,), (0,), (0, 1), (1, 0)):
            with self.subTest(values=values):
                controlled = controlled_spectrum(modified_spectrum('z', (('pow', 0.5),)), values)
                expected = self.identity(2 << len(values))
                pattern = 0
                for value in values:
                    pattern = (pattern << 1) | value
                expected[2 * pattern + 1][2 * pattern + 1] = 1j
                self.assert_matrix_close(spectral_matrix(controlled), expected)

    def test_powers_commute_with_added_controls(self):
        for gate in ('x', 'h', 't', 'swap'):
            for control in ('ctrl', 'negctrl'):
                with self.subTest(gate=gate, control=control):
                    self.assert_matrix_close(reference_matrix(gate, ((control, 1), ('pow', 0.5))),
                                             reference_matrix(gate, (('pow', 0.5), (control, 1))))

    def test_control_group_order_is_not_lost(self):
        actual = reference_matrix('z', (('negctrl', 1), ('ctrl', 1), ('pow', 0.5)))
        expected = self.identity(8)
        expected[3][3] = 1j  # |negative=0, positive=1, target=1>
        self.assert_matrix_close(actual, expected)

    def test_intrinsically_controlled_fixed_gates(self):
        for gate, target, count in (('cx', 'x', 1), ('CX', 'x', 1), ('cy', 'y', 1),
                                    ('cz', 'z', 1), ('ch', 'h', 1), ('ccx', 'x', 2),
                                    ('cswap', 'swap', 1)):
            with self.subTest(gate=gate):
                powered = modified_spectrum(target, (('pow', 0.5),))
                self.assert_matrix_close(reference_matrix(gate, (('pow', 0.5),)),
                                         spectral_matrix(controlled_spectrum(powered, (1,) * count)))

    def test_swap_only_phases_the_antisymmetric_subspace(self):
        root = reference_matrix('swap', (('pow', 0.5),))
        self.assert_matrix_close(root, [[1, 0, 0, 0],
                                       [0, (1 + 1j) / 2, (1 - 1j) / 2, 0],
                                       [0, (1 - 1j) / 2, (1 + 1j) / 2, 0],
                                       [0, 0, 0, 1]])


if __name__ == '__main__':
    unittest.main()
