#!/usr/bin/env python3

"""Independent spectral references for constant floating-point gate powers.

Check the reference semantics without a converter or backend:
  python3 bra/test/qasm2qcx_float_power_numerical.py
Also execute converted circuits against those references:
  python3 bra/test/qasm2qcx_float_power_numerical.py --bra /path/to/bra

Fixed-family eigenphases are represented exactly as rational multiples of pi.
This avoids using a floating-point eigensolver (and its branch-cut tolerance)
as the oracle for eigenvalue -1. Projectors include all global phases. Modifier
tuples are in source order, with the innermost modifier applied first.
"""

import cmath
import argparse
from fractions import Fraction
import math
import pathlib
import re
import resource
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


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to execute converted circuits')

    def check_source(self, source, expected):
        # Reuse only the backend runner, never its expected-matrix helpers.
        from qasm2qcx_inverse_numerical import check_amplitudes
        check_amplitudes(self.bra, source, expected)
        type(self).checked += 1

    def check_gate(self, gate, modifiers, basis=None):
        matrix = reference_matrix(gate, modifiers)
        size = len(matrix)
        qubits = size.bit_length() - 1
        if basis is None:
            state = [cmath.exp(1j * sum(0.17 * (wire + 1) for wire in range(qubits)
                                        if index & (1 << wire))) / math.sqrt(size) for index in range(size)]
            preparation = 'h q; ' + ' '.join(f'p({0.17 * (wire + 1)}) q[{wire}];' for wire in range(qubits))
        else:
            state = [complex(index == basis) for index in range(size)]
            preparation = ' '.join(f'x q[{wire}];' for wire in range(qubits) if basis & (1 << wire))
        expected = [sum(matrix[row][column] * state[column] for column in range(size)) for row in range(size)]
        prefix = ''.join('inv @ ' if kind == 'inv' else f'{kind}({argument}) @ ' for kind, argument in modifiers)
        operands = ', '.join(f'q[{wire}]' for wire in reversed(range(qubits)))
        self.check_source(f'qubit[{qubits}] q; {preparation} {prefix}{gate} {operands};', expected)

    def test_fixed_families_and_intrinsic_controls(self):
        gates = SpectralReferenceTests.gates + ('cx', 'CX', 'cy', 'cz', 'ch', 'ccx', 'cswap')
        for gate in gates:
            size = len(reference_matrix(gate))
            for exponent in (-1.5, -0.5, -0.0, 0.5, 1.5, 2.0):
                for basis in (0, size - 1, None):
                    with self.subTest(gate=gate, exponent=exponent, basis=basis):
                        self.check_gate(gate, (('pow', exponent),), basis)

    def test_nested_powers_and_inverse_order(self):
        profiles = ((('pow', 0.5), ('pow', 2)), (('pow', 2), ('pow', 0.5)),
                    (('pow', 0.5), ('pow', 6)), (('pow', 0.5), ('inv', None)),
                    (('inv', None), ('pow', 0.5)), (('inv', None), ('pow', 0.5), ('inv', None)),
                    (('pow', -0.5), ('pow', 3)), (('pow', 1.5), ('pow', 0.5), ('pow', -2)),
                    (('pow', 0.5), ('pow', math.nextafter(1.0, math.inf))),
                    (('pow', 0.5), ('pow', math.nextafter(1.0, 0.0))))
        for gate in SpectralReferenceTests.gates:
            for modifiers in profiles:
                with self.subTest(gate=gate, modifiers=modifiers):
                    self.check_gate(gate, modifiers)

    def test_added_controls_and_mixed_polarities(self):
        profiles = ((('ctrl', 1), ('pow', 0.5)), (('negctrl', 1), ('pow', 0.5)),
                    (('negctrl', 1), ('ctrl', 1), ('pow', -0.5)),
                    (('pow', 0.5), ('negctrl', 1), ('inv', None), ('ctrl', 1)))
        for gate in SpectralReferenceTests.gates + ('cx', 'cy', 'ch', 'ccx', 'cswap'):
            for modifiers in profiles:
                with self.subTest(gate=gate, modifiers=modifiers):
                    self.check_gate(gate, modifiers)

    def test_large_and_small_finite_exponents(self):
        for gate in ('x', 't', 'sx', 'swap'):
            for exponent in (1e308, -1e308, 1e-308, 5e-324, 0.1):
                with self.subTest(gate=gate, exponent=exponent):
                    self.check_gate(gate, (('pow', exponent),))

    def test_constant_expressions_and_casts(self):
        for expression, exponent in (('F + 0.25', 0.75), ('1.0 / N', 0.5), ('float(N)', 2.0)):
            with self.subTest(expression=expression):
                matrix = reference_matrix('x', (('pow', exponent),))
                self.check_source('qubit q; const float F = 0.5; const int N = 2; '
                                  + f'pow({expression}) @ x q;', [row[0] for row in matrix])

    def test_documented_floating_power_example(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- floating-power-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        self.check_source(examples[0].split('\n', 1)[1], [complex(index == 2) for index in range(8)])

    def test_broadcasting(self):
        # Each operation acts on a different pair, including intrinsic controls.
        for gate, operands in (('x', 'q[0:1]'), ('swap', 'q[0:1], q[2:3]'),
                               ('cx', 'q[0:1], q[2:3]')):
            with self.subTest(gate=gate):
                matrix = reference_matrix(gate, (('pow', 0.5),))
                state = [complex(index == 0) for index in range(16)]
                preparation = ''
                if gate in ('swap', 'cx'):
                    preparation = 'x q[0]; x q[1]; '
                    state = [complex(index == 3) for index in range(16)]
                for wires in ((0,), (1,)) if gate == 'x' else ((0, 2), (1, 3)):
                    updated = [0j] * 16
                    for source, amplitude in enumerate(state):
                        column = sum(((source >> wire) & 1) << (len(wires) - 1 - index)
                                     for index, wire in enumerate(wires))
                        for row in range(len(matrix)):
                            destination = source
                            for index, wire in enumerate(wires):
                                destination = ((destination & ~(1 << wire))
                                               | (((row >> (len(wires) - 1 - index)) & 1) << wire))
                            updated[destination] += matrix[row][column] * amplitude
                    state = updated
                self.check_source('qubit[4] q; ' + preparation + f'pow(0.5) @ {gate} {operands};', state)

    def test_classical_control_flow(self):
        for header, declarations, update in (('for int i in [0:1]', '', ''),
                ('for int i in [0:stop]', 'int stop = 1;', ''),
                ('while (i < 2)', 'int i = 0;', 'i += 1;')):
            with self.subTest(header=header):
                self.check_source('qubit q; ' + declarations + header + ' { pow(0.5) @ x q; '
                                  + update + ' }', [0, 1])
        self.check_source('qubit q; if (false) { pow(0.5) @ x q; }', [1, 0])
        matrix = reference_matrix('x', (('pow', 0.5),))
        a, b = matrix[0][0], matrix[1][0]
        expected = [a * a, b * a, a * b, b * b]
        self.check_source('qubit[2] q; for int i in [0:1] { pow(0.5) @ x q[i]; }', expected)
        self.check_source('qubit[2] q; for int i in [0:0] { for int j in [0:1] '
                          '{ pow(0.5) @ x q[i + j]; } }', expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    args = parser.parse_args()
    if args.bra is not None:
        BackendTests.bra = args.bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.bra is not None:
        print(f'{BackendTests.checked} floating-power amplitude comparisons passed')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
