#!/usr/bin/env python3

"""Independent analytic references for principal powers of Pauli rotations.

  python3 bra/test/qasm2qcx_rotation_power_numerical.py
  python3 bra/test/qasm2qcx_rotation_power_numerical.py --bra /path/to/bra

The optional backend audit checks existing ordinary/inverse rotations,
paired-phase preparation, and public conversion of rotation-gate powers.
Use a default-double, non-MPI executable for this audit.
The reference-only run needs only Python's standard library.

R_P(theta) has eigenvalues exp(-i*theta/2), exp(+i*theta/2) on the
projectors (I+P)/2, (I-P)/2. Track both eigenangles separately, selecting the
principal argument before each power and assigning +pi to exact negative-real
coefficients with either imaginary zero sign. Never tolerance-snap nearby
coefficients, infer one principal phase by negating the other, or reduce large
angles modulo a machine approximation of 2*pi. Machine +/-2*pi inputs need
not produce exactly -1 eigenvalues; exact endpoints are tested separately.

As with phase-gate powers, validate all modifiers before evaluating the angle
once on a reached classical path. Reject nonfinite angles and intermediate
products even for outer zero powers or inactive quantum controls. Constant
integer factors in floating chains must fit the host float's finite range.
Underflow follows ordinary floating-point rounding. Finite eigenangles need
not have a representable difference: separate projector phases avoid imposing
that unnecessary restriction. No converter lowering helper is used as an oracle.
"""

import argparse
import cmath
import math
import pathlib
import re
import resource
import subprocess
import unittest


PAULI = {
    'x': ((0j, 1 + 0j), (1 + 0j, 0j)),
    'y': ((0j, -1j), (1j, 0j)),
    'z': ((1 + 0j, 0j), (0j, -1 + 0j)),
}
ROTATIONS = {'rx': ('x', ()), 'ry': ('y', ()), 'rz': ('z', ()),
             'crx': ('x', (1,)), 'cry': ('y', (1,)), 'crz': ('z', (1,))}


class RotationPowerDomainError(ValueError):
    pass


def identity(size):
    return [[complex(row == column) for column in range(size)] for row in range(size)]


def multiply(lhs, rhs):
    return [[sum(left * right for left, right in zip(row, column))
             for column in zip(*rhs)] for row in lhs]


def adjoint(matrix):
    return [[value.conjugate() for value in column] for column in zip(*matrix)]


def projectors(axis):
    pauli = PAULI[axis]
    return tuple([[((row == column) + sign * pauli[row][column]) / 2
                   for column in range(2)] for row in range(2)] for sign in (1, -1))


def spectral_matrix(axis, coefficients):
    plus, minus = projectors(axis)
    return [[coefficients[0] * plus[row][column] + coefficients[1] * minus[row][column]
             for column in range(2)] for row in range(2)]


def condition_matrix(matrix, values):
    """Prepend controls, most significant first; inactive blocks stay identity."""
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise RotationPowerDomainError('invalid control value')
    pattern = 0
    for value in values:
        pattern = (pattern << 1) | value
    size = len(matrix)
    result = identity(size << len(values))
    offset = pattern * size
    for row in range(size):
        for column in range(size):
            result[offset + row][offset + column] = matrix[row][column]
    return result


def validated_modifiers(modifiers):
    result = []
    for kind, argument in modifiers:
        if kind == 'pow':
            if type(argument) not in (int, float):
                raise RotationPowerDomainError('exponent must be a constant real number')
            try:
                argument = float(argument)
            except OverflowError as error:
                raise RotationPowerDomainError('exponent must fit a finite host float') from error
            if not math.isfinite(argument):
                raise RotationPowerDomainError('exponent must be finite')
        elif kind == 'inv':
            if argument is not None:
                raise RotationPowerDomainError('inv takes no argument')
        elif kind in ('ctrl', 'negctrl'):
            if type(argument) is not int or argument <= 0:
                raise RotationPowerDomainError('control count must be a positive constant integer')
        else:
            raise RotationPowerDomainError(f'unsupported modifier {kind}')
        result.append((kind, argument))
    return tuple(result)


def phase_factor(angle):
    if not math.isfinite(angle):
        raise RotationPowerDomainError('angle or intermediate product must be finite')
    return cmath.exp(complex(0, angle))


def principal_argument(coefficient):
    if not (math.isfinite(coefficient.real) and math.isfinite(coefficient.imag)) or coefficient == 0:
        raise RotationPowerDomainError('coefficient must be finite and nonzero')
    if coefficient.real < 0 and coefficient.imag == 0:
        return math.pi
    return cmath.phase(coefficient)


def modified_eigenangles(theta, modifiers):
    # Validate theta before halving it, even for zero powers or inactive controls.
    if not math.isfinite(theta):
        raise RotationPowerDomainError('rotation angle must be finite')
    half = theta / 2
    angles = (-half, half)
    for kind, argument in reversed(modifiers):
        if kind == 'inv':
            angles = tuple(-angle for angle in angles)
        elif kind == 'pow':
            angles = tuple(principal_argument(phase_factor(angle)) * argument for angle in angles)
        # Reject overflow now, rather than allowing a later zero power to hide it.
        for angle in angles:
            phase_factor(angle)
    return angles


def reference_matrix(gate, angle_expression, modifiers=()):
    axis, intrinsic_controls = ROTATIONS[gate]
    modifiers = validated_modifiers(modifiers)
    theta = angle_expression()
    angles = modified_eigenangles(theta, modifiers)
    controls = tuple(value for kind, count in modifiers if kind in ('ctrl', 'negctrl')
                     for value in (int(kind == 'ctrl'),) * count) + intrinsic_controls
    return condition_matrix(spectral_matrix(axis, tuple(map(phase_factor, angles))), controls)


def direct_rotation(axis, theta):
    # Independent closed-form matrices: no spectral/projector helper calls.
    half = theta / 2
    cosine, sine = math.cos(half), math.sin(half)
    if axis == 'x':
        return [[cosine, -1j * sine], [-1j * sine, cosine]]
    if axis == 'y':
        return [[cosine, -sine], [sine, cosine]]
    if axis == 'z':
        return [[cmath.exp(-1j * half), 0j], [0j, cmath.exp(1j * half)]]
    raise ValueError(axis)


class ReferenceTests(unittest.TestCase):
    def assert_matrix_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for row, wanted_row in zip(actual, expected):
            self.assertEqual(len(row), len(wanted_row))
            for value, wanted in zip(row, wanted_row):
                self.assertTrue(math.isfinite(value.real) and math.isfinite(value.imag), value)
                self.assertTrue(cmath.isclose(value, wanted, rel_tol=2e-13, abs_tol=2e-13), (value, wanted))

    def test_projectors_are_complete_orthogonal_and_hermitian(self):
        for axis in PAULI:
            plus, minus = projectors(axis)
            for matrix in (plus, minus):
                self.assert_matrix_close(adjoint(matrix), matrix)
                self.assert_matrix_close(multiply(matrix, matrix), matrix)
            self.assert_matrix_close(multiply(plus, minus), [[0j, 0j], [0j, 0j]])
            self.assert_matrix_close(spectral_matrix(axis, (1, 1)), identity(2))
            self.assert_matrix_close(spectral_matrix(axis, (1, -1)), PAULI[axis])

    def test_projectors_match_independently_transcribed_eigenvectors(self):
        scale = 1 / math.sqrt(2)
        eigenvectors = {'x': ((scale, scale), (scale, -scale)),
                        'y': ((scale, 1j * scale), (scale, -1j * scale)),
                        'z': ((1, 0), (0, 1))}
        for axis, vectors in eigenvectors.items():
            for matrix, vector in zip(projectors(axis), vectors):
                expected = [[left * complex(right).conjugate() for right in vector] for left in vector]
                self.assert_matrix_close(matrix, expected)

    def test_unmodified_spectra_match_direct_rotation_matrices(self):
        for axis in PAULI:
            for theta in (0.0, -0.0, 0.7, -0.7, math.pi, 2 * math.pi, 3 * math.pi,
                          1e20, -1e20, 1e100, -1e100, 1e308, -1e308):
                with self.subTest(axis=axis, theta=theta):
                    self.assert_matrix_close(reference_matrix('r' + axis, lambda: theta), direct_rotation(axis, theta))

    def test_small_angle_roots_match_angle_scaling(self):
        for axis in PAULI:
            for factor in (0.5, -0.5, 2.0, -2.0):
                self.assert_matrix_close(reference_matrix('r' + axis, lambda: 0.7, (('pow', factor),)),
                                         direct_rotation(axis, factor * 0.7))

    def test_angle_scaling_can_lose_global_phase(self):
        for axis in PAULI:
            actual = reference_matrix('r' + axis, lambda: 3 * math.pi, (('pow', 0.5),))
            naive = direct_rotation(axis, 1.5 * math.pi)
            self.assert_matrix_close(actual, [[-value for value in row] for row in naive])
            self.assertGreater(abs(actual[0][0] - naive[0][0]), 1)

    def test_exact_negative_eigenvalues_need_two_independent_principal_phases(self):
        for axis in PAULI:
            for left_zero in (0.0, -0.0):
                for right_zero in (0.0, -0.0):
                    phases = tuple(principal_argument(complex(-1, zero)) for zero in (left_zero, right_zero))
                    self.assertEqual(phases, (math.pi, math.pi))
                    actual = spectral_matrix(axis, tuple(cmath.exp(0.5j * phase) for phase in phases))
                    self.assert_matrix_close(actual, [[1j, 0j], [0j, 1j]])
            incorrect = spectral_matrix(axis, (1j, -1j))
            self.assertGreater(max(abs(incorrect[row][column] - (1j if row == column else 0j))
                                   for row in range(2) for column in range(2)), 0.9)

    def test_nearby_coefficients_are_not_snapped_even_when_argument_rounds_to_pi(self):
        for imaginary in (math.ulp(0.0), -math.ulp(0.0), 1e-12, -1e-12):
            phase = principal_argument(complex(-1, imaginary))
            self.assertEqual(math.copysign(1, phase), math.copysign(1, imaginary))
            root = cmath.exp(0.5j * phase)
            self.assertEqual(math.copysign(1, root.imag), math.copysign(1, imaginary))

    def test_adjacent_angles_on_both_sides_of_two_pi(self):
        below, above = math.nextafter(math.tau, 0), math.nextafter(math.tau, math.inf)
        for theta, sign in ((below, -1), (above, 1), (-below, 1), (-above, -1)):
            actual = reference_matrix('rz', lambda: theta, (('pow', 0.5),))
            self.assertEqual(math.copysign(1, actual[0][0].imag), sign)
            self.assertEqual(math.copysign(1, actual[1][1].imag), -sign)

    def test_machine_two_pi_is_not_an_exact_endpoint(self):
        # Binary64 +/-2*pi has two nearby, not exactly negative-real,
        # eigenvalues. Do not use it as a fixture for the exact -I case above.
        for theta in (math.tau, -math.tau):
            actual = reference_matrix('rz', lambda: theta, (('pow', 0.5),))
            self.assertLess((actual[0][0] * actual[1][1].conjugate()).real, -0.99)

    def test_nested_powers_are_not_flattened(self):
        for axis in PAULI:
            actual = reference_matrix('r' + axis, lambda: math.pi / 2, (('pow', 0.5), ('pow', 6)))
            naive = reference_matrix('r' + axis, lambda: math.pi / 2, (('pow', 3.0),))
            self.assert_matrix_close(actual, direct_rotation(axis, -math.pi / 2))
            self.assertGreater(abs(actual[0][0] - naive[0][0]), 1)

    def test_inverse_order_away_from_cut_and_at_exact_negative_eigenvalues(self):
        for axis in PAULI:
            for modifiers in ((('pow', 0.5), ('inv', None)), (('inv', None), ('pow', 0.5))):
                self.assert_matrix_close(reference_matrix('r' + axis, lambda: 0.7, modifiers),
                                         direct_rotation(axis, -0.35))
            coefficient = complex(-1, -0.0)
            inverse_then_root = cmath.exp(0.5j * principal_argument(coefficient.conjugate()))
            root_then_inverse = cmath.exp(0.5j * principal_argument(coefficient)).conjugate()
            self.assertAlmostEqual(inverse_then_root, 1j)
            self.assertAlmostEqual(root_then_inverse, -1j)

    def test_large_angle_principal_roots_square_to_the_original_matrix(self):
        for axis in PAULI:
            for theta in (1e20, -1e20, 1e100, -1e100, 1e308, -1e308):
                root = reference_matrix('r' + axis, lambda: theta, (('pow', 0.5),))
                self.assert_matrix_close(multiply(root, root), direct_rotation(axis, theta))
                self.assert_matrix_close(multiply(adjoint(root), root), identity(2))

    def test_finite_projector_phases_need_not_have_a_finite_difference(self):
        modifiers = validated_modifiers((('pow', 1e308),))
        angles = modified_eigenangles(3.0, modifiers)
        self.assertTrue(all(math.isfinite(angle) for angle in angles))
        self.assertFalse(math.isfinite(angles[1] - angles[0]))
        for axis in PAULI:
            actual = reference_matrix('r' + axis, lambda: 3.0, modifiers)
            self.assert_matrix_close(multiply(adjoint(actual), actual), identity(2))

    def test_zero_signed_zero_and_subnormal_inputs(self):
        for axis in PAULI:
            for theta in (0.0, -0.0, math.ulp(0.0), -math.ulp(0.0)):
                self.assert_matrix_close(reference_matrix('r' + axis, lambda: theta, (('pow', -0.5),)), identity(2))
            for exponent in (0.0, -0.0, math.ulp(0.0)):
                self.assert_matrix_close(reference_matrix('r' + axis, lambda: 0.7, (('pow', exponent),)), identity(2))

    def test_nonfinite_angles_and_intermediate_overflow_cannot_hide(self):
        for theta in (math.inf, -math.inf, math.nan):
            for modifiers in ((('pow', 0.0),), (('ctrl', 1), ('pow', 0.0)),
                              (('negctrl', 1), ('pow', 0.5))):
                with self.subTest(theta=theta, modifiers=modifiers), self.assertRaisesRegex(RotationPowerDomainError, 'finite'):
                    reference_matrix('rx', lambda: theta, modifiers)
        for modifiers in ((('pow', 1e308),), (('pow', 0.0), ('pow', 1e308))):
            with self.assertRaisesRegex(RotationPowerDomainError, 'product'):
                reference_matrix('rx', lambda: 6.0, modifiers)

    def test_modifiers_are_validated_before_once_only_angle_evaluation(self):
        calls = []
        for invalid in ((('pow', True),), (('pow', math.nan),), (('pow', math.inf),),
                        (('pow', 0.0), ('pow', 10 ** 400)), (('pow', 'runtime'),),
                        (('ctrl', 0),), (('negctrl', -1),), (('ctrl', 1.0),), (('inv', 1),)):
            with self.subTest(invalid=invalid), self.assertRaises(RotationPowerDomainError):
                reference_matrix('rx', lambda: calls.append(1) or 0.7, invalid)
            self.assertEqual(calls, [])
        reference_matrix('rx', lambda: calls.append(0.7) or 0.7,
                         (('pow', 0.5), ('inv', None), ('pow', 2)))
        self.assertEqual(calls, [0.7])
        # The reference is only invoked on a reached classical path.
        for reached in (False, True):
            if reached:
                with self.assertRaises(RotationPowerDomainError):
                    reference_matrix('rx', lambda: calls.append('bad') or math.inf, (('pow', 0.0),))
        self.assertEqual(calls, [0.7, 'bad'])

    def test_intrinsic_and_added_control_patterns(self):
        for axis in PAULI:
            root = reference_matrix('r' + axis, lambda: 0.7, (('pow', 0.5),))
            self.assert_matrix_close(reference_matrix('cr' + axis, lambda: 0.7, (('pow', 0.5),)),
                                     condition_matrix(root, (1,)))
            for values in ((0,), (1,), (0, 1), (1, 0)):
                controls = tuple(('ctrl' if value else 'negctrl', 1) for value in values)
                modifiers = (controls[0], ('pow', 0.5), *controls[1:])
                for gate, intrinsic in (('r' + axis, ()), ('cr' + axis, (1,))):
                    self.assert_matrix_close(reference_matrix(gate, lambda: 0.7, modifiers),
                                             condition_matrix(root, values + intrinsic))
            modifiers = (('ctrl', 2), ('pow', 0.5), ('negctrl', 1))
            self.assert_matrix_close(reference_matrix('cr' + axis, lambda: 0.7, modifiers),
                                     condition_matrix(root, (1, 1, 0, 1)))

    def test_controlled_global_phase_is_relative_not_discardable(self):
        root = [[1j, 0j], [0j, 1j]]  # Principal square root of exact -I.
        self.assert_matrix_close(condition_matrix(root, (1,)),
                                 [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1j, 0], [0, 0, 0, 1j]])
        actual = reference_matrix('crx', lambda: 3 * math.pi, (('pow', 0.5),))
        naive = condition_matrix(direct_rotation('x', 1.5 * math.pi), (1,))
        self.assertEqual(actual[0][0], naive[0][0])
        self.assertGreater(abs(actual[2][2] - naive[2][2]), 1)


class LoweringAuditTests(unittest.TestCase):
    bra = None
    checked = 0

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit paired rotation-phase preparation')
        from qasm2qcx_runtime_for_numerical import qasm2qcx
        self.module = qasm2qcx

    def fixture(self, source, modifiers, expression):
        module = self.module
        program = module.openqasm3.parser.parse('OPENQASM 3; qubit q; ' + source)
        converter = module.QASM2QCXConverter(program)
        converter.visit(program)
        prefix = ' @ '.join(kind if kind == 'inv' else f'{kind}({argument!r})'
                            for kind, argument in modifiers)
        gate = module.openqasm3.parser.parse('OPENQASM 3; ' + prefix + f' @ rx({expression}) q;').statements[0]
        info = converter._QASM2QCXConverter__analyze_gate_modifiers(
            gate.modifiers, allow_runtime=True, allow_float=True)
        return converter, info, gate.arguments[0]

    def program(self, theta, modifiers, *, expression='theta', extra_source='', setup=(), skip=False):
        converter, info, expression = self.fixture(f'float theta = {theta!r}; ' + extra_source, modifiers, expression)
        lines = converter._QASM2QCXConverter__qcx_lines
        source = converter._QASM2QCXConverter__capitalize_variable_name('theta')
        lines.extend(line.format(theta=source) for line in setup)
        if skip:
            lines.append('JUMP SKIPPED')
        with converter._QASM2QCXConverter__powered_rotation_angles(expression, info) as angles:
            # Evaluate the prepared angles independently of quantum gate
            # emission, using the same exponential as the final phase gates.
            for angle in angles:
                coefficient = converter._QASM2QCXConverter__add_new_temporary_variable(self.module.ValueType.COMPLEX)
                converter._QASM2QCXConverter__emit_phase_coefficient(angle, coefficient)
                lines.extend([f'PRINTLN :REAL:{coefficient}', f'PRINTLN :IMAG:{coefficient}'])
            lines.append(f'PRINTLN {source}')
        if skip:
            lines.extend(['@SKIPPED', 'PRINTLN :INT:99'])
        return '\n'.join(lines) + '\n'

    def execute(self, program):
        return subprocess.run([str(self.bra)], input=program, text=True, capture_output=True, timeout=30)

    def check_pair(self, program, expected, source=None):
        result = self.execute(program)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + program)
        values = list(map(float, result.stdout.split()))
        self.assertEqual(len(values), 4 if source is None else 5, result.stdout)
        actual = (complex(*values[:2]), complex(*values[2:4]))
        for value, wanted in zip(actual, expected):
            self.assertTrue(cmath.isclose(value, wanted, rel_tol=1e-5, abs_tol=2e-6), (actual, expected))
        if source is not None:
            self.assertTrue(math.isclose(values[-1], source, rel_tol=1e-5, abs_tol=2e-6), result.stdout)
        type(self).checked += 1

    def test_prepared_pairs_match_analytic_reference(self):
        angles = (0.0, -0.0, 0.7, -0.7, math.pi, math.tau, -math.tau,
                  3 * math.pi, math.nextafter(math.tau, 0), math.nextafter(math.tau, math.inf),
                  1e20, -1e20, 1e100, -1e100, 1e308, -1e308)
        chains = ((('pow', 0.5),), (('pow', -0.5),), (('pow', 0.0),),
                  (('pow', 0.5), ('pow', 6)), (('pow', 6), ('pow', 0.5)),
                  (('inv', None), ('pow', 0.5)), (('pow', 0.5), ('inv', None)))
        for theta in angles:
            for modifiers in chains:
                with self.subTest(theta=theta, modifiers=modifiers):
                    expected = tuple(map(phase_factor, modified_eigenangles(theta, validated_modifiers(modifiers))))
                    self.check_pair(self.program(theta, modifiers), expected, theta)

    def test_indexed_integer_and_unsigned_expressions(self):
        modifiers = (('pow', 0.5),)
        for expression, extra, angle in (
                ('values[index]', 'array[float, 2] values = {0.7, 1.2}; int index = 0;', 0.7),
                ('theta + 0.1', '', 0.8), ('n', 'int n = -2;', -2.0),
                ('u', 'uint u = 2;', 2.0), ('2', '', 2.0)):
            with self.subTest(expression=expression):
                program = self.program(0.7, modifiers, expression=expression, extra_source=extra)
                expected = tuple(map(phase_factor, modified_eigenangles(angle, validated_modifiers(modifiers))))
                self.check_pair(program, expected, 0.7)

    def test_large_finite_pairs_do_not_require_a_finite_phase_difference(self):
        modifiers = (('pow', 1e308),)
        angles = modified_eigenangles(3.0, validated_modifiers(modifiers))
        self.assertFalse(math.isfinite(angles[1] - angles[0]))
        self.check_pair(self.program(3.0, modifiers), tuple(map(phase_factor, angles)), 3.0)

    def test_exact_and_nearby_negative_real_coefficients_are_processed_independently(self):
        for imaginary_parts in ((0.0, -0.0), (-0.0, 0.0), (-1e-300, 1e-300), (1e-12, -1e-12)):
            converter, info, _ = self.fixture('', (('pow', 0.5),), '0.0')
            lines = converter._QASM2QCXConverter__qcx_lines
            phases = []
            for imaginary in imaginary_parts:
                angle = converter._QASM2QCXConverter__add_new_temporary_variable(self.module.ValueType.FLOAT)
                coefficient = converter._QASM2QCXConverter__add_new_temporary_variable(self.module.ValueType.COMPLEX)
                unit = ':MINUS_I' if math.copysign(1, imaginary) < 0 else ':I'
                lines.extend([f'LET {coefficient} := {unit}',
                              f'LET {coefficient} *= :COMPLEX:{abs(imaginary)!r}',
                              f'LET {coefficient} -= :COMPLEX:1.0'])
                phases.append((angle, coefficient))
            converter._QASM2QCXConverter__emit_phase_modifiers(
                tuple(phases), info, converter._QASM2QCXConverter__phase_power_factors(info))
            for _, coefficient in phases:
                lines.extend([f'PRINTLN :REAL:{coefficient}', f'PRINTLN :IMAG:{coefficient}'])
            expected = tuple(cmath.exp(0.5j * principal_argument(complex(-1, imaginary))) for imaginary in imaginary_parts)
            self.check_pair('\n'.join(lines) + '\n', expected)

    def test_nonfinite_angles_and_intermediate_products_cannot_hide(self):
        for theta in (1.0, -1.0, 0.0):
            for modifiers in ((('pow', 0.0),), (('ctrl', 1), ('pow', 0.0)),
                              (('negctrl', 1), ('pow', 0.5))):
                program = self.program(theta, modifiers, setup=('LET {theta} /= 0.0',))
                result = self.execute(program)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('nonfinite', result.stderr)
                self.assertEqual(result.stdout, '')
                skipped = self.execute(self.program(theta, modifiers, setup=('LET {theta} /= 0.0',), skip=True))
                self.assertEqual(skipped.returncode, 0, skipped.stderr)
                self.assertEqual(skipped.stdout.strip(), '99')
        for expression in ('1.0 / 0.0', '0.0 / 0.0', '1e308 * 1e308'):
            result = self.execute(self.program(0.7, (('pow', 0.0),), expression=expression))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('nonfinite', result.stderr)
            skipped = self.execute(self.program(0.7, (('pow', 0.0),), expression=expression, skip=True))
            self.assertEqual(skipped.returncode, 0, skipped.stderr)
            self.assertEqual(skipped.stdout.strip(), '99')
        result = self.execute(self.program(6.0, (('pow', 0.0), ('pow', 1e308))))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('nonfinite', result.stderr)
        self.assertEqual(result.stdout, '')


class ConversionAuditTests(unittest.TestCase):
    bra = None
    checked = 0

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit converted rotation powers')
        from qasm2qcx_runtime_for_numerical import qasm2qcx
        self.module = qasm2qcx

    @staticmethod
    def apply(matrix, state):
        return [sum(value * amplitude for value, amplitude in zip(row, state)) for row in matrix]

    def check_source(self, source, expected, *, fused=False):
        program = source if source.startswith('OPENQASM ') else 'OPENQASM 3; include "stdgates.inc"; ' + source
        lines = self.module.convert(program)
        if fused:
            lines.insert(1, 'BEGIN FUSION')
            lines.append('END FUSION')
        result = subprocess.run([str(self.bra)], input='\n'.join(lines + ['DO AMPLITUDES']) + '\n',
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + source)
        actual = {int(index, 2): complex(float(real), float(imaginary))
                  for index, real, imaginary in re.findall(
                      r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
        self.assertEqual(set(actual), set(range(len(expected))), result.stdout)
        for index, wanted in enumerate(expected):
            self.assertTrue(cmath.isclose(actual[index], wanted, rel_tol=1e-5, abs_tol=2e-6),
                            (source, index, actual[index], wanted))
        type(self).checked += 1

    def check_gate(self, gate, theta, modifiers, *, basis=None, runtime=True, fused=False):
        operation = reference_matrix(gate, lambda: theta, modifiers)
        size = len(operation)
        qubits = size.bit_length() - 1
        if basis is None:
            preparation = 'h q; s q[0]; '
            state = [(1j if index & 1 else 1) / math.sqrt(size) for index in range(size)]
        else:
            preparation = ''.join(f'x q[{index}]; ' for index in range(qubits) if basis & (1 << index))
            state = [complex(index == basis) for index in range(size)]
        prefix = ' @ '.join(kind if kind == 'inv' else f'{kind}({argument!r})' for kind, argument in modifiers)
        operands = ', '.join(f'q[{index}]' for index in reversed(range(qubits)))
        argument = 'theta' if runtime else repr(theta)
        self.check_source(f'qubit[{qubits}] q; float theta = {theta!r}; ' + preparation
                          + f'{prefix} @ {gate}({argument}) {operands};',
                          self.apply(operation, state), fused=fused)

    def test_rotation_families_against_analytic_projectors(self):
        angles = (0.0, -0.0, 0.7, -0.7, math.pi, math.tau, -math.tau,
                  3 * math.pi, math.nextafter(math.tau, 0), math.nextafter(math.tau, math.inf),
                  1e20, -1e20, 1e100, -1e100, 1e308, -1e308)
        chains = ((('pow', 0.5),), (('pow', -0.5),), (('pow', 0.0),),
                  (('pow', 0.5), ('pow', 6)), (('pow', 6), ('pow', 0.5)),
                  (('inv', None), ('pow', 0.5)), (('pow', 0.5), ('inv', None)))
        for gate in ROTATIONS:
            size = 4 if gate.startswith('cr') else 2
            for theta in angles:
                bases = (*range(size), None) if theta in (0.7, 3 * math.pi) else (None,)
                for modifiers in chains:
                    for runtime in (False, True):
                        for basis in bases:
                            with self.subTest(gate=gate, theta=theta, modifiers=modifiers, runtime=runtime, basis=basis):
                                self.check_gate(gate, theta, modifiers, runtime=runtime, basis=basis)

    def test_documented_rotation_power_example(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- rotation-power-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        for fused in (False, True):
            self.check_source(examples[0], [-1, 0, 0, 0, 0, 0, 0, 0], fused=fused)

    def test_mixed_controls_matrix_columns_and_fusion(self):
        for gate in ROTATIONS:
            for values in ((0,), (1,), (0, 1), (1, 0)):
                controls = tuple(('ctrl' if value else 'negctrl', 1) for value in values)
                modifiers = (controls[0], ('pow', 0.5), *controls[1:], ('pow', 6))
                size = len(reference_matrix(gate, lambda: math.pi / 2, modifiers))
                for fused in (False, True):
                    for basis in (*range(size), None):
                        with self.subTest(gate=gate, values=values, fused=fused, basis=basis):
                            self.check_gate(gate, math.pi / 2, modifiers, basis=basis, fused=fused)
            modifiers = (('ctrl', 2), ('pow', 0.5), ('negctrl', 1))
            for fused in (False, True):
                self.check_gate(gate, 3 * math.pi, modifiers, fused=fused)

    def test_finite_eigenangles_with_an_overflowing_difference(self):
        for gate in ROTATIONS:
            for fused in (False, True):
                with self.subTest(gate=gate, fused=fused):
                    self.check_gate(gate, 3.0, (('pow', 1e308),), fused=fused)

    def test_noncontiguous_wires_and_spectators(self):
        for axis in PAULI:
            target = reference_matrix('r' + axis, lambda: 3 * math.pi, (('pow', 0.5),))
            initial = [(1j if index & 1 else 1) * (-1j if index & 4 else 1) / 4 for index in range(16)]
            expected = initial.copy()
            for index in range(16):
                if index & 8 and not index & 4 and not index & 2:
                    pair = self.apply(target, (initial[index], initial[index | 2]))
                    expected[index], expected[index | 2] = pair
            for fused in (False, True):
                self.check_source('qubit[4] q; h q; s q[0]; sdg q[2]; '
                    + f'negctrl @ pow(0.5) @ cr{axis}({3 * math.pi!r}) q[2], q[3], q[1];', expected, fused=fused)

    def test_broadcast_arrays_reused_temporaries_and_integer_angles(self):
        angle = 0.7
        operation = reference_matrix('rx', lambda: angle, (('pow', 0.5),))
        local = self.apply(operation, (1 / math.sqrt(2), 1 / math.sqrt(2)))
        self.check_source('qubit[2] q; h q; float theta = 0.7; pow(0.5) @ rx(theta) q;',
                          [local[high] * local[low] for high in range(2) for low in range(2)])
        state = (1 / math.sqrt(2), 1j / math.sqrt(2))
        for gate, theta, power in (('rx', 0.7, 0.5), ('ry', 1.2, -0.5), ('rz', -0.7, 1.0)):
            state = self.apply(reference_matrix(gate, lambda: theta, (('pow', power),)), state)
        for fused in (False, True):
            self.check_source('qubit q; h q; s q; array[float, 2] values = {0.7, 1.2}; int index = 0; '
                'pow(0.5) @ rx(values[index]) q; index = 1; pow(-0.5) @ ry(values[index]) q; '
                'rz(-values[0]) q;', state, fused=fused)
        for name, declaration, theta in (
                ('n', f'int n = {self.module.QASM2QCXConverter.QCX_INT_MIN};', float(self.module.QASM2QCXConverter.QCX_INT_MIN)),
                ('u', 'uint u = uint(-1);', float(self.module.QASM2QCXConverter.QCX_UINT_MAX))):
            for axis in PAULI:
                expected = self.apply(reference_matrix('r' + axis, lambda: theta, (('pow', 0.5),)),
                                      (1 / math.sqrt(2), 1j / math.sqrt(2)))
                self.check_source('qubit q; h q; s q; ' + declaration + f'pow(0.5) @ r{axis}({name}) q;', expected)

    def test_runtime_loops_break_continue_and_skipped_paths(self):
        state = (1 / math.sqrt(2), 1j / math.sqrt(2))
        for theta in (1.0, 2.0, 3.0):
            state = self.apply(reference_matrix('ry', lambda: theta, (('pow', 0.5),)), state)
        self.check_source('qubit q; h q; s q; int first = 1; int last = 3; '
                          'for int i in [first:last] { pow(0.5) @ ry(float(i)) q; }', state)
        state = (1 / math.sqrt(2), 1j / math.sqrt(2))
        for theta in (0.7, 1.2):
            state = self.apply(reference_matrix('rx', lambda: theta, (('pow', 0.5),)), state)
        self.check_source('qubit q; h q; s q; int count = 0; float theta = 0.7; '
            'while (count < 3) { count += 1; if (count == 2) { continue; } '
            'pow(0.5) @ rx(theta) q; theta += 0.5; if (count == 3) { break; } }', state)
        expected = self.apply(reference_matrix('ry', lambda: 0.7, (('pow', 0.5),)), (1, 0))
        for body in ('pow(0.5) @ rx(1.0 / 0.0) q;', 'pow(0.0) @ rz(1e308 * 1e308) q;',
                     'pow(0.0) @ pow(1e308) @ ry(6.0) q;'):
            for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                self.check_source('qubit q; bool flag = false; ' + wrapper.format(body) + 'pow(0.5) @ ry(0.7) q;', expected)

    def test_nonfinite_angles_and_intermediate_overflow_fail_even_when_inactive(self):
        for gate in ROTATIONS:
            operands = 'q[1], q[0]' if gate.startswith('cr') else 'q[0]'
            for declaration in ('float theta = 1.0 / 0.0;', 'float theta = -1.0 / 0.0;',
                                'float theta = 0.0 / 0.0;', 'float theta = 1e308; theta *= 1e308;'):
                for control in ('', 'ctrl @ ', 'negctrl @ '):
                    actual_operands = 'q[2], ' + operands if control else operands
                    setup = 'x q[2]; ' if control.startswith('negctrl') else ''
                    source = ('OPENQASM 3; include "stdgates.inc"; qubit[3] q; ' + declaration + setup
                              + f'{control}pow(0.0) @ {gate}(theta) {actual_operands};')
                    lines = self.module.convert(source)
                    result = subprocess.run([str(self.bra)], input='\n'.join(lines) + '\n',
                                            text=True, capture_output=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0, source)
                    self.assertIn('nonfinite', result.stderr)
                    self.assertEqual(result.stdout, '')
            source = ('OPENQASM 3; include "stdgates.inc"; qubit[2] q; '
                      + f'pow(0.0) @ pow(1e308) @ {gate}(6.0) {operands};')
            result = subprocess.run([str(self.bra)], input='\n'.join(self.module.convert(source)) + '\n',
                                    text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('nonfinite', result.stderr)


class BackendAuditTests(unittest.TestCase):
    bra = None
    checked = 0

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit existing ordinary/inverse rotations')

    def test_existing_rotation_conventions_match_analytic_projectors(self):
        # Import only the converter/backend runner, never its expected matrices.
        from qasm2qcx_inverse_numerical import check_amplitudes
        angles = (0.0, -0.0, 0.7, -0.7, math.pi, math.tau, 3 * math.pi,
                  math.nextafter(math.tau, 0), math.nextafter(math.tau, math.inf),
                  1e20, -1e20, 1e100, -1e100, 1e308, -1e308)
        for gate in ROTATIONS:
            size = 4 if gate.startswith('cr') else 2
            qubits = size.bit_length() - 1
            operands = ', '.join(f'q[{index}]' for index in reversed(range(qubits)))
            for theta in angles:
                for runtime in (False, True):
                    for inverse in (False, True):
                        operation = reference_matrix(gate, lambda: theta, (('inv', None),) if inverse else ())
                        for basis in range(size + 1):
                            if basis == size:
                                preparation = 'h q; s q[0]; '
                                initial = [(1j if index & 1 else 1) / math.sqrt(size) for index in range(size)]
                            else:
                                preparation = ''.join(f'x q[{index}]; ' for index in range(qubits) if basis & (1 << index))
                                initial = [complex(index == basis) for index in range(size)]
                            expected = [sum(value * amplitude for value, amplitude in zip(row, initial)) for row in operation]
                            source = (f'qubit[{qubits}] q; float theta = {theta!r}; ' + preparation
                                      + ('inv @ ' if inverse else '')
                                      + f'{gate}({"theta" if runtime else repr(theta)}) {operands};')
                            with self.subTest(gate=gate, theta=theta, runtime=runtime, inverse=inverse, basis=basis):
                                check_amplitudes(self.bra, source, expected)
                                type(self).checked += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    options = parser.parse_args()
    if options.bra is not None:
        BackendAuditTests.bra = options.bra.resolve()
        LoweringAuditTests.bra = options.bra.resolve()
        ConversionAuditTests.bra = options.bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if BackendAuditTests.checked:
        print(f'Existing ordinary/inverse rotation circuits checked: {BackendAuditTests.checked}')
    if LoweringAuditTests.checked:
        print(f'Prepared rotation eigenphase pairs checked: {LoweringAuditTests.checked}')
    if ConversionAuditTests.checked:
        print(f'Converted rotation-power circuits checked: {ConversionAuditTests.checked}')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
