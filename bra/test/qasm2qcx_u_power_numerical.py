#!/usr/bin/env python3

"""Independent spectral references for floating powers of u2, u3, and U.

  python3 bra/test/qasm2qcx_u_power_numerical.py
  python3 bra/test/qasm2qcx_u_power_numerical.py --bra /path/to/bra

The reference-only run needs Python's standard library, not the parser or bra.
Optional default-double, non-MPI audits exercise existing ordinary/inverse
U-family conversion, QCX eigenbasis/projector-phase gates, and private runtime
spectral preparation. Public floating-power conversion is not enabled yet.

Conventions match qasm2qcx.md: u3 is conventional W, u2 fixes theta=pi/2,
and built-in U=exp(i*theta/2)*W. Derive the spectrum from the SU(2) quaternion,
but cross-check against independently transcribed matrices and, away from
degeneracy, a separate quadratic characteristic-polynomial calculation.
No converter spectral helper is used as an oracle.

Select both FULL eigenvalues' principal arguments independently before each
power, assigning +pi to exact negative-real eigenvalues with either imaginary
zero sign. Never power the global factor separately, flatten nested powers,
epsilon-merge eigenvalues, or tolerance-snap the branch cut. A near-scalar gate
straddling the cut can have a root far from the exact scalar gate's root; that
is inherent sensitivity, not permission to discard the small axis vector.

Validate factors before evaluating parameters once, left to right. Validate
original parameters before halving, even at zero power/inactive quantum
controls. Intermediate modified eigenphases must be finite, even behind an
outer zero. Products of unit factors avoid overflowing angle sums and arbitrary
modulo-machine-two-pi reductions. Native subnormal arithmetic remains subject
to the backend's existing floating-point environment.
"""

import argparse
import cmath
from dataclasses import dataclass
import math
import pathlib
import random
import re
import resource
import subprocess
import unittest


class SpectralDomainError(ValueError):
    pass


def identity(size):
    return [[complex(row == column) for column in range(size)] for row in range(size)]


def multiply(lhs, rhs):
    return [[sum(left * right for left, right in zip(row, column))
             for column in zip(*rhs)] for row in lhs]


def adjoint(matrix):
    return [[value.conjugate() for value in column] for column in zip(*matrix)]


def phase_factor(angle):
    if not math.isfinite(angle):
        raise SpectralDomainError('nonfinite angle or intermediate product')
    return cmath.exp(complex(0, angle))


def principal_argument(coefficient):
    if not (math.isfinite(coefficient.real) and math.isfinite(coefficient.imag)) or coefficient == 0:
        raise SpectralDomainError('eigenvalue must be finite and nonzero')
    if coefficient.real < 0 and coefficient.imag == 0:
        return math.pi
    return cmath.phase(coefficient)


def parameter_count(gate):
    if gate not in ('u2', 'u3', 'U'):
        raise SpectralDomainError('unsupported gate family')
    return 2 if gate == 'u2' else 3


def finite_parameter(parameter):
    if type(parameter) not in (int, float):
        raise SpectralDomainError('parameter must be numeric')
    try:
        value = float(parameter)
    except OverflowError as error:
        raise SpectralDomainError('parameter outside finite host range') from error
    if not math.isfinite(value):
        raise SpectralDomainError('nonfinite parameter')
    return value


def validated_parameters(gate, parameters):
    if len(parameters) != parameter_count(gate):
        raise SpectralDomainError('wrong parameter count')
    return tuple(map(finite_parameter, parameters))


def direct_matrix(gate, parameters):
    """Transcribe matrices directly; no quaternion or eigenbasis calculation."""
    values = validated_parameters(gate, parameters)
    theta, phi, lam = (math.pi / 2, *values) if gate == 'u2' else values
    phi_factor, lam_factor = phase_factor(phi), phase_factor(lam)
    if gate == 'U':
        # Published built-in matrix, not exp(i*theta/2) times another helper.
        coefficient = phase_factor(theta)
        return [[(1 + coefficient) / 2, -1j * lam_factor * (1 - coefficient) / 2],
                [1j * phi_factor * (1 - coefficient) / 2,
                 phi_factor * lam_factor * (1 + coefficient) / 2]]
    cosine, sine = ((1 / math.sqrt(2),) * 2 if gate == 'u2'
                    else (math.cos(theta / 2), math.sin(theta / 2)))
    return [[cosine, -lam_factor * sine],
            [phi_factor * sine, phi_factor * lam_factor * cosine]]


@dataclass(frozen=True)
class Spectrum:
    beta: float
    eta: float
    coefficients: tuple[complex, complex]

    def basis(self):
        # Conventional u3(beta, eta, 0), including its own irrelevant basis
        # phase. It cancels in V D V-dagger; do not add a gate global correction.
        cosine, sine = math.cos(self.beta / 2), math.sin(self.beta / 2)
        factor = phase_factor(self.eta)
        return [[cosine, -sine], [factor * sine, factor * cosine]]


def quaternion_spectrum(scalar, vector, overall):
    x, y, z = vector
    radial = math.hypot(x, y)
    length = math.hypot(radial, z)
    if length == 0:
        # Exactly scalar: arbitrary eigenbasis, but both eigenvalues must
        # carry the same full phase, including the -I case.
        return Spectrum(0.0, 0.0, (overall * scalar,) * 2)
    eta = math.atan2(y, x) if radial != 0 else 0.0
    beta = math.atan2(radial, z)
    return Spectrum(beta, eta, (overall * complex(scalar, -length),
                               overall * complex(scalar, length)))


def spectrum(gate, parameters):
    values = validated_parameters(gate, parameters)
    theta, phi, lam = (math.pi / 2, *values) if gate == 'u2' else values
    # No phi+lambda, phi-lambda, or theta+phi+lambda arithmetic. All
    # multiplication below is on bounded unit factors/components.
    phi_half, lam_half = phase_factor(phi / 2), phase_factor(lam / 2)
    sum_factor = phi_half * lam_half
    difference_factor = phi_half * lam_half.conjugate()
    cosine, sine = ((1 / math.sqrt(2),) * 2 if gate == 'u2'
                    else (math.cos(theta / 2), math.sin(theta / 2)))
    scalar = cosine * sum_factor.real
    vector = (-sine * difference_factor.imag, sine * difference_factor.real,
              cosine * sum_factor.imag)
    overall = sum_factor * phase_factor(theta / 2) if gate == 'U' else sum_factor
    return quaternion_spectrum(scalar, vector, overall)


def reconstruct(data, coefficients=None):
    coefficients = data.coefficients if coefficients is None else coefficients
    basis = data.basis()
    return [[sum(basis[row][k] * coefficients[k] * basis[column][k].conjugate() for k in range(2))
             for column in range(2)] for row in range(2)]


def validated_modifiers(modifiers):
    values = []
    for kind, argument in modifiers:
        if kind == 'pow':
            if type(argument) not in (int, float):
                raise SpectralDomainError('exponent must be a constant real number')
            try:
                argument = float(argument)
            except OverflowError as error:
                raise SpectralDomainError('exponent outside finite host range') from error
            if not math.isfinite(argument):
                raise SpectralDomainError('nonfinite exponent')
        elif kind == 'inv':
            if argument is not None:
                raise SpectralDomainError('inv takes no argument')
        elif kind in ('ctrl', 'negctrl'):
            if type(argument) is not int or argument <= 0:
                raise SpectralDomainError('control count must be a positive constant integer')
        else:
            raise SpectralDomainError('unsupported modifier')
        values.append((kind, argument))
    return tuple(values)


def modified_coefficients(coefficients, modifiers):
    for kind, argument in reversed(modifiers):
        if kind == 'inv':
            coefficients = tuple(value.conjugate() for value in coefficients)
        elif kind == 'pow':
            # phase_factor checks each intermediate product before processing
            # any outer modifier, including a later zero power.
            coefficients = tuple(phase_factor(principal_argument(value) * argument) for value in coefficients)
    return coefficients


def condition_matrix(matrix, values):
    pattern = 0
    for value in values:
        pattern = (pattern << 1) | value
    size = len(matrix)
    result = identity(size << len(values))
    for row in range(size):
        for column in range(size):
            result[pattern * size + row][pattern * size + column] = matrix[row][column]
    return result


def reference_matrix(gate, parameter_expressions, modifiers=()):
    modifiers = validated_modifiers(modifiers)
    if len(parameter_expressions) != parameter_count(gate):
        raise SpectralDomainError('wrong parameter count')
    # Model once-only, left-to-right source evaluation, with immediate checks
    # of the original parameters, before any halving or spectral preparation.
    values = []
    for expression in parameter_expressions:
        values.append(finite_parameter(expression()))
    data = spectrum(gate, values)
    coefficients = modified_coefficients(data.coefficients, modifiers)
    controls = tuple(int(kind == 'ctrl') for kind, count in modifiers
                     if kind in ('ctrl', 'negctrl') for _ in range(count))
    return condition_matrix(reconstruct(data, coefficients), controls)


def quadratic_power(matrix, exponent):
    """Separate eigensolver for well-separated test cases, never production."""
    a, b = matrix[0]
    c, d = matrix[1]
    gap = cmath.sqrt((a - d) ** 2 + 4 * b * c)
    if abs(gap) < 1e-5:
        raise ValueError('quadratic cross-check requires well-separated eigenvalues')
    first, second = ((a + d + gap) / 2, (a + d - gap) / 2)
    # Exclude branch-sensitive cases from this roundoff-based cross-check;
    # exact endpoints and adjacent inputs have dedicated tests, not snapping.
    if any(value.real < 0 and abs(value.imag) < 1e-10 for value in (first, second)):
        raise ValueError('quadratic cross-check too close to branch cut')
    coefficients = tuple(phase_factor(principal_argument(value) * exponent) for value in (first, second))
    return [[coefficients[0] * (matrix[row][column] - second * (row == column)) / gap
             + coefficients[1] * (first * (row == column) - matrix[row][column]) / gap
             for column in range(2)] for row in range(2)]


def distance(lhs, rhs):
    return max(abs(a - b) for row, wanted in zip(lhs, rhs) for a, b in zip(row, wanted))


class ReferenceTests(unittest.TestCase):
    def assert_matrix_close(self, actual, expected, tolerance=3e-13):
        self.assertEqual(len(actual), len(expected))
        for row, wanted in zip(actual, expected):
            self.assertEqual(len(row), len(wanted))
            for value, reference in zip(row, wanted):
                self.assertTrue(math.isfinite(value.real) and math.isfinite(value.imag), value)
                self.assertTrue(cmath.isclose(value, reference, rel_tol=tolerance, abs_tol=tolerance),
                                (value, reference))

    @staticmethod
    def powered(gate, parameters, modifiers=(('pow', 0.5),)):
        return reference_matrix(gate, tuple(lambda value=value: value for value in parameters), modifiers)

    def test_reconstruction_against_direct_matrices(self):
        rng = random.Random(352)
        for gate in ('u2', 'u3', 'U'):
            for _ in range(300):
                parameters = tuple(rng.uniform(-20, 20) for _ in range(parameter_count(gate)))
                self.assert_matrix_close(reconstruct(spectrum(gate, parameters)), direct_matrix(gate, parameters))

    def test_basis_is_unitary_and_diagonalizes_direct_matrix(self):
        for gate, parameters in (('u2', (0.4, -0.7)), ('u3', (0.7, 0.4, -0.2)),
                                 ('U', (1.2, -0.7, 0.9)), ('u3', (0, 0.4, 0.7))):
            data = spectrum(gate, parameters)
            basis = data.basis()
            self.assert_matrix_close(multiply(adjoint(basis), basis), identity(2))
            self.assert_matrix_close(multiply(multiply(adjoint(basis), direct_matrix(gate, parameters)), basis),
                                     [[data.coefficients[0], 0], [0, data.coefficients[1]]])

    def test_well_separated_powers_against_independent_quadratic_eigensolver(self):
        rng = random.Random(354)
        checked = 0
        for gate in ('u2', 'u3', 'U'):
            for _ in range(100):
                parameters = tuple(rng.uniform(-5, 5) for _ in range(parameter_count(gate)))
                for exponent in (0.5, -0.5, 1.25, 2.0):
                    try:
                        expected = quadratic_power(direct_matrix(gate, parameters), exponent)
                    except ValueError:
                        continue
                    self.assert_matrix_close(self.powered(gate, parameters, (('pow', exponent),)), expected, 2e-12)
                    checked += 1
        self.assertGreater(checked, 1100)

    def test_u2_specialization_has_no_degeneracy(self):
        for phi in (-1e308, -math.pi, 0, 0.4, math.pi, 1e308):
            for lam in (-1e308, -math.pi, 0, 0.7, math.pi, 1e308):
                data = spectrum('u2', (phi, lam))
                self.assertGreaterEqual(abs(data.coefficients[0] - data.coefficients[1]), math.sqrt(2) - 1e-14)
                self.assert_matrix_close(reconstruct(data), direct_matrix('u3', (math.pi / 2, phi, lam)))

    def test_builtin_u_shares_basis_but_not_full_eigenphases_with_u3(self):
        parameters = (0.7, 0.4, -0.2)
        conventional, builtin = spectrum('u3', parameters), spectrum('U', parameters)
        self.assertEqual((builtin.beta, builtin.eta), (conventional.beta, conventional.eta))
        for first, second in zip(builtin.coefficients, conventional.coefficients):
            self.assertAlmostEqual(first, phase_factor(parameters[0] / 2) * second)
        self.assertGreater(distance(self.powered('U', parameters), self.powered('u3', parameters)), 0.1)
        self.assert_matrix_close(direct_matrix('U', (0, 0, math.pi)), [[1, 0], [0, -1]])

    def test_global_factor_cannot_be_powered_separately(self):
        parameters = (0, 0, 1.5 * math.pi)
        data = spectrum('u3', parameters)
        global_factor = phase_factor(parameters[2] / 2)
        separated = tuple(phase_factor(principal_argument(global_factor) / 2)
                          * phase_factor(principal_argument(value / global_factor) / 2)
                          for value in data.coefficients)
        actual = self.powered('u3', parameters)
        self.assert_matrix_close(actual, [[1, 0], [0, phase_factor(-math.pi / 4)]])
        self.assertGreater(distance(actual, reconstruct(data, separated)), 1.9)

    def test_hadamard_specializations_have_independently_known_roots(self):
        scale = 1 / math.sqrt(2)
        hadamard = [[scale, scale], [scale, -scale]]
        for gate, parameters, shift in (('u2', (0, math.pi), 0),
                                        ('u3', (math.pi / 2, 0, math.pi), 0),
                                        ('U', (math.pi / 2, 0, math.pi), math.pi / 4)):
            # H's projectors are (I+H)/2 and (I-H)/2. For built-in U the
            # full negative eigenvalue rotates to phase -3*pi/4, not +5*pi/4.
            coefficients = (phase_factor(shift / 2),
                            phase_factor((math.pi if shift == 0 else -3 * math.pi / 4) / 2))
            expected = [[coefficients[0] * ((row == column) + hadamard[row][column]) / 2
                         + coefficients[1] * ((row == column) - hadamard[row][column]) / 2
                         for column in range(2)] for row in range(2)]
            self.assert_matrix_close(self.powered(gate, parameters), expected)

    def test_float_integral_powers_match_direct_matrix_products(self):
        for gate, parameters in (('u2', (0.4, -0.2)), ('u3', (0.7, 0.4, -0.2)),
                                 ('U', (0.7, 0.4, -0.2))):
            matrix = direct_matrix(gate, parameters)
            for exponent in (-2.0, 0.0, 2.0):
                operation = adjoint(matrix) if exponent < 0 else matrix
                expected = identity(2)
                for _ in range(int(abs(exponent))):
                    expected = multiply(operation, expected)
                self.assert_matrix_close(self.powered(gate, parameters, (('pow', exponent),)), expected)

    def test_exact_scalar_cases_and_signed_zero_cut(self):
        for scalar in (1, -1):
            for overall in (1 + 0j, phase_factor(0.7), phase_factor(-1.2)):
                data = quaternion_spectrum(scalar, (0, -0.0, 0), overall)
                self.assertEqual((data.beta, data.eta), (0, 0))
                expected = phase_factor(principal_argument(overall * scalar) / 2)
                coefficients = modified_coefficients(data.coefficients, (('pow', 0.5),))
                self.assert_matrix_close(reconstruct(data, coefficients), [[expected, 0], [0, expected]])
        for first in (0.0, -0.0):
            for second in (0.0, -0.0):
                coefficients = (complex(-1, first), complex(-1, second))
                root = modified_coefficients(coefficients, (('pow', 0.5),))
                self.assert_matrix_close(reconstruct(Spectrum(0, 0, coefficients), root), [[1j, 0], [0, 1j]])

    def test_diagonal_and_polar_axes_avoid_undefined_azimuth(self):
        for vector, beta in (((0, 0, 1), 0), ((0, 0, -1), math.pi)):
            data = quaternion_spectrum(0, vector, 1)
            self.assertEqual(data.eta, 0)
            self.assertEqual(data.beta, beta)
            self.assert_matrix_close(reconstruct(data), [[-1j * vector[2], 0], [0, 1j * vector[2]]])
        for theta in (0.0, -0.0):
            self.assert_matrix_close(self.powered('u3', (theta, 0.4, -1.6)),
                                     [[1, 0], [0, phase_factor(-0.6)]])

    def test_tiny_nonzero_axis_is_not_epsilon_merged_or_squared_away(self):
        data = quaternion_spectrum(-1, (0, 1e-300, 0), 1)
        self.assertEqual(data.beta, math.pi / 2)
        self.assertEqual(data.eta, math.pi / 2)
        root = reconstruct(data, modified_coefficients(data.coefficients, (('pow', 0.5),)))
        self.assertGreater(distance(root, [[1j, 0], [0, 1j]]), 0.9)
        # A polar angle can also remain meaningful below the square-root-of-
        # minimum threshold: hypot/atan2 must not use x*x+y*y directly.
        polar = quaternion_spectrum(0, (1e-300, 0, 1), 1)
        self.assertGreater(polar.beta, 0)
        self.assertAlmostEqual(polar.beta / 1e-300, 1)

    def test_near_scalar_branch_sensitivity_is_not_a_decomposition_error(self):
        below, above = math.nextafter(math.tau, 0), math.nextafter(math.tau, math.inf)
        first, second = ((theta, 0, 0) for theta in (below, above))
        self.assertLess(distance(direct_matrix('u3', first), direct_matrix('u3', second)), 2e-15)
        self.assertGreater(distance(self.powered('u3', first), self.powered('u3', second)), 1.9)
        for parameters in (first, second):
            root = self.powered('u3', parameters)
            self.assert_matrix_close(multiply(root, root), direct_matrix('u3', parameters))

    def test_diagonal_branch_cut_neighbors_are_not_snapped(self):
        for sign in (1, -1):
            for lam in (math.nextafter(math.pi, 0), math.nextafter(math.pi, math.inf)):
                parameters = (0, 0, sign * lam)
                actual = self.powered('u3', parameters)
                expected = phase_factor(principal_argument(phase_factor(sign * lam)) / 2)
                self.assert_matrix_close(actual, [[1, 0], [0, expected]])
        for imaginary in (math.ulp(0.0), -math.ulp(0.0), 1e-12, -1e-12):
            self.assertEqual(math.copysign(1, principal_argument(complex(-1, imaginary))), math.copysign(1, imaginary))

    def test_nested_powers_reselect_arguments_and_are_not_flattened(self):
        parameters = (math.pi / 2, 0, 0)
        nested = self.powered('u3', parameters, (('pow', 0.5), ('pow', 6)))
        flattened = self.powered('u3', parameters, (('pow', 3.0),))
        self.assert_matrix_close(nested, direct_matrix('u3', (-math.pi / 2, 0, 0)))
        self.assertGreater(distance(nested, flattened), 1)
        coefficients = (complex(-1, -0.0),) * 2
        before = modified_coefficients(coefficients, (('pow', 0.5), ('inv', None)))
        after = modified_coefficients(coefficients, (('inv', None), ('pow', 0.5)))
        self.assertAlmostEqual(before[0], 1j)
        self.assertAlmostEqual(after[0], -1j)

    def test_large_parameters_need_no_finite_angle_sum(self):
        maximum = float.fromhex('0x1.fffffffffffffp+1023')
        self.assertFalse(math.isfinite(maximum + maximum))
        for gate in ('u2', 'u3', 'U'):
            parameters = (maximum,) * parameter_count(gate)
            self.assert_matrix_close(reconstruct(spectrum(gate, parameters)), direct_matrix(gate, parameters))
            root = self.powered(gate, parameters)
            self.assert_matrix_close(multiply(root, root), direct_matrix(gate, parameters))
            self.assert_matrix_close(multiply(adjoint(root), root), identity(2))
        self.assert_matrix_close(reconstruct(spectrum('u3', (0, maximum, -maximum))), identity(2))

    def test_finite_projector_phases_need_not_have_a_finite_difference(self):
        data = spectrum('u3', (3.0, 0, 0))
        angles = tuple(principal_argument(value) * 1e308 for value in data.coefficients)
        self.assertTrue(all(map(math.isfinite, angles)))
        self.assertFalse(math.isfinite(angles[1] - angles[0]))
        result = self.powered('u3', (3.0, 0, 0), (('pow', 1e308),))
        self.assert_matrix_close(multiply(adjoint(result), result), identity(2))

    def test_reached_nonfinite_parameters_and_products_cannot_hide(self):
        for gate in ('u2', 'u3', 'U'):
            for index in range(parameter_count(gate)):
                for bad in (math.inf, -math.inf, math.nan):
                    parameters = [0.7] * parameter_count(gate)
                    parameters[index] = bad
                    for modifiers in ((('pow', 0.0),), (('ctrl', 1), ('pow', 0.0))):
                        with self.assertRaisesRegex(SpectralDomainError, 'nonfinite parameter'):
                            self.powered(gate, parameters, modifiers)
        with self.assertRaisesRegex(SpectralDomainError, 'intermediate product'):
            self.powered('u3', (6.0, 0, 0), (('pow', 0.0), ('pow', 1e308)))

    def test_validation_precedes_once_only_left_to_right_parameter_evaluation(self):
        calls = []
        suppliers = tuple(lambda index=index: calls.append(index) or 0.7 for index in range(3))
        for modifiers in ((('pow', True),), (('pow', 'runtime'),), (('pow', math.inf),),
                          (('pow', 0.0), ('pow', 10 ** 400)), (('ctrl', 0),), (('inv', 1),)):
            with self.assertRaises(SpectralDomainError):
                reference_matrix('U', suppliers, modifiers)
            self.assertEqual(calls, [])
        reference_matrix('U', suppliers, (('pow', 0.5),))
        self.assertEqual(calls, [0, 1, 2])
        with self.assertRaises(SpectralDomainError):
            reference_matrix('cu', suppliers, (('pow', 0.5),))
        self.assertEqual(calls, [0, 1, 2])

    def test_mixed_controls_preserve_relative_global_phase(self):
        parameters = (0.7, 0.4, -0.2)
        powered = self.powered('U', parameters)
        for controls, values in (((('ctrl', 1),), (1,)), ((('negctrl', 1),), (0,)),
                                 ((('ctrl', 2), ('negctrl', 1)), (1, 1, 0))):
            self.assert_matrix_close(self.powered('U', parameters, (*controls, ('pow', 0.5))),
                                     condition_matrix(powered, values))


class PreparationAuditTests(unittest.TestCase):
    bra = None
    checked = 0

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit runtime U-family spectral preparation')
        from qasm2qcx_runtime_for_numerical import qasm2qcx
        self.module = qasm2qcx

    def fixture(self, gate, arguments, modifiers, source=''):
        module = self.module
        program = module.openqasm3.parser.parse('OPENQASM 3; qubit q; ' + source)
        converter = module.QASM2QCXConverter(program)
        converter.visit(program)
        prefix = ' @ '.join(kind if kind == 'inv' else f'{kind}({argument!r})'
                            for kind, argument in modifiers)
        statement = module.openqasm3.parser.parse(
            f'OPENQASM 3; {prefix} @ {gate}({arguments}) q;').statements[0]
        info = converter._QASM2QCXConverter__analyze_gate_modifiers(
            statement.modifiers, allow_runtime=True, allow_float=True)
        return converter, statement, info

    def print_prepared(self, converter, prepared, sources=()):
        beta, eta, plus, minus = prepared
        lines = converter._QASM2QCXConverter__qcx_lines
        lines.extend([f'PRINTLN {beta}', f'PRINTLN {eta}'])
        for angle in (plus, minus):
            coefficient = converter._QASM2QCXConverter__add_new_temporary_variable(self.module.ValueType.COMPLEX)
            converter._QASM2QCXConverter__emit_phase_coefficient(angle, coefficient)
            lines.extend([f'PRINTLN :REAL:{coefficient}', f'PRINTLN :IMAG:{coefficient}'])
        lines.extend('PRINTLN ' + converter._QASM2QCXConverter__capitalize_variable_name(name) for name in sources)

    def program(self, gate, parameters, modifiers=(('pow', 0.5),), *, runtime=True,
                arguments=None, extra_source='', sources=None, skip=False):
        declarations = ''.join(f'float p{index} = {value!r}; ' for index, value in enumerate(parameters))
        if arguments is None:
            arguments = ', '.join(f'p{index}' if runtime else repr(value) for index, value in enumerate(parameters))
        if sources is None:
            sources = tuple(f'p{index}' for index in range(len(parameters)))
        converter, statement, info = self.fixture(gate, arguments, modifiers, declarations + extra_source)
        if skip:
            converter._QASM2QCXConverter__qcx_lines.append('JUMP SKIPPED')
        with converter._QASM2QCXConverter__powered_u_gate_spectrum(statement, info) as prepared:
            self.print_prepared(converter, prepared, sources)
        if skip:
            converter._QASM2QCXConverter__qcx_lines.extend(['@SKIPPED', 'PRINTLN :INT:99'])
        return '\n'.join(converter._QASM2QCXConverter__qcx_lines) + '\n'

    def execute(self, program):
        return subprocess.run([str(self.bra)], input=program, capture_output=True, text=True, timeout=30)

    def check(self, program, data, modifiers, sources):
        result = self.execute(program)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + program)
        values = list(map(float, result.stdout.split()))
        self.assertEqual(len(values), 6 + len(sources), result.stdout)
        expected = modified_coefficients(data.coefficients, validated_modifiers(modifiers))
        for actual, wanted in zip(values[:2], (data.beta, data.eta)):
            self.assertTrue(math.isclose(actual, wanted, rel_tol=1e-5, abs_tol=2e-6), (values, data))
        for actual, wanted in zip((complex(*values[2:4]), complex(*values[4:6])), expected):
            self.assertTrue(cmath.isclose(actual, wanted, rel_tol=1e-5, abs_tol=2e-6),
                            (actual, wanted, data, modifiers))
        for actual, wanted in zip(values[6:], sources):
            self.assertTrue(math.isclose(actual, wanted, rel_tol=1e-5, abs_tol=2e-6), result.stdout)
        type(self).checked += 1

    def test_runtime_spectra_match_independent_reference(self):
        maximum = float.fromhex('0x1.fffffffffffffp+1023')
        profiles = ((0.7, 0.4, -0.2), (0, 0, 0), (0, 0.4, -1.6), (0, 0, 1.5 * math.pi),
                    (3 * math.pi, 0, 0), (math.tau, 0, 0),
                    (math.nextafter(math.tau, 0), 0, 0), (math.nextafter(math.tau, math.inf), 0, 0),
                    (1e-200, 0, 0), (1e20, -1e20, 1e20), (1e308, 1e308, -1e308),
                    (maximum, maximum, maximum), (0, maximum, -maximum))
        chains = ((('pow', 0.5),), (('pow', -0.5),), (('pow', 0.0),),
                  (('pow', 0.5), ('inv', None)), (('inv', None), ('pow', 0.5)),
                  (('pow', 0.5), ('pow', 6)), (('pow', 6), ('pow', 0.5)))
        for gate in ('u2', 'u3', 'U'):
            for angles in profiles:
                parameters = angles[1:] if gate == 'u2' else angles
                data = spectrum(gate, parameters)
                for modifiers in chains:
                    for runtime in (False, True):
                        with self.subTest(gate=gate, parameters=parameters, modifiers=modifiers, runtime=runtime):
                            self.check(self.program(gate, parameters, modifiers, runtime=runtime), data, modifiers, parameters)

    def test_arrays_arithmetic_and_native_integer_parameters(self):
        cases = (
            ('values[index], values[index + 1], values[index + 2]',
             'array[float, 3] values = {0.7, 0.4, -0.2}; int index = 0;', (0.7, 0.4, -0.2)),
            ('p0 + 0.1, p1, -p2', '', (0.8, 0.4, 0.2)),
            ('n, u, 0', 'int n = -2; uint u = uint(-1);', (-2, self.module.QASM2QCXConverter.QCX_UINT_MAX, 0)),
        )
        for arguments, extra, parameters in cases:
            modifiers = (('pow', 0.5), ('inv', None), ('pow', 6))
            self.check(self.program('U', (0.7, 0.4, -0.2), modifiers, arguments=arguments, extra_source=extra),
                       spectrum('U', parameters), modifiers, (0.7, 0.4, -0.2))

    def test_hadamard_specializations_and_diagonal_branch_neighbors(self):
        for gate in ('u2', 'u3', 'U'):
            parameters = (0, math.pi) if gate == 'u2' else (math.pi / 2, 0, math.pi)
            modifiers = (('pow', 0.5),)
            self.check(self.program(gate, parameters), spectrum(gate, parameters), modifiers, parameters)
        for gate in ('u3', 'U'):
            for sign in (1, -1):
                for lam in (math.nextafter(math.pi, 0), math.nextafter(math.pi, math.inf)):
                    parameters = (0, 0, sign * lam)
                    modifiers = (('pow', 0.5), ('inv', None))
                    self.check(self.program(gate, parameters, modifiers), spectrum(gate, parameters), modifiers, parameters)

    def test_exact_scalar_polar_and_tiny_axes(self):
        for scalar, vector, overall in ((1, (0, 0, 0), 1), (-1, (0, 0, 0), 1),
                (1, (0, 0, 0), phase_factor(0.7)), (-1, (0, 0, 0), phase_factor(-1.2)),
                (0, (0, 0, 1), 1), (0, (0, 0, -1), 1),
                (-1, (0, 1e-300, 0), 1), (0, (1e-300, 0, 1), 1)):
            for modifiers in ((('pow', 0.5),), (('pow', 0.5), ('inv', None)),
                              (('inv', None), ('pow', 0.5))):
                converter, _, info = self.fixture('U', '0, 0, 0', modifiers)
                scalar_name, x, y, z = (converter._QASM2QCXConverter__add_new_temporary_variable(self.module.ValueType.FLOAT)
                                       for _ in range(4))
                overall_name = converter._QASM2QCXConverter__add_new_temporary_variable(self.module.ValueType.COMPLEX)
                lines = converter._QASM2QCXConverter__qcx_lines
                lines.extend(f'LET {name} := {value!r}' for name, value in zip((scalar_name, x, y, z), (scalar, *vector)))
                converter._QASM2QCXConverter__emit_complex_components(
                    overall_name, repr(complex(overall).real), repr(complex(overall).imag))
                beta, eta, coefficients = converter._QASM2QCXConverter__emit_u_gate_eigenbasis(
                    scalar_name, (x, y, z), overall_name)
                phases = tuple((converter._QASM2QCXConverter__add_new_temporary_variable(self.module.ValueType.FLOAT), coefficient)
                               for coefficient in coefficients)
                for angle, coefficient in phases:
                    converter._QASM2QCXConverter__emit_principal_phase(angle, coefficient)
                converter._QASM2QCXConverter__emit_phase_modifiers(
                    phases, info, converter._QASM2QCXConverter__phase_power_factors(info), preserve_inverse_coefficients=True)
                self.print_prepared(converter, (beta, eta, phases[0][0], phases[1][0]))
                self.check('\n'.join(lines) + '\n', quaternion_spectrum(scalar, vector, overall), modifiers, ())

    def test_finite_eigenphases_with_overflowing_difference(self):
        modifiers = (('pow', 1e308),)
        self.check(self.program('u3', (3.0, 0, 0), modifiers), spectrum('u3', (3.0, 0, 0)), modifiers, (3.0, 0, 0))

    def test_nonfinite_original_parameters_and_products_cannot_hide(self):
        for gate in ('u2', 'u3', 'U'):
            for index in range(parameter_count(gate)):
                for expression in ('1.0 / 0.0', '-1.0 / 0.0', '0.0 / 0.0', '1e308 * 1e308'):
                    arguments = ['0.7'] * parameter_count(gate)
                    arguments[index] = expression
                    for modifiers in ((('pow', 0.0),), (('ctrl', 1), ('pow', 0.0))):
                        program = self.program(gate, (0.7,) * len(arguments), modifiers, arguments=', '.join(arguments))
                        result = self.execute(program)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('nonfinite', result.stderr)
                        self.assertEqual(result.stdout, '')
                    skipped = self.execute(self.program(gate, (0.7,) * len(arguments), arguments=', '.join(arguments), skip=True))
                    self.assertEqual(skipped.returncode, 0, skipped.stderr)
                    self.assertEqual(skipped.stdout.strip(), '99')
        for gate, parameters in (('u2', (0, math.pi)), ('u3', (6.0, 0, 0)), ('U', (0, 0, 3.0))):
            modifiers = (('pow', 0.0), ('pow', 1e308))
            result = self.execute(self.program(gate, parameters, modifiers))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('nonfinite', result.stderr)
            self.assertEqual(result.stdout, '')
            skipped = self.execute(self.program(gate, parameters, modifiers, skip=True))
            self.assertEqual(skipped.returncode, 0, skipped.stderr)
            self.assertEqual(skipped.stdout.strip(), '99')


class BackendAuditTests(unittest.TestCase):
    bra = None
    checked = 0

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit existing U-family and eigenbasis gates')

    @staticmethod
    def apply(matrix, state):
        return [sum(value * amplitude for value, amplitude in zip(row, state)) for row in matrix]

    def check(self, lines, expected):
        result = subprocess.run([str(self.bra)], input='\n'.join([*lines, 'DO AMPLITUDES']) + '\n',
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + '\n' + '\n'.join(lines))
        actual = {int(index, 2): complex(float(real), float(imaginary))
                  for index, real, imaginary in re.findall(
                      r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
        self.assertEqual(set(actual), set(range(len(expected))), result.stdout)
        for index, wanted in enumerate(expected):
            self.assertTrue(cmath.isclose(actual[index], wanted, rel_tol=1e-5, abs_tol=2e-6),
                            (lines, index, actual[index], wanted))
        type(self).checked += 1

    @staticmethod
    def inputs(qubits, basis):
        if basis is None:
            return ([f'H {wire}' for wire in range(qubits)] + ['S 0'],
                    [(1j if index & 1 else 1) / math.sqrt(1 << qubits) for index in range(1 << qubits)])
        return ([f'X {wire}' for wire in range(qubits) if basis & (1 << wire)],
                [complex(index == basis) for index in range(1 << qubits)])

    def test_existing_ordinary_and_inverse_conversion(self):
        from qasm2qcx_runtime_for_numerical import qasm2qcx
        maximum = float.fromhex('0x1.fffffffffffffp+1023')
        profiles = ((0.7, 0.4, -0.2), (0, 0, 0), (0, 0.4, -1.6),
                    (math.tau, 0, 0), (math.nextafter(math.tau, 0), 0, 0),
                    (math.nextafter(math.tau, math.inf), 0, 0), (1e-200, 0, 0),
                    (1e20, -1e20, 1e20), (1e308, 1e308, -1e308),
                    (maximum, maximum, maximum))
        for gate in ('u2', 'u3', 'U'):
            for angles in profiles:
                parameters = angles[1:] if gate == 'u2' else angles
                for values in ((), (1,), (0,), (0, 1)):
                    qubits = len(values) + 1
                    for inverse in (False, True):
                        operation = direct_matrix(gate, parameters)
                        if inverse:
                            operation = adjoint(operation)
                        operation = condition_matrix(operation, values)
                        for runtime in (False, True):
                            declarations = ''.join(f'float p{index} = {value!r}; '
                                                   for index, value in enumerate(parameters)) if runtime else ''
                            arguments = ', '.join(f'p{index}' if runtime else repr(value)
                                                  for index, value in enumerate(parameters))
                            modifiers = ''.join(('ctrl' if value else 'negctrl') + ' @ ' for value in values)
                            modifiers += 'inv @ ' if inverse else ''
                            operands = ', '.join(f'q[{wire}]' for wire in reversed(range(qubits)))
                            for basis in (*range(1 << qubits), None):
                                _, initial = self.inputs(qubits, basis)
                                # QASM source preparation is independent of QCX's formatting.
                                source_preparation = ''.join(f'x q[{wire}]; ' for wire in range(qubits)
                                                             if basis is not None and basis & (1 << wire))
                                if basis is None:
                                    source_preparation = 'h q; s q[0]; '
                                source = ('OPENQASM 3; include "stdgates.inc"; ' + declarations
                                          + f'qubit[{qubits}] q; ' + source_preparation
                                          + f'{modifiers}{gate}({arguments}) {operands};')
                                with self.subTest(gate=gate, parameters=parameters, values=values,
                                                  inverse=inverse, runtime=runtime, basis=basis):
                                    self.check(qasm2qcx.convert(source), self.apply(operation, initial))

    def test_existing_qcx_basis_and_projector_phases(self):
        for gate, parameters in (('u2', (0.4, -0.2)), ('u3', (0.7, 0.4, -0.2)), ('U', (0.7, 0.4, -0.2))):
            data = spectrum(gate, parameters)
            for exponent in (0.5, -0.5, 1.25):
                # Independent characteristic-polynomial reference, not the
                # basis/eigenvalue helper used to assemble this QCX circuit.
                powered = quadratic_power(direct_matrix(gate, parameters), exponent)
                coefficients = modified_coefficients(data.coefficients, (('pow', exponent),))
                phases = tuple(principal_argument(value) for value in coefficients)
                for values in ((), (1,), (0,), (0, 1)):
                    qubits = len(values) + 1
                    controls = [f'{wire}' + ('#0' if not value else '')
                                for wire, value in zip(reversed(range(1, qubits)), values)]
                    phase_gate = 'U1' if not values else f'C{len(values)}U1'
                    for fused in (False, True):
                        for basis in (*range(1 << qubits), None):
                            preparation, initial = self.inputs(qubits, basis)
                            sequence = ([f'U3+ 0 {data.beta!r} {data.eta!r} 0.0',
                                ' '.join([phase_gate, *controls, '0#0', repr(phases[0])]),
                                ' '.join([phase_gate, *controls, '0', repr(phases[1])]),
                                f'U3 0 {data.beta!r} {data.eta!r} 0.0'])
                            if fused:
                                sequence = ['BEGIN FUSION', *sequence, 'END FUSION']
                            expected = self.apply(condition_matrix(powered, values), initial)
                            with self.subTest(gate=gate, exponent=exponent, values=values, fused=fused, basis=basis):
                                self.check([f'QUBITS {qubits}', *preparation, *sequence], expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    options = parser.parse_args()
    if options.bra is not None:
        BackendAuditTests.bra = options.bra.resolve()
        PreparationAuditTests.bra = options.bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if BackendAuditTests.checked:
        print(f'Existing U-family/eigenbasis circuits checked: {BackendAuditTests.checked}')
    if PreparationAuditTests.checked:
        print(f'Prepared U-family spectra checked: {PreparationAuditTests.checked}')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
