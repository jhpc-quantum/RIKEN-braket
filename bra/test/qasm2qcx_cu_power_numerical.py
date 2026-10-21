#!/usr/bin/env python3

"""Independent references for principal floating powers of cu (issue #358).

  python3 bra/test/qasm2qcx_cu_power_numerical.py
  python3 bra/test/qasm2qcx_cu_power_numerical.py --bra /path/to/bra

The reference-only run uses Python's standard library. Reuse the independent
U-family math references, never converter spectral helpers. cu's active block
is exp(i*gamma)*U, NOT exp(i*gamma)*conventional-u3. Multiply gamma's FULL-angle
factor into BOTH active eigenvalues before applying ordered spectral powers.
The intrinsic positive control and any added controls leave inactive blocks
exactly identity; a scalar active block is still a relative controlled phase.

The optional default-double, non-MPI audit exercises existing ordinary/integer
cu conversion, raw QCX basis/projector-phase instructions, private runtime
spectral preparation, and complete converted floating-power circuits.
Ordinary cu currently forms gamma+theta/2;
its audit uses profiles where that sum is well conditioned. Independent
references also cover loss/overflow in that sum, which must not be inherited
by the new floating path. No ordinary-path numerical fix is made here.
"""

import argparse
import cmath
import math
import pathlib
import random
import resource
import subprocess
import unittest

import qasm2qcx_u_power_numerical as uref


def validated_parameters(parameters):
    if len(parameters) != 4:
        raise uref.SpectralDomainError('cu requires four parameters')
    return tuple(map(uref.finite_parameter, parameters))


def direct_active_matrix(parameters):
    theta, phi, lam, gamma = validated_parameters(parameters)
    # This uses the independent transcription of the published built-in U
    # matrix, not its quaternion spectrum or the converter's phase correction.
    factor = uref.phase_factor(gamma)
    return [[factor * value for value in row] for row in uref.direct_matrix('U', (theta, phi, lam))]


def spectrum(parameters):
    theta, phi, lam, gamma = validated_parameters(parameters)
    data = uref.spectrum('U', (theta, phi, lam))
    factor = uref.phase_factor(gamma)
    return uref.Spectrum(data.beta, data.eta, tuple(factor * value for value in data.coefficients))


def reference_matrix(parameter_expressions, modifiers=()):
    modifiers = uref.validated_modifiers(modifiers)
    if len(parameter_expressions) != 4:
        raise uref.SpectralDomainError('cu requires four parameters')
    # Validate factors before evaluating source expressions once, left to
    # right. The fourth parameter is checked even at zero power/inactive control.
    parameters = tuple(uref.finite_parameter(expression()) for expression in parameter_expressions)
    data = spectrum(parameters)
    coefficients = uref.modified_coefficients(data.coefficients, modifiers)
    added = tuple(int(kind == 'ctrl') for kind, count in modifiers
                  if kind in ('ctrl', 'negctrl') for _ in range(count))
    return uref.condition_matrix(uref.reconstruct(data, coefficients), (*added, 1))


def powered(parameters, modifiers=(('pow', 0.5),)):
    return reference_matrix(tuple(lambda value=value: value for value in parameters), modifiers)


def integer_reference(parameters, exponent, inverse=False, controls=()):
    # Independently multiply the complete directly transcribed active matrix;
    # do not route ordinary/integer audits through floating spectral powers.
    operation = direct_active_matrix(parameters)
    if inverse != (exponent < 0):
        operation = uref.adjoint(operation)
    result = uref.identity(2)
    for _ in range(abs(exponent)):
        result = uref.multiply(operation, result)
    return uref.condition_matrix(result, (*controls, 1))


class ReferenceTests(unittest.TestCase):
    assert_matrix_close = uref.ReferenceTests.assert_matrix_close

    def test_active_spectrum_reconstructs_published_matrix(self):
        rng = random.Random(358)
        for _ in range(300):
            parameters = tuple(rng.uniform(-10, 10) for _ in range(4))
            self.assert_matrix_close(uref.reconstruct(spectrum(parameters)), direct_active_matrix(parameters))

    def test_gamma_preserves_basis_but_rotates_both_full_eigenvalues(self):
        parameters = (0.7, 0.4, -0.2, 0.9)
        data, base = spectrum(parameters), uref.spectrum('U', parameters[:3])
        self.assertEqual((data.beta, data.eta), (base.beta, base.eta))
        for actual, original in zip(data.coefficients, base.coefficients):
            self.assertAlmostEqual(actual, uref.phase_factor(parameters[3]) * original)
            self.assertGreater(abs(actual - uref.phase_factor(parameters[3] / 2) * original), 0.4)

    def test_builtin_u_phase_must_not_be_replaced_by_conventional_u3(self):
        parameters = (math.pi / 2, 0, math.pi, 0)
        correct = direct_active_matrix(parameters)
        conventional = uref.direct_matrix('u3', parameters[:3])
        self.assertGreater(uref.distance(correct, conventional), 0.5)
        # exp(i*pi/4)*H squared is i*I on the active block, not I.
        self.assert_matrix_close(powered(parameters, (('pow', 2.0),)),
                                 [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1j, 0], [0, 0, 0, 1j]])

    def test_gamma_must_be_included_before_principal_argument_selection(self):
        parameters = (0, 0, 1.5 * math.pi, -0.75 * math.pi)
        actual = powered(parameters)
        expected = uref.condition_matrix([[uref.phase_factor(-3 * math.pi / 8), 0],
                                          [0, uref.phase_factor(3 * math.pi / 8)]], (1,))
        self.assert_matrix_close(actual, expected)
        base = uref.spectrum('U', parameters[:3])
        gamma_root = uref.phase_factor(uref.principal_argument(uref.phase_factor(parameters[3])) / 2)
        separately_powered = tuple(gamma_root * uref.phase_factor(uref.principal_argument(value) / 2)
                                  for value in base.coefficients)
        wrong = uref.condition_matrix(uref.reconstruct(base, separately_powered), (1,))
        self.assertGreater(uref.distance(actual, wrong), 1.9)

    def test_well_separated_roots_against_independent_quadratic_solver(self):
        rng = random.Random(359)
        checked = 0
        for _ in range(150):
            parameters = tuple(rng.uniform(-5, 5) for _ in range(4))
            for exponent in (0.5, -0.5, 1.25):
                try:
                    expected = uref.quadratic_power(direct_active_matrix(parameters), exponent)
                except ValueError:
                    continue
                self.assert_matrix_close(powered(parameters, (('pow', exponent),)),
                                         uref.condition_matrix(expected, (1,)), 2e-12)
                checked += 1
        self.assertGreater(checked, 400)

    def test_scalar_active_blocks_remain_controlled_relative_phases(self):
        for gamma in (0.0, 0.7, -1.2, 1.5 * math.pi, 1e308):
            factor = uref.phase_factor(uref.principal_argument(uref.phase_factor(gamma)) / 2)
            self.assert_matrix_close(powered((0, 0, 0, gamma)),
                                     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, factor, 0], [0, 0, 0, factor]])
        for first in (0.0, -0.0):
            for second in (0.0, -0.0):
                data = uref.Spectrum(0, 0, (complex(-1, first), complex(-1, second)))
                coefficients = uref.modified_coefficients(data.coefficients, (('pow', 0.5),))
                self.assert_matrix_close(uref.condition_matrix(uref.reconstruct(data, coefficients), (1,)),
                                         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1j, 0], [0, 0, 0, 1j]])

    def test_active_eigenvalue_one_can_coincide_with_inactive_identity(self):
        self.assert_matrix_close(powered((0, 0, 1.0, 0)),
                                 [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                  [0, 0, 0, uref.phase_factor(0.5)]])
        self.assert_matrix_close(powered((0, 0, 0, 0)), uref.identity(4))

    def test_gamma_branch_neighbors_are_not_tolerance_snapped(self):
        for sign in (1, -1):
            below, above = (sign * math.nextafter(math.pi, toward) for toward in (0, math.inf))
            left, right = powered((0, 0, 0, below)), powered((0, 0, 0, above))
            for gamma, result in ((below, left), (above, right)):
                expected = uref.phase_factor(uref.principal_argument(uref.phase_factor(gamma)) / 2)
                self.assertAlmostEqual(result[2][2], expected)
                self.assertAlmostEqual(result[3][3], expected)
            self.assertGreater(abs(left[2][2] - right[2][2]), 1.9)

    def test_near_scalar_cut_sensitivity_is_not_a_reason_to_merge(self):
        theta = 1e-12
        near = (theta, 0, 0, math.pi - theta / 2)
        scalar = (0, 0, 0, math.pi)
        self.assertLess(uref.distance(direct_active_matrix(near), direct_active_matrix(scalar)), 2e-12)
        self.assertGreater(uref.distance(powered(near), powered(scalar)), 0.9)
        root = powered(near)
        self.assert_matrix_close(uref.multiply(root, root), uref.condition_matrix(direct_active_matrix(near), (1,)))

    def test_nested_powers_and_inverse_order(self):
        parameters = (0, 0, 0, 1.5 * math.pi)
        nested = powered(parameters, (('pow', 0.5), ('pow', 2)))
        flattened = powered(parameters, (('pow', 1.0),))
        self.assertGreater(uref.distance(nested, flattened), 1.9)
        exact = (complex(-1, -0.0),) * 2
        before = uref.modified_coefficients(exact, (('pow', 0.5), ('inv', None)))
        after = uref.modified_coefficients(exact, (('inv', None), ('pow', 0.5)))
        self.assertAlmostEqual(before[0], 1j)
        self.assertAlmostEqual(after[0], -1j)

    def test_added_mixed_controls_leave_every_inactive_subspace_identity(self):
        parameters = (0.7, 0.4, -0.2, 0.9)
        base = powered(parameters)
        for modifiers, values in (((('ctrl', 1),), (1,)), ((('negctrl', 1),), (0,)),
                                  ((('negctrl', 1), ('ctrl', 2)), (0, 1, 1))):
            result = powered(parameters, (*modifiers, ('pow', 0.5)))
            self.assert_matrix_close(result, uref.condition_matrix(base, values))
            for index in range(len(result)):
                # The least significant control (wire 1) is intrinsic and
                # always positive, independently of added control polarities.
                if not index & 2:
                    self.assertEqual(result[index], [complex(index == column) for column in range(len(result))])

    def test_float_integral_powers_match_direct_matrix_products(self):
        parameters = (0.7, 0.4, -0.2, 0.9)
        for exponent in (-2, 0, 3):
            self.assert_matrix_close(powered(parameters, (('pow', float(exponent)),)), integer_reference(parameters, exponent))

    def test_large_finite_parameters_need_no_finite_phase_sum(self):
        maximum = float.fromhex('0x1.fffffffffffffp+1023')
        parameters = (maximum,) * 4
        self.assertFalse(math.isfinite(parameters[3] + parameters[0] / 2))
        self.assert_matrix_close(uref.reconstruct(spectrum(parameters)), direct_active_matrix(parameters))
        root = powered(parameters)
        self.assert_matrix_close(uref.multiply(root, root), uref.condition_matrix(direct_active_matrix(parameters), (1,)))
        self.assert_matrix_close(uref.multiply(uref.adjoint(root), root), uref.identity(4))

    def test_finite_active_eigenphases_need_not_have_a_finite_difference(self):
        parameters = (3.0, 0, 0, -1.5)
        phases = tuple(uref.principal_argument(value) * 1e308 for value in spectrum(parameters).coefficients)
        self.assertTrue(all(map(math.isfinite, phases)))
        self.assertFalse(math.isfinite(phases[1] - phases[0]))
        result = powered(parameters, (('pow', 1e308),))
        self.assert_matrix_close(uref.multiply(uref.adjoint(result), result), uref.identity(4))

    def test_combining_large_gamma_and_small_theta_can_lose_a_phase(self):
        parameters = (0.7, 0.4, -0.2, 1e20)
        self.assertEqual(parameters[3] + parameters[0] / 2, parameters[3])
        combined = uref.phase_factor(parameters[3] + parameters[0] / 2)
        wrong = [[combined * value for value in row] for row in uref.direct_matrix('u3', parameters[:3])]
        self.assertGreater(uref.distance(direct_active_matrix(parameters), wrong), 0.3)

    def test_original_gamma_and_intermediate_overflow_are_checked(self):
        for index in range(4):
            for invalid in (math.inf, -math.inf, math.nan):
                parameters = [0.7] * 4
                parameters[index] = invalid
                for modifiers in ((('pow', 0.0),), (('ctrl', 1), ('pow', 0.0))):
                    with self.assertRaisesRegex(uref.SpectralDomainError, 'nonfinite parameter'):
                        powered(parameters, modifiers)
        with self.assertRaisesRegex(uref.SpectralDomainError, 'intermediate product'):
            powered((0, 0, 0, 3.0), (('pow', 0.0), ('pow', 1e308)))

    def test_factors_are_checked_before_four_once_only_parameter_evaluations(self):
        calls = []
        suppliers = tuple(lambda index=index: calls.append(index) or 0.7 for index in range(4))
        for modifiers in ((('pow', True),), (('pow', 'runtime'),), (('pow', math.inf),),
                          (('pow', 0.0), ('pow', 10 ** 400)), (('ctrl', 0),)):
            with self.assertRaises(uref.SpectralDomainError):
                reference_matrix(suppliers, modifiers)
            self.assertEqual(calls, [])
        reference_matrix(suppliers, (('pow', 0.5),))
        self.assertEqual(calls, [0, 1, 2, 3])
        with self.assertRaises(uref.SpectralDomainError):
            reference_matrix(suppliers[:3], (('pow', 0.5),))
        self.assertEqual(calls, [0, 1, 2, 3])


class PreparationAuditTests(unittest.TestCase):
    bra = None
    checked = 0
    print_prepared = uref.PreparationAuditTests.print_prepared
    execute = uref.PreparationAuditTests.execute
    check = uref.PreparationAuditTests.check

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit runtime cu spectral preparation')
        from qasm2qcx_runtime_for_numerical import qasm2qcx
        self.module = qasm2qcx

    def fixture(self, gate, arguments, modifiers, source=''):
        self.assertEqual(gate, 'cu')
        controls = sum(count for kind, count in modifiers if kind in ('ctrl', 'negctrl'))
        qubits = controls + 2
        program = self.module.openqasm3.parser.parse(f'OPENQASM 3; qubit[{qubits}] q; ' + source)
        converter = self.module.QASM2QCXConverter(program)
        converter.visit(program)
        prefix = ' @ '.join(kind if kind == 'inv' else f'{kind}({argument!r})' for kind, argument in modifiers)
        operands = ', '.join(f'q[{wire}]' for wire in reversed(range(qubits)))
        statement = self.module.openqasm3.parser.parse(
            f'OPENQASM 3; {prefix} @ cu({arguments}) {operands};').statements[0]
        info = converter._QASM2QCXConverter__analyze_gate_modifiers(
            statement.modifiers, allow_runtime=True, allow_float=True)
        return converter, statement, info

    def program(self, parameters, modifiers=(('pow', 0.5),), **options):
        return uref.PreparationAuditTests.program(self, 'cu', parameters, modifiers, **options)

    def test_full_gamma_factor_and_shared_basis_match_independent_reference(self):
        maximum = float.fromhex('0x1.fffffffffffffp+1023')
        profiles = ((0.7, 0.4, -0.2, 0.9), (0, 0, 0, 0), (0, 0, 0, 0.7),
                    (0, 0, 0, math.nextafter(math.pi, 0)), (0, 0, 0, math.nextafter(math.pi, math.inf)),
                    (0, 0, 1.5 * math.pi, -0.75 * math.pi), (0.7, 0.4, -0.2, 1e20),
                    (1e20, -1e20, 1e20, -1e20), (maximum,) * 4,
                    (1e308, 1e308, -1e308, -1e308), (0, maximum, -maximum, maximum),
                    (1e-200, 0, 0, math.pi), (1e-12, 0, 0, math.pi - 5e-13))
        chains = ((('pow', 0.5),), (('pow', -0.5),), (('pow', 0.0),),
                  (('pow', 0.5), ('inv', None)), (('inv', None), ('pow', 0.5)),
                  (('pow', 0.5), ('pow', 6)), (('pow', 6), ('pow', 0.5)),
                  (('negctrl', 1), ('pow', 0.5), ('ctrl', 1)))
        for parameters in profiles:
            for modifiers in chains:
                for runtime in (False, True):
                    with self.subTest(parameters=parameters, modifiers=modifiers, runtime=runtime):
                        self.check(self.program(parameters, modifiers, runtime=runtime), spectrum(parameters), modifiers, parameters)

    def test_array_arithmetic_integer_and_unsigned_gamma_parameters(self):
        maximum_uint = self.module.QASM2QCXConverter.QCX_UINT_MAX
        minimum_int = self.module.QASM2QCXConverter.QCX_INT_MIN
        cases = (
            (', '.join(f'values[index + {offset}]' for offset in range(4)),
             'array[float, 4] values = {0.7, 0.4, -0.2, 0.9}; int index = 0;', (0.7, 0.4, -0.2, 0.9)),
            ('p0, p1, p2, p3 + 0.1', '', (0.7, 0.4, -0.2, 1.0)),
            ('p0, p1, p2, p0', '', (0.7, 0.4, -0.2, 0.7)),
            ('p0, p1, p2, n', f'int n = {minimum_int};', (0.7, 0.4, -0.2, minimum_int)),
            ('p0, p1, p2, u', 'uint u = uint(-1);', (0.7, 0.4, -0.2, maximum_uint)),
        )
        original = (0.7, 0.4, -0.2, 0.9)
        modifiers = (('pow', 0.5), ('inv', None), ('pow', 6))
        for arguments, extra, parameters in cases:
            self.check(self.program(original, modifiers, arguments=arguments, extra_source=extra),
                       spectrum(parameters), modifiers, original)

    def test_finite_active_eigenphases_need_no_finite_difference(self):
        parameters = (3.0, 0, 0, -1.5)
        modifiers = (('pow', 1e308),)
        self.check(self.program(parameters, modifiers), spectrum(parameters), modifiers, parameters)

    def test_nonfinite_inputs_and_intermediate_products_cannot_hide(self):
        for index in range(4):
            for expression in ('1.0 / 0.0', '-1.0 / 0.0', '0.0 / 0.0', '1e308 * 1e308'):
                arguments = ['0.7'] * 4
                arguments[index] = expression
                for modifiers in ((('pow', 0.0),), (('ctrl', 1), ('pow', 0.0)),
                                  (('negctrl', 1), ('pow', 0.0))):
                    result = self.execute(self.program((0.7,) * 4, modifiers, arguments=', '.join(arguments)))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('nonfinite', result.stderr)
                    self.assertEqual(result.stdout, '')
                skipped = self.execute(self.program((0.7,) * 4, arguments=', '.join(arguments), skip=True))
                self.assertEqual(skipped.returncode, 0, skipped.stderr)
                self.assertEqual(skipped.stdout.strip(), '99')
        parameters, modifiers = (0, 0, 0, 3.0), (('pow', 0.0), ('pow', 1e308))
        result = self.execute(self.program(parameters, modifiers))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('nonfinite', result.stderr)
        self.assertEqual(result.stdout, '')
        skipped = self.execute(self.program(parameters, modifiers, skip=True))
        self.assertEqual(skipped.returncode, 0, skipped.stderr)
        self.assertEqual(skipped.stdout.strip(), '99')


class BackendAuditTests(unittest.TestCase):
    bra = None
    checked = 0
    check = uref.BackendAuditTests.check
    apply = staticmethod(uref.BackendAuditTests.apply)
    inputs = staticmethod(uref.BackendAuditTests.inputs)

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit existing cu conversion and projector gates')

    def test_existing_ordinary_inverse_and_integer_power_conversion(self):
        from qasm2qcx_runtime_for_numerical import qasm2qcx
        maximum = float.fromhex('0x1.fffffffffffffp+1023')
        profiles = ((0.7, 0.4, -0.2, 0.3), (0, 0, 0, 0.7),
                    (math.pi / 2, 0, math.pi, -0.2), (0, 0, 3.0, -0.5),
                    (0, 0.4, -0.2, 1e20), (1e20, 0.4, -0.2, 0),
                    (maximum, maximum, maximum, 0), (0, 0, 0, maximum))
        for profile, parameters in enumerate(profiles):
            for values in ((), (1,), (0,), (0, 1)):
                qubits = len(values) + 2
                for inverse in (False, True):
                    for runtime in (False, True):
                        declarations = ''.join(f'float p{index} = {value!r}; '
                                               for index, value in enumerate(parameters)) if runtime else ''
                        arguments = ', '.join(f'p{index}' if runtime else repr(value)
                                              for index, value in enumerate(parameters))
                        prefix = ''.join(('ctrl' if value else 'negctrl') + ' @ ' for value in values)
                        prefix += 'inv @ ' if inverse else ''
                        operands = ', '.join(f'q[{wire}]' for wire in reversed(range(qubits)))
                        for exponent in (None, -2, 0, 3):
                            actual_power = 1 if exponent is None else exponent
                            operation = integer_reference(parameters, actual_power, inverse, values)
                            power = '' if exponent is None else f'pow({"n" if runtime else exponent}) @ '
                            count = f'int n = {actual_power}; ' if exponent is not None and runtime else ''
                            for basis in ((*range(1 << qubits), None) if profile < 2 else (None,)):
                                _, initial = self.inputs(qubits, basis)
                                preparation = ('h q; s q[0]; ' if basis is None else
                                    ''.join(f'x q[{wire}]; ' for wire in range(qubits) if basis & (1 << wire)))
                                source = ('OPENQASM 3; include "stdgates.inc"; ' + declarations + count
                                    + f'qubit[{qubits}] q; ' + preparation + prefix + power + f'cu({arguments}) {operands};')
                                with self.subTest(parameters=parameters, values=values, inverse=inverse,
                                                  runtime=runtime, exponent=exponent, basis=basis):
                                    self.check(qasm2qcx.convert(source), self.apply(operation, initial))

    def test_existing_qcx_basis_and_intrinsically_controlled_projector_phases(self):
        for parameters in ((0.7, 0.4, -0.2, 0.9), (0, 0, 0, 0.7), (0, 0, 3, -1)):
            data = spectrum(parameters)
            for exponent in (0.5, -0.5, 1.25):
                if parameters[:3] == (0, 0, 0):
                    coefficient = uref.phase_factor(uref.principal_argument(uref.phase_factor(parameters[3])) * exponent)
                    active = [[coefficient, 0], [0, coefficient]]
                else:
                    active = uref.quadratic_power(direct_active_matrix(parameters), exponent)
                coefficients = uref.modified_coefficients(data.coefficients, (('pow', exponent),))
                phases = tuple(uref.principal_argument(value) for value in coefficients)
                for values in ((), (1,), (0,), (0, 1)):
                    qubits = len(values) + 2
                    conditions = [str(wire) + ('#0' if not value else '')
                                  for wire, value in zip(reversed(range(2, qubits)), values)] + ['1']
                    mnemonic = 'CU1' if len(conditions) == 1 else f'C{len(conditions)}U1'
                    operation = uref.condition_matrix(active, (*values, 1))
                    for fused in (False, True):
                        for basis in (*range(1 << qubits), None):
                            preparation, initial = self.inputs(qubits, basis)
                            sequence = [f'U3+ 0 {data.beta!r} {data.eta!r} 0.0',
                                ' '.join([mnemonic, *conditions, '0#0', repr(phases[0])]),
                                ' '.join([mnemonic, *conditions, '0', repr(phases[1])]),
                                f'U3 0 {data.beta!r} {data.eta!r} 0.0']
                            if fused:
                                sequence = ['BEGIN FUSION', *sequence, 'END FUSION']
                            with self.subTest(parameters=parameters, exponent=exponent, values=values, fused=fused, basis=basis):
                                self.check([f'QUBITS {qubits}', *preparation, *sequence], self.apply(operation, initial))


class ConversionAuditTests(unittest.TestCase):
    bra = None
    checked = 0
    check = BackendAuditTests.check
    apply = staticmethod(BackendAuditTests.apply)
    inputs = staticmethod(BackendAuditTests.inputs)
    check_source = uref.ConversionAuditTests.check_source

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit converted cu powers')
        from qasm2qcx_runtime_for_numerical import qasm2qcx
        self.module = qasm2qcx

    def check_gate(self, parameters, modifiers=(('pow', 0.5),), *, runtime=True, basis=None, fused=False):
        operation = powered(parameters, modifiers)
        qubits = len(operation).bit_length() - 1
        _, initial = self.inputs(qubits, basis)
        preparation = ('h q; s q[0]; ' if basis is None else
                       ''.join(f'x q[{wire}]; ' for wire in range(qubits) if basis & (1 << wire)))
        declarations = ''.join(f'float p{index} = {value!r}; ' for index, value in enumerate(parameters))
        arguments = ', '.join(f'p{index}' if runtime else repr(value) for index, value in enumerate(parameters))
        prefix = ' @ '.join(kind if kind == 'inv' else f'{kind}({argument!r})' for kind, argument in modifiers)
        operands = ', '.join(f'q[{wire}]' for wire in reversed(range(qubits)))
        self.check_source(f'qubit[{qubits}] q; ' + declarations + preparation
                          + f'{prefix} @ cu({arguments}) {operands};', self.apply(operation, initial), fused=fused)

    def test_full_circuits_against_analytic_spectra(self):
        maximum = float.fromhex('0x1.fffffffffffffp+1023')
        profiles = ((0.7, 0.4, -0.2, 0.9), (0, 0, 0, 0), (0, 0, 0, 0.7),
                    (0, 0, 1.5 * math.pi, -0.75 * math.pi),
                    (math.pi / 2, 0, math.pi, 0), (1e-200, 0, 0, math.pi),
                    (1e-12, 0, 0, math.pi - 5e-13),
                    (0, 0, 0, math.nextafter(math.pi, 0)),
                    (0, 0, 0, math.nextafter(math.pi, math.inf)),
                    (0.7, 0.4, -0.2, 1e20), (1e20, -1e20, 1e20, -1e20),
                    (maximum,) * 4, (1e308, 1e308, -1e308, -1e308),
                    (0, maximum, -maximum, maximum))
        chains = ((('pow', 0.5),), (('pow', -0.5),), (('pow', 0.0),),
                  (('pow', 2.0),), (('pow', 0.5), ('inv', None)),
                  (('inv', None), ('pow', 0.5)), (('pow', 0.5), ('pow', 6)),
                  (('pow', 6), ('pow', 0.5)))
        for index, parameters in enumerate(profiles):
            for modifiers in chains:
                for runtime in (False, True):
                    for basis in ((*range(4), None) if index < 5 else (None,)):
                        with self.subTest(parameters=parameters, modifiers=modifiers, runtime=runtime, basis=basis):
                            self.check_gate(parameters, modifiers, runtime=runtime, basis=basis)
        for fused in (False, True):
            self.check_gate((3.0, 0, 0, -1.5), (('pow', 1e308),), fused=fused)

    def test_seeded_circuits_against_independent_quadratic_solver(self):
        rng = random.Random(3580)
        for _ in range(20):
            parameters = tuple(rng.uniform(-5, 5) for _ in range(4))
            for exponent in (0.5, -0.75):
                operation = uref.condition_matrix(uref.quadratic_power(direct_active_matrix(parameters), exponent), (1,))
                _, initial = self.inputs(2, None)
                arguments = ', '.join(map(repr, parameters))
                for fused in (False, True):
                    self.check_source(f'qubit[2] q; h q; s q[0]; pow({exponent!r}) @ cu({arguments}) q[1], q[0];',
                                      self.apply(operation, initial), fused=fused)

    def test_mixed_controls_and_scalar_relative_phase_with_fusion(self):
        for parameters in ((0.7, 0.4, -0.2, 0.9), (0, 0, 0, 0.7)):
            for values in ((), (0,), (1,), (0, 1), (1, 0)):
                controls = tuple(('ctrl' if value else 'negctrl', 1) for value in values)
                modifiers = (('pow', 0.5), *controls, ('inv', None))
                qubits = len(values) + 2
                for fused in (False, True):
                    for basis in (*range(1 << qubits), None):
                        self.check_gate(parameters, modifiers, basis=basis, fused=fused)
        for fused in (False, True):
            for basis in (*range(32), None):
                self.check_gate((0.7, 0.4, -0.2, 0.9),
                                (('ctrl', 2), ('pow', 0.5), ('negctrl', 1)), basis=basis, fused=fused)

    def test_noncontiguous_wires_and_spectators(self):
        active = uref.quadratic_power(direct_active_matrix((0.7, 0.4, -0.2, 0.9)), 0.5)
        initial = [(1j if index & 1 else 1) * (-1j if index & 4 else 1) / math.sqrt(32) for index in range(32)]
        expected = initial.copy()
        for index in range(32):
            if not index & 4 and index & 8 and index & 16 and not index & 2:
                expected[index], expected[index | 2] = self.apply(active, (initial[index], initial[index | 2]))
        for fused in (False, True):
            self.check_source('qubit[5] q; h q; s q[0]; sdg q[2]; '
                'negctrl @ pow(0.5) @ ctrl @ cu(0.7, 0.4, -0.2, 0.9) q[2], q[3], q[4], q[1];', expected, fused=fused)

    def test_broadcast_arrays_and_reused_temporaries(self):
        parameters = (0.7, 0.4, -0.2, 0.9)
        operation = powered(parameters)
        _, pair = self.inputs(2, None)
        low = self.apply(operation, pair)
        high = self.apply(operation, (0.5,) * 4)
        self.check_source('qubit[2] lo; qubit[2] hi; h lo; h hi; s lo[0]; '
            'pow(0.5) @ cu(0.7, 0.4, -0.2, 0.9) hi, lo;',
            [high[(h << 1) | l] * low[(i << 1) | j]
             for h in range(2) for i in range(2) for l in range(2) for j in range(2)])
        # The tensor factors above occupy interleaved physical pairs:
        # (hi[0],lo[0]) and (hi[1],lo[1]), not adjacent register halves.
        _, state = self.inputs(2, None)
        for gamma, exponent in ((0.9, 0.5), (1.2, -0.5), (0.7, 0.5)):
            state = self.apply(powered((0.7, 0.4, -0.2, gamma), (('pow', exponent),)), state)
        for fused in (False, True):
            self.check_source('qubit[2] q; h q; s q[0]; array[float, 4] values = {0.7, 0.4, -0.2, 0.9}; int index = 0; '
                'pow(0.5) @ cu(values[index], values[index+1], values[index+2], values[index+3]) q[1], q[0]; '
                'values[3] = 1.2; pow(-0.5) @ cu(values[0], values[1], values[2], values[3]) q[1], q[0]; '
                'pow(0.5) @ cu(values[0], values[1], values[2], values[0]) q[1], q[0];', state, fused=fused)
        for declaration, gamma in (('int n = -2;', -2), ('uint n = uint(-1);', self.module.QASM2QCXConverter.QCX_UINT_MAX)):
            self.check_source('qubit[2] q; h q; s q[0]; ' + declaration
                + 'pow(0.5) @ cu(0.7, 0.4, -0.2, n) q[1], q[0];',
                self.apply(powered((0.7, 0.4, -0.2, gamma)), pair))

    def test_runtime_loops_and_skipped_invalid_arithmetic(self):
        _, initial = self.inputs(2, None)
        state = initial
        for gamma in (1.0, 2.0, 3.0):
            state = self.apply(powered((0.7, 0.4, -0.2, gamma)), state)
        self.check_source('qubit[2] q; h q; s q[0]; int first = 1; int last = 3; '
            'for int i in [first:last] { pow(0.5) @ cu(0.7, 0.4, -0.2, float(i)) q[1], q[0]; }', state)
        state = initial
        for gamma in (0.9, 1.4):
            state = self.apply(powered((0.7, 0.4, -0.2, gamma)), state)
        self.check_source('qubit[2] q; h q; s q[0]; int count = 0; float gamma = 0.9; '
            'while (count < 3) { count += 1; if (count == 2) { continue; } '
            'pow(0.5) @ cu(0.7, 0.4, -0.2, gamma) q[1], q[0]; gamma += 0.5; if (count == 3) { break; } }', state)
        for call in ('pow(0.0) @ cu(0, 0, 0, 1.0 / 0.0) q[1], q[0];',
                     'pow(0.0) @ pow(1e308) @ cu(0, 0, 0, 3.0) q[1], q[0];'):
            for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                self.check_source('qubit[2] q; h q; s q[0]; bool flag = false; ' + wrapper.format(call), initial)

    def test_nonfinite_parameters_fail_even_with_inactive_controls_and_zero_power(self):
        for index in range(4):
            for expression in ('1.0 / 0.0', '-1.0 / 0.0', '0.0 / 0.0', 'large * large'):
                arguments = ['0.7'] * 4
                arguments[index] = 'bad'
                for prefix, operands in (('', 'q[1], q[0]'), ('ctrl @ ', 'q[2], q[1], q[0]'),
                                         ('negctrl @ ', 'q[2], q[1], q[0]')):
                    source = ('OPENQASM 3; include "stdgates.inc"; qubit[3] q; float large = 1e308; '
                        + f'float bad = {expression}; {prefix}pow(0.0) @ cu({", ".join(arguments)}) {operands};')
                    result = subprocess.run([str(self.bra)], input='\n'.join(self.module.convert(source)) + '\n',
                                            text=True, capture_output=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('nonfinite', result.stderr)
                    self.assertEqual(result.stdout, '')


    def test_intermediate_overflow_cannot_be_hidden_by_outer_zero(self):
        for prefix, operands in (('', 'q[1], q[0]'), ('ctrl @ ', 'q[2], q[1], q[0]'),
                                 ('negctrl @ ', 'q[2], q[1], q[0]')):
            source = ('OPENQASM 3; include "stdgates.inc"; qubit[3] q; ' + prefix
                      + f'pow(0.0) @ pow(1e308) @ cu(0, 0, 0, 3.0) {operands};')
            result = subprocess.run([str(self.bra)], input='\n'.join(self.module.convert(source)) + '\n',
                                    text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('nonfinite', result.stderr)
            self.assertEqual(result.stdout, '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    options = parser.parse_args()
    if options.bra is not None:
        BackendAuditTests.bra = options.bra.resolve()
        PreparationAuditTests.bra = options.bra.resolve()
        ConversionAuditTests.bra = options.bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if BackendAuditTests.checked:
        print(f'Existing cu/intrinsic-projector circuits checked: {BackendAuditTests.checked}')
    if PreparationAuditTests.checked:
        print(f'Prepared cu spectra checked: {PreparationAuditTests.checked}')
    if ConversionAuditTests.checked:
        print(f'Converted cu power circuits checked: {ConversionAuditTests.checked}')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
