#!/usr/bin/env python3

"""Numerical-policy references and lowering helpers for phase-gate powers.

  python3 bra/test/qasm2qcx_phase_power_numerical.py
  python3 bra/test/qasm2qcx_phase_power_numerical.py --bra /path/to/bra

The public converter feature is not enabled yet. With --bra, audit existing
native phase gates and the Stage 2 EXP/ARG lowering helpers separately.

Policy for the parameterized floating-power path:
* Exponents are finite compile-time floats or constant integers. Runtime
  exponents/factors remain outside this subset. Validate all modifiers before
  evaluating an angle, including modifiers hidden by an outer zero power.
* Evaluate the angle expression exactly once on each reached classical path.
  Nonfinite angles and nonfinite intermediate products are errors, even at
  zero power or under inactive quantum controls. Skipped classical paths do
  not evaluate angles or perform these runtime checks.
* Select principal arguments from exp(i*angle), as used by ket::utility::exp_i,
  rather than reducing the angle modulo a machine approximation of 2*pi.
* Exact negative-real eigenvalues (including either sign of imaginary zero)
  have argument +pi. Do not snap nearby eigenvalues to this endpoint. A negative
  argument very near the cut can round to machine -PI; preserve its side rather
  than blindly replacing every scalar -PI result by +PI.
* In particular, an evaluated machine -PI angle has exp(i*angle) with a tiny
  negative imaginary component on typical libm implementations. It is near,
  not exactly on, the cut; its root need not equal the fixed Z gate's root.
  This is an unavoidable numerical distinction, not a different fixed-gate
  convention. Test exact endpoints via complex(-1, +/-0), not an epsilon rule.
* Apply modifiers innermost first. Before every power, select the argument of
  the current eigenvalue again. Inversion negates the current angle. Controls
  leave inactive eigenvalues at 1 and never skip classical parameter checks.
* Initially reject intermediate overflow, rather than silently emitting NaN or
  assuming a zero outer power removes the need to evaluate inner operations.
  This can reject mathematically finite unitary powers with huge multipliers;
  extending that range is separate numerical work. Underflow follows ordinary
  floating-point rounding. Existing integer-only/fixed-family paths are unchanged.

The mathematical oracle does not import converter reduction or power helpers.
The separate lowering audit executes emitted QCX against this oracle.
"""

import argparse
import cmath
import math
import pathlib
import resource
import subprocess
import unittest


class PhasePowerDomainError(ValueError):
    pass


def validate_modifiers(modifiers):
    for kind, argument in modifiers:
        if kind == 'pow':
            if type(argument) not in (int, float):
                raise PhasePowerDomainError('power exponent must be a constant real number')
            if isinstance(argument, float) and not math.isfinite(argument):
                raise PhasePowerDomainError('power exponent must be finite')
        elif kind == 'inv':
            if argument is not None:
                raise PhasePowerDomainError('inv takes no argument')
        elif kind in ('ctrl', 'negctrl'):
            if type(argument) is not int or argument <= 0:
                raise PhasePowerDomainError('control count must be a positive constant integer')
        else:
            raise PhasePowerDomainError(f'unsupported modifier {kind}')


def phase_factor(angle):
    if not math.isfinite(angle):
        raise PhasePowerDomainError('angle must be finite')
    return cmath.exp(complex(0, angle))


def principal_argument(coefficient):
    if not (math.isfinite(coefficient.real) and math.isfinite(coefficient.imag)) or coefficient == 0:
        raise PhasePowerDomainError('coefficient must be finite and nonzero')
    if coefficient.real < 0 and coefficient.imag == 0:
        return math.pi
    # Do not remap an atan2 result merely because it rounded to machine -PI.
    return cmath.phase(coefficient)


def powered_phase_factor(angle_expression, modifiers):
    """The supplier models once-only evaluation on a reached classical path."""
    validate_modifiers(modifiers)
    angle = angle_expression()
    phase_factor(angle)  # Also validate for zero powers or inactive controls.
    for kind, argument in reversed(modifiers):
        if kind == 'inv':
            angle = -angle
        elif kind == 'pow':
            phase = principal_argument(phase_factor(angle))
            try:
                angle = phase * argument
            except OverflowError as error:
                raise PhasePowerDomainError('intermediate power product overflow') from error
            if not math.isfinite(angle):
                raise PhasePowerDomainError('intermediate power product must be finite')
    return phase_factor(angle)


def reference_diagonal(gate, factor, control_values=()):
    """Control arguments are most significant, with a phase target last."""
    if gate not in ('p', 'phase', 'u1', 'cp', 'cphase', 'gphase'):
        raise ValueError(gate)
    if gate in ('cp', 'cphase'):
        control_values = (*control_values, 1)
    if any(value not in (0, 1) for value in control_values):
        raise ValueError('invalid control value')
    pattern = 0
    for value in control_values:
        pattern = (pattern << 1) | value
    target_size = 1 if gate == 'gphase' else 2
    diagonal = [1 + 0j] * (target_size << len(control_values))
    diagonal[pattern * target_size + target_size - 1] = factor
    return diagonal


class ReferenceTests(unittest.TestCase):
    def assert_complex_close(self, actual, expected):
        self.assertTrue(cmath.isclose(actual, expected, rel_tol=1e-14, abs_tol=1e-14), (actual, expected))

    def test_exact_negative_real_endpoint_including_signed_zero(self):
        for zero in (0.0, -0.0):
            self.assertEqual(principal_argument(complex(-1, zero)), math.pi)
            self.assert_complex_close(cmath.exp(0.5j * principal_argument(complex(-1, zero))), 1j)

    def test_nearby_eigenvalues_keep_their_side_even_if_argument_rounds_to_pi(self):
        tiny = math.ulp(0.0)
        for imaginary in (tiny, -tiny, 1e-12, -1e-12):
            with self.subTest(imaginary=imaginary):
                phase = principal_argument(complex(-1, imaginary))
                self.assertEqual(math.copysign(1, phase), math.copysign(1, imaginary))
                self.assertEqual(math.copysign(1, cmath.exp(0.5j * phase).imag), math.copysign(1, imaginary))

    def test_adjacent_machine_angles_do_not_get_tolerance_snapped(self):
        below, above = math.nextafter(math.pi, 0), math.nextafter(math.pi, math.inf)
        self.assertGreater(powered_phase_factor(lambda: below, (('pow', 0.5),)).imag, 0)
        self.assertLess(powered_phase_factor(lambda: above, (('pow', 0.5),)).imag, 0)
        self.assertGreater(powered_phase_factor(lambda: -above, (('pow', 0.5),)).imag, 0)
        self.assertLess(powered_phase_factor(lambda: -below, (('pow', 0.5),)).imag, 0)

    def test_wrapping_precedes_multiplication(self):
        actual = powered_phase_factor(lambda: 3 * math.pi / 2, (('pow', 0.5),))
        self.assert_complex_close(actual, cmath.exp(-0.25j * math.pi))
        naive = cmath.exp(0.75j * math.pi)
        self.assertGreater(abs(actual - naive), 1)

    def test_large_angles_are_not_reduced_modulo_machine_two_pi(self):
        for angle in (1e20, -1e20, 1e100, -1e100, 1e308, -1e308):
            with self.subTest(angle=angle):
                original = phase_factor(angle)
                reduced = cmath.exp(1j * math.remainder(angle, math.tau))
                self.assertGreater(abs(original - reduced), 0.1)
                actual = powered_phase_factor(lambda: angle, (('pow', 0.5),))
                self.assert_complex_close(actual * actual, original)
                self.assertGreaterEqual(actual.real, 0)  # Principal root.

    def test_nested_powers_reselect_the_argument(self):
        # The inner power produces phase 3*pi/2; its principal phase is -pi/2.
        actual = powered_phase_factor(lambda: math.pi / 4, (('pow', 0.5), ('pow', 6)))
        self.assert_complex_close(actual, cmath.exp(-0.25j * math.pi))
        naive = powered_phase_factor(lambda: math.pi / 4, (('pow', 3.0),))
        self.assertGreater(abs(actual - naive), 1)

    def test_inversion_order_matches_spectral_definition_away_from_the_cut(self):
        angle = 0.7
        for modifiers in ((('inv', None), ('pow', 0.5)), (('pow', 0.5), ('inv', None))):
            self.assert_complex_close(powered_phase_factor(lambda: angle, modifiers), cmath.exp(-0.35j))

    def test_nonfinite_angle_is_rejected_even_at_zero_power_or_inactive_control(self):
        for angle in (math.inf, -math.inf, math.nan):
            for modifiers in ((('pow', 0.0),), (('ctrl', 1), ('pow', 0.0)),
                              (('negctrl', 1), ('pow', 0.5))):
                with self.subTest(angle=angle, modifiers=modifiers), self.assertRaises(PhasePowerDomainError):
                    powered_phase_factor(lambda: angle, modifiers)

    def test_intermediate_overflow_is_not_hidden_by_outer_zero(self):
        for modifiers in ((('pow', 1e308),), (('pow', 0.0), ('pow', 1e308)),
                          (('pow', 0.5), ('pow', 10 ** 1000))):
            with self.subTest(modifiers=modifiers), self.assertRaisesRegex(PhasePowerDomainError, 'product'):
                powered_phase_factor(lambda: 3.0, modifiers)
        self.assert_complex_close(powered_phase_factor(lambda: 3.0, (('pow', 0.0),)), 1)

    def test_zero_signed_zero_and_subnormal_inputs(self):
        for angle in (0.0, -0.0):
            self.assert_complex_close(powered_phase_factor(lambda: angle, (('pow', -0.5),)), 1)
        self.assert_complex_close(powered_phase_factor(lambda: math.ulp(0.0), (('pow', 0.5),)), 1)
        for exponent in (0.0, -0.0):
            self.assert_complex_close(powered_phase_factor(lambda: 0.7, (('pow', exponent),)), 1)

    def test_exponents_and_controls_are_validated_before_angle_evaluation(self):
        for invalid in ((('pow', 0.0), ('pow', math.nan)), (('pow', math.inf),),
                        (('pow', True),), (('pow', 0.5), ('ctrl', 0)), (('inv', 1),)):
            calls = []
            with self.subTest(invalid=invalid), self.assertRaises(PhasePowerDomainError):
                powered_phase_factor(lambda: calls.append(1) or 0.7, invalid)
            self.assertEqual(calls, [])

    def test_once_only_capture_and_skipped_classical_paths(self):
        calls = []
        supplier = lambda: calls.append(0.7) or 0.7
        powered_phase_factor(supplier, (('pow', 0.5), ('inv', None), ('pow', 2)))
        self.assertEqual(calls, [0.7])
        supplier = lambda: calls.append('bad') or math.inf
        for reached in (False, True):
            if reached:
                with self.assertRaises(PhasePowerDomainError):
                    powered_phase_factor(supplier, (('pow', 0.0),))
        self.assertEqual(calls, [0.7, 'bad'])

    def test_diagonal_phase_and_global_phase_keep_control_patterns(self):
        factor = cmath.exp(-0.25j * math.pi)
        for gate in ('p', 'phase', 'u1'):
            self.assertEqual(reference_diagonal(gate, factor), [1, factor])
        for gate in ('cp', 'cphase'):
            self.assertEqual(reference_diagonal(gate, factor), [1, 1, 1, factor])
        self.assertEqual(reference_diagonal('gphase', factor), [factor])
        self.assertEqual(reference_diagonal('gphase', factor, (0, 1)), [1, factor, 1, 1])
        self.assertEqual(reference_diagonal('p', factor, (0, 1)), [1, 1, 1, factor, 1, 1, 1, 1])


class LoweringAuditTests(unittest.TestCase):
    bra = None

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit EXP/ARG lowering helpers')
        from qasm2qcx_runtime_for_numerical import qasm2qcx
        self.module = qasm2qcx

    def program(self, angle, modifiers, *, coefficient_setup=None, skip=False, setup=None):
        module = self.module
        program = module.openqasm3.parser.parse('OPENQASM 3; qubit q;')
        converter = module.QASM2QCXConverter(program)
        converter.visit(program)
        lines = converter._QASM2QCXConverter__qcx_lines
        lines.extend(['VAR SOURCE REAL', f'LET SOURCE := {angle!r}'])
        if setup:
            lines.extend(setup)
        if skip:
            lines.append('JUMP SKIPPED')
        with converter._QASM2QCXConverter__captured_phase_angle(
                ('SOURCE', module.ValueType.FLOAT, module.ValueKind.LVALUE)) as (storage, coefficient):
            if coefficient_setup:
                lines.extend(line.format(coefficient=coefficient) for line in coefficient_setup)
            for kind, exponent in reversed(modifiers):
                if kind == 'pow':
                    converter._QASM2QCXConverter__emit_principal_phase(storage, coefficient)
                    lines.append(f'LET {storage} *= {float(exponent)!r}')
                elif kind == 'inv':
                    lines.append(f'LET {storage} *= -1.0')
                else:
                    continue
                converter._QASM2QCXConverter__emit_phase_coefficient(storage, coefficient)
            lines.extend([f'PRINTLN :REAL:{coefficient}', f'PRINTLN :IMAG:{coefficient}', 'PRINTLN SOURCE'])
        if skip:
            lines.extend(['@SKIPPED', 'PRINTLN :INT:99'])
        return '\n'.join(lines) + '\n'

    def execute(self, program):
        return subprocess.run([str(self.bra)], input=program, text=True, capture_output=True, timeout=30)

    def test_ordered_powers_match_independent_reference(self):
        angles = (0.0, -0.0, 0.7, -0.7, math.pi, -math.pi, 1.5 * math.pi,
                  math.nextafter(math.pi, 0), math.nextafter(math.pi, math.inf),
                  1e20, -1e20, 1e100, -1e100, 1e308, -1e308)
        chains = ((('pow', 0.5),), (('pow', -0.5),), (('pow', 0.0),),
                  (('pow', 0.5), ('pow', 6)), (('inv', None), ('pow', 0.5)),
                  (('pow', 0.5), ('inv', None)))
        for angle in angles:
            for modifiers in chains:
                with self.subTest(angle=angle, modifiers=modifiers):
                    program = self.program(angle, modifiers)
                    result = self.execute(program)
                    self.assertEqual(result.returncode, 0, result.stderr + program)
                    real, imaginary, source = map(float, result.stdout.split())
                    self.assertTrue(cmath.isclose(complex(real, imaginary),
                        powered_phase_factor(lambda: angle, modifiers), rel_tol=1e-5, abs_tol=2e-6), result.stdout)
                    self.assertTrue(math.isclose(source, angle, rel_tol=1e-5, abs_tol=2e-6))

    def test_exact_negative_real_endpoints_and_nearby_coefficients(self):
        # Build the coefficient without an approximate PI angle. Subnormal
        # arithmetic follows the backend environment; these near-cut probes
        # use normal values, including values whose ARG rounds to -PI.
        for imaginary in (0.0, -0.0, 1e-300, -1e-300, 1e-12, -1e-12):
            setup = ['LET {coefficient} := ' + (':MINUS_I' if math.copysign(1, imaginary) < 0 else ':I'),
                     'LET {coefficient} *= :COMPLEX:' + repr(abs(imaginary)),
                     'LET {coefficient} -= :COMPLEX:1.0']
            with self.subTest(imaginary=imaginary):
                result = self.execute(self.program(0.0, (('pow', 0.5),), coefficient_setup=setup))
                self.assertEqual(result.returncode, 0, result.stderr)
                real, imaginary_result, _ = map(float, result.stdout.split())
                expected = cmath.exp(0.5j * principal_argument(complex(-1, imaginary)))
                self.assertTrue(cmath.isclose(complex(real, imaginary_result), expected,
                                             rel_tol=1e-5, abs_tol=2e-6), result.stdout)

    def test_nonfinite_angles_and_intermediate_products_are_not_hidden(self):
        for initial, setup in ((1.0, ['LET SOURCE /= 0.0']),
                               (-1.0, ['LET SOURCE /= 0.0']),
                               (0.0, ['LET SOURCE /= 0.0'])):
            for modifiers in ((('pow', 0.0),), (('ctrl', 1), ('pow', 0.0))):
                with self.subTest(initial=initial, modifiers=modifiers):
                    result = self.execute(self.program(initial, modifiers, setup=setup))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('nonfinite', result.stderr)
                    self.assertEqual(result.stdout, '')
                    skipped = self.execute(self.program(initial, modifiers, setup=setup, skip=True))
                    self.assertEqual(skipped.returncode, 0, skipped.stderr)
                    self.assertEqual(skipped.stdout.strip(), '99')
        result = self.execute(self.program(3.0, (('pow', 0.0), ('pow', 1e308))))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('nonfinite', result.stderr)
        self.assertEqual(result.stdout, '')


class BackendAuditTests(unittest.TestCase):
    bra = None

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit existing phase gates')

    def test_existing_phase_gates_use_the_unreduced_angle(self):
        from qasm2qcx_inverse_numerical import check_amplitudes
        for angle in (0.7, 3 * math.pi / 2, math.pi, -math.pi, 1e20, -1e20, 1e100, -1e100, 1e308, -1e308):
            for runtime in (False, True):
                for gate in ('p', 'phase', 'u1', 'cp', 'cphase', 'gphase'):
                    with self.subTest(angle=angle, runtime=runtime, gate=gate):
                        factor = phase_factor(angle)
                        diagonal = reference_diagonal(gate, factor)
                        # Use one qubit even for a zero-qubit gate.
                        if gate == 'gphase':
                            diagonal = [factor, factor]
                        qubits = len(diagonal).bit_length() - 1
                        expected = [value / math.sqrt(len(diagonal)) for value in diagonal]
                        argument = 'theta' if runtime else repr(angle)
                        operands = '' if gate == 'gphase' else ('q[1], q[0]' if gate in ('cp', 'cphase') else 'q[0]')
                        check_amplitudes(self.bra, f'qubit[{qubits}] q; float theta = {angle!r}; h q; '
                                         + f'{gate}({argument}) {operands};', expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    args = parser.parse_args()
    if args.bra is not None:
        BackendAuditTests.bra = args.bra.resolve()
        LoweringAuditTests.bra = args.bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
