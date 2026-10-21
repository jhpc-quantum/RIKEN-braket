#!/usr/bin/env python3

"""Regress ordinary/integer cu phase-sum loss and overflow (issue #360).

  python3 bra/test/qasm2qcx_cu_phase_numerical.py
  python3 bra/test/qasm2qcx_cu_phase_numerical.py --bra /path/to/bra

Reference tests use only Python's standard library. Expected matrices come
from the independent direct cu matrix and integer matrix products, not from
converter phase operands or floating-power spectral preparation. Compare
complex amplitudes without aligning away global or controlled relative phase.

The converter regressions check literal and runtime parameters. The native
audit independently verifies the separate-factor instruction sequence,
including inverse/integer sequences and fusion.
Converted circuits additionally cover mixed controls, integer exponent signs
and types, indexed arrays, broadcasts, source preservation, and loop re-entry.
"""

import argparse
import cmath
import math
import pathlib
import re
import resource
import unittest

import qasm2qcx_cu_power_numerical as cref


LOSS = (0.7, 0.4, -0.2, 1e20)
OVERFLOW = (1.5e308, 0.4, -0.2, 1.5e308)


def apply_cu(state, parameters, exponent=1, inverse=False, *, target=0, controls=((1, 1),)):
    """Embed the independently multiplied active matrix on explicit wires."""
    operation = cref.integer_reference(parameters, exponent, inverse)
    active = [row[2:] for row in operation[2:]]
    result = state.copy()
    for zero in range(len(state)):
        if zero & (1 << target) or any((zero >> wire) & 1 != value for wire, value in controls):
            continue
        one = zero | (1 << target)
        result[zero], result[one] = (
            active[0][0] * state[zero] + active[0][1] * state[one],
            active[1][0] * state[zero] + active[1][1] * state[one])
    return result


class ReferenceTests(unittest.TestCase):
    assert_matrix_close = cref.uref.ReferenceTests.assert_matrix_close

    def test_large_gamma_discards_theta_correction_and_changes_relative_phase(self):
        theta, phi, lam, gamma = LOSS
        self.assertEqual(gamma + theta / 2, gamma)
        correct = cref.direct_active_matrix(LOSS)
        factor = cref.uref.phase_factor(gamma + theta / 2)
        wrong = [[factor * value for value in row]
                 for row in cref.uref.direct_matrix('u3', (theta, phi, lam))]
        self.assertGreater(cref.uref.distance(correct, wrong), 0.3)
        # Entry magnitudes agree: probabilities alone would miss this bug.
        self.assert_matrix_close([[abs(value) for value in row] for row in correct],
                                 [[abs(value) for value in row] for row in wrong])
        correct, wrong = (cref.uref.condition_matrix(matrix, (1,)) for matrix in (correct, wrong))
        self.assertEqual(correct[0], wrong[0])  # Same inactive control block.
        # An active amplitude ratio is not the inactive block's identity phase.
        self.assertGreater(abs(correct[2][2] / wrong[2][2] - 1), 0.3)

    def test_finite_parameters_can_have_an_overflowing_sum_but_finite_unitary(self):
        for sign in (1, -1):
            parameters = tuple(sign * value for value in OVERFLOW)
            self.assertTrue(all(map(math.isfinite, parameters)))
            self.assertFalse(math.isfinite(parameters[3] + parameters[0] / 2))
            operation = cref.integer_reference(parameters, 1)
            self.assertTrue(all(math.isfinite(value.real) and math.isfinite(value.imag)
                                for row in operation for value in row))
            self.assert_matrix_close(cref.uref.multiply(cref.uref.adjoint(operation), operation),
                                     cref.uref.identity(4))

    def test_separate_phase_factors_match_direct_matrix_without_combined_angles(self):
        for parameters in (LOSS, OVERFLOW, tuple(-value for value in LOSS), tuple(-value for value in OVERFLOW)):
            theta, phi, lam, gamma = parameters
            factor = cref.uref.phase_factor(gamma) * cref.uref.phase_factor(theta / 2)
            factored = [[factor * value for value in row]
                        for row in cref.uref.direct_matrix('u3', (theta, phi, lam))]
            self.assert_matrix_close(factored, cref.direct_active_matrix(parameters))

    def test_explicit_wire_embedding_matches_full_controlled_matrix(self):
        state = [value / math.sqrt(30) for value in (1, 2j, -3, -4j)]
        for parameters in (LOSS, OVERFLOW):
            for exponent in (-2, 0, 1, 3):
                operation = cref.integer_reference(parameters, exponent)
                expected = cref.BackendAuditTests.apply(operation, state)
                actual = apply_cu(state, parameters, exponent)
                for value, wanted in zip(actual, expected):
                    self.assertAlmostEqual(value, wanted)
                self.assertEqual(actual[:2], state[:2])


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    check = cref.BackendAuditTests.check
    apply = staticmethod(cref.BackendAuditTests.apply)
    inputs = staticmethod(cref.BackendAuditTests.inputs)

    def setUp(self):
        if self.bra is None:
            self.skipTest('pass --bra to audit cu phase-sum regressions')
        from qasm2qcx_runtime_for_numerical import qasm2qcx
        self.module = qasm2qcx

    def check_source(self, source, expected, *, fused=False, preserved=()):
        source = source if source.startswith('OPENQASM ') else 'OPENQASM 3; include "stdgates.inc"; ' + source
        program = self.module.openqasm3.parser.parse(source)
        converter = self.module.QASM2QCXConverter(program)
        converter.visit(program)
        lines = list(converter)
        if fused:
            lines.insert(1, 'BEGIN FUSION')
            lines.append('END FUSION')
        # Exact native comparisons check source storage independently of
        # amplitude comparisons, including after all queued gates are flushed.
        for name, value in preserved:
            operand = converter._QASM2QCXConverter__capitalize_variable_name(name)
            lines.append(f'ASSERT {operand} == {value!r}')
        self.check(lines, expected)

    def check_ordinary_conversion(self, parameters, *, runtime):
        declarations = ''.join(f'float p{index} = {value!r}; ' for index, value in enumerate(parameters))
        arguments = ', '.join(f'p{index}' if runtime else repr(value) for index, value in enumerate(parameters))
        _, initial = self.inputs(2, None)
        self.check_source('qubit[2] q; ' + declarations + 'h q; s q[0]; '
                          + f'cu({arguments}) q[1], q[0];',
                          self.apply(cref.integer_reference(parameters, 1), initial))

    def test_literal_cu_retains_small_theta_beside_large_gamma(self):
        self.check_ordinary_conversion(LOSS, runtime=False)

    def test_runtime_cu_retains_small_theta_beside_large_gamma(self):
        self.check_ordinary_conversion(LOSS, runtime=True)

    def test_literal_cu_accepts_finite_parameters_with_overflowing_sum(self):
        self.check_ordinary_conversion(OVERFLOW, runtime=False)

    def test_runtime_cu_accepts_finite_parameters_with_overflowing_sum(self):
        self.check_ordinary_conversion(OVERFLOW, runtime=True)

    def test_documented_large_gamma_example(self):
        document = (pathlib.Path(__file__).parents[1] / 'qcx/qasm2qcx.md').read_text(encoding='utf-8')
        examples = re.findall(r'<!-- cu-phase-factor-example -->\s*```qasm\n(.*?)\n```',
                              document, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        # Published built-in U matrix on target |0>, with the intrinsic
        # control in superposition. No gamma+theta/2 correction is used here.
        theta, gamma = cmath.exp(0.7j), cmath.exp(1e20j)
        scale = 1 / (2 * math.sqrt(2))
        expected = [1 / math.sqrt(2), 0, gamma * (1 + theta) * scale, 1j * gamma * (1 - theta) * scale]
        for fused in (False, True):
            self.check_source(examples[0], expected, fused=fused,
                              preserved=(('theta', 0.7), ('gamma', 1e20)))

    def test_native_separate_factors_preserve_complete_matrix_and_integer_powers(self):
        for parameters in (LOSS, OVERFLOW, tuple(-value for value in OVERFLOW)):
            theta, phi, lam, gamma = parameters
            forward = [f'U1 1 {gamma!r}', f'U1 1 {theta / 2!r}',
                       f'CU3 1 0 {theta!r} {phi!r} {lam!r}']
            adjoint = [f'CU3+ 1 0 {theta!r} {phi!r} {lam!r}',
                       f'U1+ 1 {theta / 2!r}', f'U1+ 1 {gamma!r}']
            for exponent in (1, -1, 2, -2, 0):
                operation = cref.integer_reference(parameters, exponent)
                for fused in (False, True):
                    for basis in (*range(4), None):
                        preparation, initial = self.inputs(2, basis)
                        sequence = (adjoint if exponent < 0 else forward) * abs(exponent)
                        if fused:
                            sequence = ['BEGIN FUSION', *sequence, 'END FUSION']
                        with self.subTest(parameters=parameters, exponent=exponent, fused=fused, basis=basis):
                            self.check(['QUBITS 2', *preparation, *sequence], self.apply(operation, initial))

    def test_ordinary_inverse_and_integer_powers_with_literal_and_runtime_parameters(self):
        maximum = float.fromhex('0x1.fffffffffffffp+1023')
        profiles = (LOSS, OVERFLOW, tuple(-value for value in LOSS), tuple(-value for value in OVERFLOW),
                    (maximum, maximum, -maximum, maximum))
        forms = (('', '', 1, False), ('inv @ ', '', 1, True), ('pow(0) @ ', '', 0, False),
                 ('pow(2) @ ', '', 2, False), ('pow(-2) @ ', '', -2, False),
                 ('pow(n) @ ', 'int n = 2; ', 2, False),
                 ('pow(n) @ ', 'int n = -2; ', -2, False),
                 ('inv @ pow(n) @ ', 'uint n = 2; ', 2, True),
                 ('pow(n) @ ', 'int n = 0; ', 0, False),
                 ('inv @ pow(-2) @ pow(n) @ ', 'int n = -1; ', -2, False))
        for parameters in profiles:
            for prefix, count, exponent, inverse in forms:
                operation = cref.integer_reference(parameters, exponent, inverse)
                for runtime in (False, True):
                    declarations = ''.join(f'float p{index} = {value!r}; ' for index, value in enumerate(parameters))
                    arguments = ', '.join(f'p{index}' if runtime else repr(value)
                                          for index, value in enumerate(parameters))
                    for fused in (False, True):
                        for basis in (*range(4), None):
                            _, initial = self.inputs(2, basis)
                            preparation = ('h q; s q[0]; ' if basis is None else
                                ''.join(f'x q[{wire}]; ' for wire in range(2) if basis & (1 << wire)))
                            with self.subTest(parameters=parameters, prefix=prefix, count=count,
                                              runtime=runtime, fused=fused, basis=basis):
                                self.check_source('qubit[2] q; ' + declarations + count + preparation
                                    + prefix + f'cu({arguments}) q[1], q[0];', self.apply(operation, initial),
                                    fused=fused, preserved=tuple((f'p{index}', value) for index, value in enumerate(parameters)))

    def test_added_mixed_controls_condition_every_factor_and_keep_intrinsic_control_positive(self):
        patterns = (('negctrl @ ', (0,)), ('ctrl @ ', (1,)),
                    ('negctrl @ ctrl @ ', (0, 1)), ('ctrl @ negctrl @ ', (1, 0)),
                    ('negctrl @ ctrl(2) @ ', (0, 1, 1)))
        for parameters in (LOSS, OVERFLOW):
            arguments = ', '.join(f'p{index}' for index in range(4))
            declarations = ''.join(f'float p{index} = {value!r}; ' for index, value in enumerate(parameters))
            for controls, values in patterns:
                qubits = len(values) + 2
                operands = ', '.join(f'q[{wire}]' for wire in reversed(range(qubits)))
                operation = cref.integer_reference(parameters, -2, True, values)
                for fused in (False, True):
                    for basis in (*range(1 << qubits), None):
                        _, initial = self.inputs(qubits, basis)
                        preparation = ('h q; s q[0]; ' if basis is None else
                            ''.join(f'x q[{wire}]; ' for wire in range(qubits) if basis & (1 << wire)))
                        with self.subTest(parameters=parameters, values=values, fused=fused, basis=basis):
                            self.check_source(f'qubit[{qubits}] q; int n = -2; ' + declarations + preparation
                                + controls + f'pow(n) @ inv @ cu({arguments}) {operands};',
                                self.apply(operation, initial), fused=fused,
                                preserved=tuple((f'p{index}', value) for index, value in enumerate(parameters)) + (('n', -2),))

    def test_indexed_arrays_mutation_aliasing_and_reused_temporaries(self):
        _, state = self.inputs(2, None)
        for parameters, exponent, inverse in ((LOSS, -2, False), (OVERFLOW, 1, True), (OVERFLOW, 2, False)):
            state = apply_cu(state, parameters, exponent, inverse)
        for fused in (False, True):
            self.check_source('qubit[2] q; h q; s q[0]; int index = 0; int n = -2; '
                'array[float, 4] values = {0.7, 0.4, -0.2, 1e20}; '
                'pow(n) @ cu(values[index], values[index+1], values[index+2], values[index+3]) q[1], q[0]; '
                'values[0] = 1.5e308; values[3] = 1.5e308; '
                'inv @ cu(values[index], values[index+1], values[index+2], values[index+3]) q[1], q[0]; '
                'pow(2) @ cu(values[index], values[index+1], values[index+2], values[index]) q[1], q[0]; '
                'float saved0 = values[0]; float saved1 = values[1]; '
                'float saved2 = values[2]; float saved3 = values[3];', state, fused=fused,
                preserved=tuple((f'saved{index}', value) for index, value in enumerate(OVERFLOW)) + (('index', 0), ('n', -2)))

    def test_noncontiguous_broadcasts_with_shared_negative_control(self):
        for parameters in (LOSS, OVERFLOW):
            _, state = self.inputs(5, None)
            for control, target in ((3, 1), (2, 0)):
                state = apply_cu(state, parameters, -2, target=target, controls=((4, 0), (control, 1)))
            declarations = ''.join(f'float p{index} = {value!r}; ' for index, value in enumerate(parameters))
            for fused in (False, True):
                self.check_source('qubit[5] q; h q; s q[0]; int n = -2; ' + declarations
                    + 'pow(n) @ negctrl @ cu(p0, p1, p2, p3) q[4], q[3:-1:2], q[1:-1:0];',
                    state, fused=fused, preserved=tuple((f'p{index}', value) for index, value in enumerate(parameters)))

    def test_runtime_for_and_while_reentry_with_continue_and_break(self):
        _, initial = self.inputs(2, None)
        # Keep phi/lambda fixed: only theta and gamma change between entries.
        negative = (-OVERFLOW[0], OVERFLOW[1], OVERFLOW[2], -OVERFLOW[3])
        for_state = apply_cu(apply_cu(initial, LOSS, -1), negative, 1)
        while_state = apply_cu(apply_cu(initial, LOSS, -2), OVERFLOW, 2)
        for fused in (False, True):
            self.check_source('qubit[2] q; h q; s q[0]; int first = 1; int last = 3; '
                'float theta = 0.7; float gamma = 1e20; '
                'for int i in [first:last] { if (i == 2) { continue; } '
                'pow(i-2) @ cu(theta, 0.4, -0.2, gamma) q[1], q[0]; theta = -1.5e308; gamma = -1.5e308; }',
                for_state, fused=fused, preserved=(('theta', -1.5e308), ('gamma', -1.5e308)))
            self.check_source('qubit[2] q; h q; s q[0]; int count = 0; int n = -2; '
                'float theta = 0.7; float gamma = 1e20; '
                'while (count < 3) { count += 1; if (count == 2) { continue; } '
                'pow(n) @ cu(theta, 0.4, -0.2, gamma) q[1], q[0]; n = -n; '
                'theta = 1.5e308; gamma = 1.5e308; if (count == 3) { break; } }',
                while_state, fused=fused, preserved=(('theta', 1.5e308), ('gamma', 1.5e308), ('n', -2), ('count', 3)))

    def test_skipped_classical_paths_do_not_evaluate_parameter_or_exponent_checks(self):
        _, initial = self.inputs(2, None)
        calls = ('cu(values[index], 0.4, -0.2, 1e20) q[1], q[0];',
                 'cu(float(1 / zero), 0.4, -0.2, 1e20) q[1], q[0];',
                 'pow(1 / zero) @ cu(0.7, 0.4, -0.2, 1e20) q[1], q[0];')
        for call in calls:
            for wrapper in ('if (flag) {{ {} }}', 'while (flag) {{ {} }}',
                            'for int i in [first:last] {{ {} }}', 'for int i in [1:0] {{ {} }}'):
                self.check_source('qubit[2] q; h q; s q[0]; bool flag = false; '
                    'int first = 2; int last = 1; int zero = 0; int index = 2; array[float, 1] values = {0.7}; '
                    + wrapper.format(call), initial)

    def test_signed_and_unsigned_parameter_promotions(self):
        minimum = self.module.QASM2QCXConverter.QCX_INT_MIN
        maximum = self.module.QASM2QCXConverter.QCX_UINT_MAX
        cases = (('int theta = ' + str(minimum) + '; uint gamma = uint(-1); ', minimum, maximum),
                 ('uint theta = uint(-1); int gamma = ' + str(minimum) + '; ', maximum, minimum))
        _, initial = self.inputs(2, None)
        for declarations, theta, gamma in cases:
            for fused in (False, True):
                self.check_source('qubit[2] q; h q; s q[0]; ' + declarations
                    + 'inv @ cu(theta, 0.4, -0.2, gamma) q[1], q[0];',
                    apply_cu(initial, (theta, 0.4, -0.2, gamma), inverse=True), fused=fused,
                    preserved=(('theta', theta), ('gamma', gamma)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    options = parser.parse_args()
    if options.bra is not None:
        BackendTests.bra = options.bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if BackendTests.checked:
        print(f'Separate-factor cu circuits checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
