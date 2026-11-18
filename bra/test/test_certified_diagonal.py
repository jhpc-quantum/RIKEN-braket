#!/usr/bin/env python3
"""Tests for the maintained, currently unconnected diagonal foundation."""

import cmath
from fractions import Fraction
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

import test_qasm2qcx_single_qubit_power_reference as reference


MODULE_PATH = Path(__file__).resolve().parents[1] / 'qcx/certified_diagonal.py'
SPEC = importlib.util.spec_from_file_location('certified_diagonal_test_module', MODULE_PATH)
foundation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = foundation
SPEC.loader.exec_module(foundation)


def as_complex(value):
    from math import sqrt
    return complex(value.a + value.b * sqrt(2), value.c + value.d * sqrt(2)) / 2**value.k


class BoundedCompositionTests(unittest.TestCase):
    def test_exact_identities_scalar_phase_and_classification(self):
        for names in ((), ('h', 'h'), ('x', 'x'), ('s', 'sdg'), ('t',) * 8):
            with self.subTest(sequence=names):
                budget = foundation.Budget()
                matrix = foundation.compose_fixed_gates(names, budget)
                self.assertEqual(matrix, foundation.FIXED_GATES['id'])
                self.assertEqual(foundation.classify_matrix(matrix, budget), 'scalar')
        budget = foundation.Budget()
        self.assertEqual(foundation.compose_fixed_gates(('t',) * 4, budget), foundation.FIXED_GATES['z'])
        scalar = foundation.compose_fixed_gates(('z', 'y', 'x'), budget)
        self.assertEqual(scalar, ((foundation.I, foundation.ZERO), (foundation.ZERO, foundation.I)))
        self.assertEqual(foundation.classify_matrix(scalar, budget), 'scalar')
        self.assertEqual(foundation.classify_matrix(foundation.FIXED_GATES['z'], budget), 'diagonal')
        self.assertEqual(foundation.classify_matrix(foundation.FIXED_GATES['h'], budget), 'non-diagonal')

    def test_all_sequences_up_to_three_match_independent_matrices(self):
        for size in range(4):
            for names in itertools.product(reference.GATES, repeat=size):
                actual = foundation.compose_fixed_gates(names, foundation.Budget())
                expected = reference.sequence_matrix(names)
                for row, column in itertools.product(range(2), repeat=2):
                    with self.subTest(gates=names, row=row, column=column):
                        self.assertTrue(cmath.isclose(as_complex(actual[row][column]), expected[row][column],
                                                      rel_tol=1e-14, abs_tol=1e-14))

    def test_noncommuting_order_and_bad_inputs(self):
        hs = foundation.compose_fixed_gates(('h', 's'), foundation.Budget())
        sh = foundation.compose_fixed_gates(('s', 'h'), foundation.Budget())
        self.assertNotEqual(hs, sh)
        with self.assertRaises(foundation.UnsupportedGate):
            foundation.compose_fixed_gates(('h', 'bad'), foundation.Budget())
        with self.assertRaises(foundation.InvalidInput):
            foundation.compose_fixed_gates(('h',), None)
        with self.assertRaises(foundation.InvalidInput):
            foundation.classify_matrix(((foundation.Exact(a=2, k=1), foundation.ZERO),
                                        (foundation.ZERO, foundation.ONE)), foundation.Budget())


class SharedBudgetTests(unittest.TestCase):
    def test_gate_and_nesting_budgets_are_shared_and_monotonic(self):
        budget = foundation.Budget(foundation.BudgetLimits(gate_steps=1))
        foundation.compose_fixed_gates(('x',), budget)
        work_before = budget.work
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'gate steps'):
            foundation.compose_fixed_gates(('z',), budget)
        self.assertEqual(budget.gate_steps, 1)
        self.assertGreater(budget.work, work_before)
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'nesting'):
            foundation.Budget(foundation.BudgetLimits(nesting=1)).charge_gate(depth=2)

    def test_work_exhaustion_does_not_return_a_partial_matrix(self):
        budget = foundation.Budget(foundation.BudgetLimits(work=10))
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'work'):
            foundation.compose_fixed_gates(('h',), budget)
        self.assertEqual(budget.work, 10)
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'work'):
            foundation.compose_fixed_gates(('x',), budget)

    def test_checked_integer_growth_and_exact_limits(self):
        budget = foundation.Budget(foundation.BudgetLimits(integer_bits=4))
        self.assertEqual(foundation._checked_shift(1, 3, budget), 8)
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'intermediate integer'):
            foundation._checked_shift(1, 4, budget)
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'intermediate integer'):
            foundation._checked_multiply(8, 8, budget)
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'exact coefficient'):
            foundation.classify_matrix(((foundation.Exact(a=2), foundation.ZERO),
                                        (foundation.ZERO, foundation.ONE)),
                                       foundation.Budget(foundation.BudgetLimits(exact_bits=1)))
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'exact coefficient'):
            foundation.compose_fixed_gates((), foundation.Budget(foundation.BudgetLimits(exact_bits=0)))
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'denominator exponent'):
            foundation.compose_fixed_gates(('h',), foundation.Budget(
                foundation.BudgetLimits(denominator_exponent=0)))

    def test_output_reservations_are_cumulative_and_atomic(self):
        budget = foundation.Budget(foundation.BudgetLimits(output_chars=4, output_lines=2))
        budget.reserve_output(chars=2, lines=1)
        budget.reserve_output(chars=2, lines=1)
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'output characters'):
            budget.reserve_output(chars=1)
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'output lines'):
            budget.reserve_output(lines=1)
        self.assertEqual((budget.output_chars, budget.output_lines), (4, 2))


class BoundedPhaseAndSerializationTests(unittest.TestCase):
    PI_LOW = Fraction('3.14159265358979323846264338327950288419716939937510')
    PI_HIGH = PI_LOW + Fraction(1, 10**50)

    def test_phase_bounds_match_independent_exact_pi_intervals(self):
        cases = ((foundation.FIXED_GATES['id'], (Fraction(0), Fraction(0))),
                 (foundation.FIXED_GATES['z'], (Fraction(0), Fraction(1))),
                 (foundation.FIXED_GATES['t'], (Fraction(0), Fraction(1, 4))),
                 (foundation.FIXED_GATES['tdg'], (Fraction(0), Fraction(-1, 4))),
                 (((foundation.I, foundation.ZERO), (foundation.ZERO, foundation.I)),
                  (Fraction(1, 2), Fraction(1, 2))))
        for matrix, coefficients in cases:
            for exponent in (Fraction(0), Fraction(1, 2), Fraction(-1, 2), Fraction(2), Fraction(10**8)):
                with self.subTest(matrix=matrix, exponent=exponent):
                    result = foundation.certify_diagonal_power(
                        matrix, exponent, Fraction(1, 10**12), foundation.Budget(),
                        phase_bits=120, decimal_places=40)
                    self.assertLessEqual(result.error_bound, Fraction(1, 10**12))
                    for angle, coefficient in zip(result.angles, coefficients):
                        limits = (coefficient * exponent * self.PI_LOW, coefficient * exponent * self.PI_HIGH)
                        self.assertLessEqual(max(abs(Fraction(angle) - limit) for limit in limits),
                                             result.error_bound)

    def test_matches_previous_diagonal_prototype(self):
        import qasm2qcx_certified_diagonal_prototype as previous
        import qasm2qcx_exact_single_qubit_prototype as old_exact
        import qasm2qcx_spectral_target_prototype as target
        request = target.AccuracyRequest(target.BINARY64, Fraction(1, 10**12))
        for name in ('id', 'z', 's', 'sdg', 't', 'tdg'):
            with self.subTest(gate=name):
                actual = foundation.certify_diagonal_power(
                    foundation.FIXED_GATES[name], Fraction(1, 2), Fraction(1, 10**12), foundation.Budget())
                expected = previous.certify_diagonal_power(old_exact.GATES[name], Fraction(1, 2), request)
                self.assertEqual(actual.angles, expected.angles)
                self.assertEqual(actual.branches, expected.branches)
                self.assertEqual(actual.error_bound, expected.assessment.total)

    def test_exact_decimal_ties_and_negative_zero(self):
        for value, expected in ((Fraction(1, 8), '0.12'), (Fraction(27, 200), '0.14'),
                                (Fraction(-1, 8), '-0.12'), (Fraction(-27, 200), '-0.14'),
                                (Fraction(-1, 1000), '0.00')):
            with self.subTest(value=value):
                text, serialized = foundation._serialize_decimal(value, 2, foundation.Budget())
                self.assertEqual(text, expected)
                self.assertEqual(serialized, Fraction(expected))

    def test_synthetic_near_cut_phases_are_not_snapped(self):
        # The exact complex inputs exercise the phase primitive; their unit
        # modulus and eligibility as fixed-gate eigenvalues are not assumed.
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                budget = foundation.Budget()
                pi = foundation._pi_interval(208, budget)
                root2 = foundation._sqrt2_interval(200, budget)
                value = foundation.Exact(a=-(2**120), c=sign, k=120)
                branch, (lower, upper) = foundation._principal_phase(value, 208, root2, pi, budget)
                if sign > 0:
                    self.assertEqual(branch, 'above')
                    self.assertLess(upper, pi[0])
                else:
                    self.assertEqual(branch, 'below')
                    self.assertGreater(lower, -pi[0])

    def test_shared_work_and_output_are_charged_across_both_phases(self):
        budget = foundation.Budget(foundation.BudgetLimits(output_chars=30))
        with self.assertRaisesRegex(foundation.BudgetExceeded, 'output characters'):
            foundation.certify_diagonal_power(foundation.FIXED_GATES['z'], Fraction(1, 2),
                                             Fraction(1, 10**12), budget)
        self.assertGreater(budget.output_chars, 0)
        self.assertLessEqual(budget.output_chars, 30)
        self.assertGreater(budget.work, 0)

    def test_precision_term_size_and_accuracy_failures_are_explicit(self):
        cases = ((foundation.BudgetLimits(precision_bits=40), {}, 'precision bits'),
                 (foundation.BudgetLimits(series_terms=1), {}, 'series terms'),
                 (foundation.BudgetLimits(rational_bits=8), {}, 'rational bits'),
                 (foundation.BudgetLimits(integer_bits=16), {}, 'intermediate integer bits'),
                 (foundation.BudgetLimits(input_bits=8), {}, 'input bits'),
                 (foundation.BudgetLimits(decimal_places=2), {}, 'decimal places'),
                 (foundation.BudgetLimits(work=10), {}, 'work'))
        for limits, kwargs, category in cases:
            with self.subTest(category=category), self.assertRaisesRegex(foundation.BudgetExceeded, category):
                foundation.certify_diagonal_power(foundation.FIXED_GATES['z'], Fraction(1, 2),
                                                 Fraction(1, 10**12), foundation.Budget(limits), **kwargs)
        with self.assertRaises(foundation.AccuracyNotMet):
            foundation.certify_diagonal_power(foundation.FIXED_GATES['z'], Fraction(1, 2),
                                             Fraction(1, 10**30), foundation.Budget(), decimal_places=2)

    def test_rejects_nonunitary_nondiagonal_and_invalid_goal(self):
        with self.assertRaisesRegex(foundation.InvalidInput, 'diagonal'):
            foundation.certify_diagonal_power(foundation.FIXED_GATES['h'], Fraction(1, 2),
                                             Fraction(1, 100), foundation.Budget())
        nonunitary = ((foundation.Exact(2), foundation.ZERO), (foundation.ZERO, foundation.ONE))
        with self.assertRaisesRegex(foundation.InvalidInput, 'unitary'):
            foundation.certify_diagonal_power(nonunitary, Fraction(1, 2), Fraction(1, 100), foundation.Budget())
        with self.assertRaisesRegex(foundation.InvalidInput, 'strictly between'):
            foundation.certify_diagonal_power(foundation.FIXED_GATES['z'], Fraction(1, 2),
                                             Fraction(0), foundation.Budget())


if __name__ == '__main__':
    unittest.main()
