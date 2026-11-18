#!/usr/bin/env python3
"""Tests for the maintained, currently unconnected diagonal foundation."""

import cmath
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


if __name__ == '__main__':
    unittest.main()
