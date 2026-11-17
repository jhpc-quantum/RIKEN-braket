#!/usr/bin/env python3
"""Compare the composition prototype with independent direct matrices."""

import itertools
import math
import unittest

from qasm2qcx_single_qubit_composition_prototype import as_matrix, compose_sequence
import test_qasm2qcx_single_qubit_power_reference as reference


class SingleQubitCompositionPrototypeTests(unittest.TestCase):
    assertMatrixClose = reference.SingleQubitPowerReferenceTests.assertMatrixClose
    def test_fixed_gate_sequences_match_direct_matrix_products(self):
        for length in range(4):
            for names in itertools.product(reference.GATES, repeat=length):
                with self.subTest(gates=names):
                    self.assertMatrixClose(as_matrix(compose_sequence(names)), reference.sequence_matrix(names))

    def test_noncommuting_sequences_keep_order(self):
        forward = as_matrix(compose_sequence(('h', 's')))
        reverse = as_matrix(compose_sequence(('s', 'h')))
        self.assertMatrixClose(forward, reference.sequence_matrix(('h', 's')))
        self.assertMatrixClose(reverse, reference.sequence_matrix(('s', 'h')))
        self.assertNotEqual(forward, reverse)

    def test_scalar_phase_is_not_discarded(self):
        result = compose_sequence(('z', 'y', 'x'))
        self.assertEqual(result.vector, (0.0, 0.0, 0.0))
        self.assertMatrixClose(as_matrix(result), ((1j, 0), (0, 1j)))

    def test_repeated_gates_are_not_normalized_or_snapped(self):
        result = compose_sequence(('h', 'h'))
        self.assertEqual(result.vector, (0.0, 0.0, 0.0))
        self.assertEqual(result.scalar, -2 * math.sqrt(0.5) ** 2)
        self.assertMatrixClose(as_matrix(result), reference.sequence_matrix(('h', 'h')))


if __name__ == '__main__':
    unittest.main()
