#!/usr/bin/env python3
"""Tests for explicit spectral target and accuracy descriptors."""

from fractions import Fraction
import unittest

import qasm2qcx_spectral_target_prototype as model


class SpectralTargetTests(unittest.TestCase):
    def test_binary_profiles_use_exact_rational_roundoff(self):
        self.assertEqual(model.BINARY32.unit_roundoff, Fraction(1, 2**24))
        self.assertEqual(model.BINARY64.unit_roundoff, Fraction(1, 2**53))
        self.assertEqual(model.BINARY32.smallest_positive_exponent, -149)
        self.assertEqual(model.BINARY64.smallest_positive_exponent, -1074)

    def test_extended_precision_must_be_described_explicitly(self):
        extended = model.BinaryTarget('example-extended', 64, -16382, 16383, True)
        self.assertEqual(extended.unit_roundoff, Fraction(1, 2**64))
        self.assertEqual(extended.smallest_positive_exponent, -16445)
        self.assertNotEqual(extended.unit_roundoff, model.BINARY64.unit_roundoff)
        self.assertFalse(hasattr(model, 'LONG_DOUBLE'))

    def test_subnormal_policy_is_explicit(self):
        target = model.BinaryTarget('flush-model', 24, -126, 127, False)
        self.assertEqual(target.smallest_positive_exponent, -126)

    def test_invalid_target_descriptors_are_rejected(self):
        for kwargs in ({'precision': 1}, {'precision': 4097}, {'emin': 1}, {'emax': -1},
                       {'subnormals': 1}, {'rounding': 'toward-zero'}, {'name': ''}):
            arguments = dict(name='target', precision=24, emin=-126, emax=127, subnormals=True)
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                model.BinaryTarget(**arguments)

    def test_accuracy_request_is_explicit_and_exact(self):
        request = model.AccuracyRequest(model.BINARY64, Fraction(1, 2**48))
        self.assertEqual(request.max_operator_error, Fraction(1, 2**48))
        for value in (0, 1, Fraction(-1, 10), Fraction(1), 0.1):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                model.AccuracyRequest(model.BINARY64, value)
        with self.assertRaises(TypeError):
            model.AccuracyRequest('double', Fraction(1, 10))


if __name__ == '__main__':
    unittest.main()
