"""Stage 1 exact-reference checks; no converter or bra executable required."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import itertools
from pathlib import Path
import struct
import unittest

from anglebits2phase_reference import (
    ChainSyntaxError, DEFAULT_LIMITS, Limits, Modifier, PhaseDomainError,
    PhaseResourceError, decode_chain, decode_exponent, evaluate_turns, principal_turn)


class DescriptorTests(unittest.TestCase):
    def test_empty_chain_and_case_insensitive_evaluation_order(self):
        self.assertEqual(decode_chain([]), ())
        self.assertEqual(decode_chain(['inv', 'PoW', '0.5']),
                         (Modifier('INV'), Modifier('POW', Fraction(1, 2))))
        self.assertNotEqual(decode_chain(['INV', 'POW', '0.5']), decode_chain(['POW', '0.5', 'INV']))

    def test_integer_literals_are_exact_even_above_native_and_binary64_ranges(self):
        for value in (0, -3, (1 << 100) + 1, -(1 << 4096) + 1):
            self.assertEqual(decode_exponent(str(value)), value)
        self.assertEqual(decode_exponent('+0003'), 3)
        self.assertEqual(decode_exponent('9007199254740993'), 9007199254740993)
        self.assertEqual(decode_exponent('9007199254740993.0'), 9007199254740992)

    def test_floating_literals_preserve_stored_binary64_not_decimal_rationals(self):
        self.assertEqual(decode_exponent('0.1'), Fraction(3602879701896397, 36028797018963968))
        self.assertNotEqual(decode_exponent('0.1'), Fraction(1, 10))
        for token, expected in (('.5', Fraction(1, 2)), ('-2.', -2), ('+2.0E-1', Fraction.from_float(0.2)),
                                ('2e0', 2), ('-0.0', 0), ('1e-4000', 0)):
            with self.subTest(token=token):
                self.assertEqual(decode_exponent(token), expected)

    def test_binary64_boundary_and_tie_decoding(self):
        self.assertEqual(decode_exponent('5e-324'), Fraction(1, 1 << 1074))
        self.assertEqual(decode_exponent('1.7976931348623157e308'), Fraction.from_float(float.fromhex('0x1.fffffffffffffp1023')))
        self.assertEqual(decode_exponent('1.00000000000000011102230246251565404236316680908203125'), 1)
        self.assertEqual(decode_exponent('1.00000000000000033306690738754696212708950042724609375'),
                         Fraction.from_float(1.0000000000000004))

    def test_nonfinite_factors_are_rejected_even_after_zero(self):
        for token in ('inf', '-inf', '+infinity', 'NaN', '-nan', '1e309'):
            with self.subTest(token=token), self.assertRaises(PhaseDomainError):
                decode_chain(['POW', '0', 'POW', token])

    def test_malformed_and_nonliteral_chains_are_rejected(self):
        for tokens in (['POW'], ['INV', '0.5'], ['CTRL'], ['0.5'], ['POW', 'value'],
                       ['POW', ':REAL:value'], ['POW', 'pi'], ['POW', '1/2'],
                       ['POW', '0x1p-1'], ['POW', '1_000'], ['POW', '1.0junk'], ['POW', True]):
            with self.subTest(tokens=tokens), self.assertRaises(ChainSyntaxError):
                decode_chain(tokens)

    def test_descriptors_and_limits_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            Modifier('INV').kind = 'POW'
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_LIMITS.max_modifiers = 2
        for values in ({'uint_width': 0}, {'max_modifiers': -1}, {'max_literal_chars': True},
                       {'max_exact_bits': 0}):
            with self.assertRaises(ValueError):
                Limits(**values)


class ExactTurnTests(unittest.TestCase):
    def test_principal_interval_and_both_half_turn_representatives(self):
        for value, expected in ((0, 0), (1, 0), (-1, 0), (Fraction(1, 2), Fraction(1, 2)),
                                (Fraction(-1, 2), Fraction(1, 2)), (Fraction(3, 2), Fraction(1, 2)),
                                (Fraction(3, 4), Fraction(-1, 4)), (Fraction(-3, 4), Fraction(1, 4))):
            self.assertEqual(principal_turn(Fraction(value)), expected)

    def test_every_width_and_initial_pattern_has_exact_principal_turns(self):
        for width in range(1, DEFAULT_LIMITS.uint_width + 1):
            half = 1 << (width - 1)
            for bits in sorted({0, 1, half - 1, half, min(half + 1, (1 << width) - 1), (1 << width) - 1}):
                signed = bits if bits <= half else bits - (1 << width)
                self.assertEqual(evaluate_turns(bits, width), Fraction(signed, 1 << width))

    def test_integer_branch_decisions_survive_float32_collisions(self):
        width = DEFAULT_LIMITS.uint_width
        half = 1 << (width - 1)
        if width <= 25:
            self.skipTest('native UINT has no bits below binary32 half-turn spacing')
        values = [evaluate_turns(bits, width) for bits in (half - 1, half, half + 1)]
        self.assertTrue(values[0] > 0 and values[1] == Fraction(1, 2) and values[2] < 0)
        rounded = [struct.unpack('f', struct.pack('f', float(Fraction(bits, 1 << width))))[0]
                   for bits in (half - 1, half, half + 1)]
        self.assertEqual(rounded, [0.5, 0.5, 0.5])

    def test_endpoint_examples_distinguish_inverse_and_power_order(self):
        for tokens, expected in ((['POW', '0.5'], Fraction(1, 4)),
                                  (['POW', '0.5', 'INV'], Fraction(-1, 4)),
                                  (['INV', 'POW', '0.5'], Fraction(1, 4)),
                                  (['POW', '2', 'POW', '0.5'], 0)):
            self.assertEqual(evaluate_turns(8, 4, decode_chain(tokens)), expected)

    def test_nested_powers_reselect_intermediate_branches(self):
        chain = decode_chain(['POW', '2', 'POW', '0.5'])
        self.assertEqual(evaluate_turns(6, 3, chain), Fraction(1, 4))
        self.assertEqual(evaluate_turns(6, 3), Fraction(-1, 4))
        # An intermediate inverse of an exact negative-real eigenvalue stays
        # exactly on the cut; EXP's tiny imaginary error must not choose it.
        chain = decode_chain(['POW', '4.0', 'INV', 'POW', '0.5'])
        self.assertEqual(evaluate_turns(3, 3, chain), Fraction(1, 4))

    def test_results_can_need_more_bits_than_the_source_angle(self):
        self.assertEqual(evaluate_turns(1, 1, decode_chain(['POW', '0.5'])), Fraction(1, 4))
        chain = decode_chain(['POW', '0.5'] * 100)
        self.assertEqual(evaluate_turns(1, 1, chain), Fraction(1, 1 << 101))

    def test_no_intermediate_float_underflow_or_overflow(self):
        smallest = decode_exponent('5e-324')
        huge = decode_exponent('1.7976931348623157e308')
        chain = decode_chain(['POW', '5e-324', 'POW', '1.7976931348623157e308'])
        self.assertEqual(evaluate_turns(1, 1, chain), smallest * huge / 2)
        self.assertNotEqual(evaluate_turns(1, 1, chain), 0)
        # Reducing a finite huge power can give exact identity without a
        # floating-point intermediate product or its overflow check.
        self.assertEqual(evaluate_turns(1, 1, decode_chain(['POW', '1.7976931348623157e308'])), 0)

    def test_small_angles_exhaustively_match_independent_integer_recurrence(self):
        chains = [(), decode_chain(['INV'])]
        chains += [decode_chain(['POW', token]) for token in ('0', '1', '-1', '2', '0.5', '1.5', '-0.5')]
        chains += [decode_chain(list(tokens)) for tokens in
                   (('INV', 'POW', '0.5'), ('POW', '0.5', 'INV'), ('POW', '2', 'POW', '0.5'))]
        for width in range(1, 7):
            for bits, chain in itertools.product(range(1 << width), chains):
                # Independent dyadic integer recurrence, not Fraction modulo
                # or the principal_turn helper used by the reference model.
                numerator, precision = bits, width
                for modifier in chain:
                    if modifier.kind == 'INV':
                        numerator = -numerator
                    else:
                        modulus = 1 << precision
                        numerator %= modulus
                        if numerator > modulus // 2:
                            numerator -= modulus
                        numerator *= modifier.exponent.numerator
                        precision += modifier.exponent.denominator.bit_length() - 1
                modulus = 1 << precision
                numerator %= modulus
                if numerator > modulus // 2:
                    numerator -= modulus
                self.assertEqual(evaluate_turns(bits, width, chain), Fraction(numerator, modulus))

    def test_runtime_inputs_are_validated_even_for_a_zero_power(self):
        chain = decode_chain(['POW', '0.0'])
        for bits, width in ((0, 0), (0, DEFAULT_LIMITS.uint_width + 1), (-1, 4), (16, 4), (True, 4), (0, True)):
            with self.subTest(bits=bits, width=width), self.assertRaises(PhaseDomainError):
                evaluate_turns(bits, width, chain)

    def test_no_tolerance_window_and_no_negative_zero_in_exact_results(self):
        for width in (4, DEFAULT_LIMITS.uint_width):
            half = 1 << (width - 1)
            roots = [evaluate_turns(bits, width, decode_chain(['POW', '0.5']))
                     for bits in (half - 1, half, half + 1)]
            self.assertTrue(roots[0] > 0 and roots[1] == Fraction(1, 4) and roots[2] < 0)
        for tokens in (['POW', '-0.0'], ['INV', 'POW', '0'], ['POW', '2', 'INV']):
            result = evaluate_turns(1, 1, decode_chain(tokens))
            self.assertEqual((result.numerator, result.denominator), (0, 1))


class ResourceTests(unittest.TestCase):
    def test_modifier_limit_has_a_precise_inclusive_boundary(self):
        limits = replace(DEFAULT_LIMITS, max_modifiers=3)
        self.assertEqual(len(decode_chain(['INV'] * 3, limits)), 3)
        with self.assertRaises(PhaseResourceError):
            decode_chain(['INV'] * 4, limits)

    def test_literal_length_is_checked_before_parsing(self):
        limits = replace(DEFAULT_LIMITS, max_literal_chars=3)
        self.assertEqual(decode_exponent('123', limits), 123)
        with self.assertRaises(PhaseResourceError):
            decode_exponent('1234', limits)
        with self.assertRaises(PhaseResourceError):
            decode_exponent('x' * 10000, limits)

    def test_exact_factor_limits_cannot_be_hidden_by_zero(self):
        limits = replace(DEFAULT_LIMITS, max_exact_bits=8)
        self.assertEqual(decode_exponent('255', limits), 255)
        with self.assertRaises(PhaseResourceError):
            decode_chain(['POW', '0', 'POW', '256'], limits)
        with self.assertRaises(PhaseResourceError):
            decode_exponent('0.1', limits)

    def test_later_zero_does_not_hide_intermediate_precision_resource_errors(self):
        limits = replace(DEFAULT_LIMITS, max_exact_bits=12)
        chain = decode_chain(['POW', '0.125'] * 3 + ['POW', '0'], limits)
        with self.assertRaises(PhaseResourceError):
            evaluate_turns(1, 4, chain, limits)
        self.assertEqual(evaluate_turns(0, 4, chain, limits), 0)

    def test_metadata_is_validated_before_runtime_input_values(self):
        for chain in ((Modifier('INV', Fraction(1)),), (Modifier('POW'),),
                       (Modifier('POW', Fraction(1, 3)),), (Modifier('CTRL'),)):
            with self.subTest(chain=chain), self.assertRaises((ChainSyntaxError, PhaseDomainError)):
                evaluate_turns(0, 0, chain)

    def test_design_document_records_integration_status_and_shared_limits(self):
        document = (Path(__file__).parents[2] / 'docs' / 'anglebits2phase-design.md').read_text()
        self.assertIn('Status: implemented in bra and qasm2qcx', document)
        for limit in (DEFAULT_LIMITS.max_modifiers, DEFAULT_LIMITS.max_literal_chars, DEFAULT_LIMITS.max_exact_bits):
            self.assertIn(f'{limit:,}', document)


if __name__ == '__main__':
    unittest.main()
