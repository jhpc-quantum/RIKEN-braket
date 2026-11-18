#!/usr/bin/env python3
"""Conditional bound connection for exactly diagonal one-qubit gates.

This restricted experiment emits decimal angle strings, not converter output.
It certifies exact-real parameter serialization only; bra's parsing and runtime
arithmetic remain outside its bound.
"""

from dataclasses import dataclass
from fractions import Fraction

import qasm2qcx_certified_phase_prototype as phase
import qasm2qcx_exact_single_qubit_prototype as exact
import qasm2qcx_exact_spectral_bridge_prototype as bridge
import qasm2qcx_spectral_error_bound_prototype as error
import qasm2qcx_spectral_target_prototype as target


@dataclass(frozen=True)
class DiagonalCertificate:
    angles: tuple[str, str]
    branches: tuple[str, str]
    assessment: error.ConditionalBound


def _decimal_nearest_even(value, digits):
    """Return a decimal string and exact value rounded to fixed places."""
    scale = 10**digits
    numerator = value.numerator * scale
    sign = '-' if numerator < 0 else ''
    quotient, remainder = divmod(abs(numerator), value.denominator)
    if 2 * remainder > value.denominator or (2 * remainder == value.denominator and quotient % 2):
        quotient += 1
    if not quotient:
        sign = ''
    whole, fractional = divmod(quotient, scale)
    text = f'{sign}{whole}.{fractional:0{digits}d}'
    return text, Fraction(-quotient if sign else quotient, scale)


def certify_diagonal_power(matrix, exponent, request, *, phase_bits=48, decimal_digits=24,
                           max_precision=256, max_terms=512, max_input_bits=4096):
    """Bound a single power using the fixed computational eigenbasis.

    The caller supplies a Fraction exponent, interpreted as the exact ideal
    exponent. The result checks its conditional error bound against request;
    it does not verify a bra binary or certify native decimal parsing.
    """
    if not isinstance(request, target.AccuracyRequest) or not isinstance(exponent, Fraction):
        raise TypeError('an exact Fraction exponent and accuracy request are required')
    if (type(phase_bits) is not int or not 1 <= phase_bits <= 128
            or type(decimal_digits) is not int or not 1 <= decimal_digits <= 200
            or type(max_precision) is not int or not 8 <= max_precision <= 1024
            or type(max_terms) is not int or not 1 <= max_terms <= 4096
            or type(max_input_bits) is not int or not 1 <= max_input_bits <= 4096):
        raise ValueError('invalid diagonal-certification budget')
    if max(abs(exponent.numerator).bit_length(), exponent.denominator.bit_length()) > max_input_bits:
        raise bridge.CertificationLimit('exponent exceeds coefficient budget')
    bits, denominator = exact.representation_size(matrix)
    if max(bits, denominator) > max_input_bits:
        raise bridge.CertificationLimit('exact input exceeds coefficient budget')
    bridge.validate_unitary(matrix)
    if matrix[0][1] != exact.ZERO or matrix[1][0] != exact.ZERO:
        raise ValueError('prototype supports only exactly diagonal matrices')

    precision = min(max_precision, max(8, phase_bits + 16))
    width = Fraction(1, 1 << phase_bits)
    while True:
        try:
            roots = tuple(bridge._exact_eigenvalue(matrix[i][i], precision) for i in range(2))
            pi = phase._pi_bounds(precision + 8, max_terms)
            intervals = tuple(phase._phase_bounds(root, precision + 8, max_terms, pi) for root in roots)
            if all(hi - lo <= width for lo, hi in intervals):
                break
        except bridge.CertificationLimit as problem:
            if 'term budget' in str(problem):
                raise
        if precision == max_precision:
            raise bridge.CertificationLimit('diagonal phase enclosure unresolved within precision budget')
        precision = min(precision * 2, max_precision)

    angles, assumptions = [], []
    for index, (root, (lower, upper)) in enumerate(zip(roots, intervals)):
        midpoint = (lower + upper) / 2
        text, serialized = _decimal_nearest_even(exponent * midpoint, decimal_digits)
        angles.append(text)
        assumptions.append(error.PhaseErrorAssumption(
            f'q{index}', root.branch, (upper - lower) / 2, abs(serialized - exponent * midpoint)))
    inputs = error.SpectralErrorAssumptions(tuple(assumptions), ('q0', 'q1'), exponent,
                                            Fraction(0), Fraction(0), Fraction(0))
    assessment = error.assess_bound(request, inputs)
    if not assessment.meets_request:
        raise bridge.CertificationLimit('operator bound exceeds requested error')
    return DiagonalCertificate(tuple(angles), tuple(root.branch for root in roots), assessment)
