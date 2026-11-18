#!/usr/bin/env python3
"""Rational principal-phase enclosures for the #638 feasibility study.

This experimental code is separate from the converter. Bounds use exact
rational arithmetic and alternating-series remainders, not host float math.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import qasm2qcx_exact_spectral_bridge_prototype as bridge


@dataclass(frozen=True)
class PhaseEnclosure:
    # index pairs this phase with the root in the accompanying certification
    # result; it is not yet a certified eigenvector or QCX basis label.
    index: int
    branch: str
    lower: Fraction
    upper: Fraction


def _atan_series(value, bits, max_terms):
    """Enclose atan(value) for |value| <= 1/2 by alternating remainders."""
    if value < 0:
        lo, hi = _atan_series(-value, bits, max_terms)
        return -hi, -lo
    if value > Fraction(1, 2):
        raise ValueError('atan series argument outside reduced range')
    if not value:
        return Fraction(0), Fraction(0)
    tolerance = Fraction(1, 1 << bits)
    squared = value * value
    power = value
    total = Fraction(0)
    for index in range(max_terms):
        term = power / (2 * index + 1)
        total = total + term if index % 2 == 0 else total - term
        power *= squared
        next_term = power / (2 * index + 3)
        if next_term <= tolerance:
            next_total = total - next_term if index % 2 == 0 else total + next_term
            return min(total, next_total), max(total, next_total)
    raise bridge.CertificationLimit('atan series exceeded term budget')


@lru_cache(maxsize=32)
def _pi_bounds(bits, max_terms):
    # Machin's identity; both arctangent bounds use rational remainders.
    a = _atan_series(Fraction(1, 5), bits + 8, max_terms)
    b = _atan_series(Fraction(1, 239), bits + 8, max_terms)
    return 16 * a[0] - 4 * b[1], 16 * a[1] - 4 * b[0]


def _atan(value, bits, max_terms, pi):
    if value < 0:
        lo, hi = _atan(-value, bits, max_terms, pi)
        return -hi, -lo
    if value > 1:
        lo, hi = _atan(1 / value, bits, max_terms, pi)
        return pi[0] / 2 - hi, pi[1] / 2 - lo
    if value > Fraction(1, 2):
        lo, hi = _atan_series((value - 1) / (value + 1), bits, max_terms)
        return pi[0] / 4 + lo, pi[1] / 4 + hi
    return _atan_series(value, bits, max_terms)


def _phase_bounds(root, bits, max_terms, pi):
    if root.branch == 'negative-endpoint':
        return pi
    if root.branch == 'positive-real':
        return Fraction(0), Fraction(0)
    x, y = root.real, root.imaginary
    if y[0] <= 0 <= y[1]:
        raise bridge.CertificationLimit('eigenvalue enclosure crosses the real axis')
    reciprocal = (1 / y[1], 1 / y[0])
    ratio = bridge._mul(x, reciprocal)
    lower_atan = _atan(ratio[0], bits, max_terms, pi)[0]
    upper_atan = _atan(ratio[1], bits, max_terms, pi)[1]
    if root.branch == 'above':
        return pi[0] / 2 - upper_atan, pi[1] / 2 - lower_atan
    if root.branch == 'below':
        return -pi[1] / 2 - upper_atan, -pi[0] / 2 - lower_atan
    raise ValueError('unsupported branch')


def certify_principal_phases(matrix, *, target_bits=32, max_precision=256,
                             max_terms=512, max_input_bits=4096):
    """Enclose principal eigenphases to absolute width <= 2**-target_bits.

    Refinement and series iterations are bounded. No eigenbasis, floating
    exponent, serialized angle, or backend error is certified by this function.
    """
    if (type(target_bits) is not int or not 1 <= target_bits <= 128
            or type(max_precision) is not int or not 8 <= max_precision <= 1024
            or type(max_terms) is not int or not 1 <= max_terms <= 4096
            or type(max_input_bits) is not int or not 1 <= max_input_bits <= 4096):
        raise ValueError('invalid phase-certification budget')
    precision = min(max_precision, max(8, target_bits + 16))
    target_width = Fraction(1, 1 << target_bits)
    while True:
        try:
            roots = bridge.certify_eigenvalue_branches(
                matrix, max_bits=precision, min_bits=precision, max_input_bits=max_input_bits)
            pi = _pi_bounds(precision + 8, max_terms)
            result = tuple(PhaseEnclosure(index, root.branch,
                                          *_phase_bounds(root, precision + 8, max_terms, pi))
                           for index, root in enumerate(roots))
            if all(phase.upper - phase.lower <= target_width for phase in result):
                return result
        except bridge.CertificationLimit as error:
            if 'term budget' in str(error) or 'coefficient budget' in str(error):
                raise
        if precision == max_precision:
            raise bridge.CertificationLimit('phase enclosure unresolved within precision budget')
        precision = min(precision * 2, max_precision)
