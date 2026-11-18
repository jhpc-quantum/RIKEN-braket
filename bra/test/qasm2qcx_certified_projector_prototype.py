#!/usr/bin/env python3
"""Rational spectral-projector enclosures for the #638 study.

This does not produce U3 angles or QCX. For distinct roots it encloses
P_j = (G - lambda_k I) / (lambda_j - lambda_k), with root indices retained.
"""

from dataclasses import dataclass
from fractions import Fraction

import qasm2qcx_exact_single_qubit_prototype as exact
import qasm2qcx_exact_spectral_bridge_prototype as bridge


@dataclass(frozen=True)
class ScalarProjectors:
    # Scalar gates need no arbitrary eigenbasis or division by a zero gap.
    eigenvalue: bridge.CertifiedEigenvalue


@dataclass(frozen=True)
class ProjectorEnclosure:
    index: int
    eigenvalue: bridge.CertifiedEigenvalue
    # Each entry is ((real_lower, real_upper), (imag_lower, imag_upper)).
    entries: tuple


def _cadd(left, right):
    return bridge._add(left[0], right[0]), bridge._add(left[1], right[1])


def _cneg(value):
    return bridge._neg(value[0]), bridge._neg(value[1])


def _cmul(left, right):
    real = bridge._add(bridge._mul(left[0], right[0]), bridge._neg(bridge._mul(left[1], right[1])))
    imag = bridge._add(bridge._mul(left[0], right[1]), bridge._mul(left[1], right[0]))
    return real, imag


def _cdivide(numerator, denominator):
    length_squared = bridge._add(bridge._square(denominator[0]), bridge._square(denominator[1]))
    if length_squared[0] <= 0:
        raise bridge.CertificationLimit('eigenvalue gap enclosure includes zero')
    conjugate = denominator[0], bridge._neg(denominator[1])
    product = _cmul(numerator, conjugate)
    inverse = 1 / length_squared[1], 1 / length_squared[0]
    return bridge._mul(product[0], inverse), bridge._mul(product[1], inverse)


def certify_projectors(matrix, *, target_bits=32, max_precision=256, max_input_bits=4096):
    """Enclose both distinct spectral projectors to the requested width.

    Scalar gates return ScalarProjectors instead. Exhaustion rejects rather
    than merging roots; the returned enclosures do not certify phase angles.
    """
    if (type(target_bits) is not int or not 1 <= target_bits <= 128
            or type(max_precision) is not int or not 8 <= max_precision <= 1024
            or type(max_input_bits) is not int or not 1 <= max_input_bits <= 4096):
        raise ValueError('invalid projector-certification budget')
    precision = min(max_precision, max(8, target_bits + 16))
    width = Fraction(1, 1 << target_bits)
    while True:
        try:
            roots = bridge.certify_eigenvalue_branches(
                matrix, max_bits=precision, min_bits=precision, max_input_bits=max_input_bits)
            if exact.scalar_value(matrix) is not None:
                return ScalarProjectors(roots[0])
            root_values = tuple((root.real, root.imaginary) for root in roots)
            exact_entries = tuple(tuple(bridge._exact_interval(entry, precision) for entry in row)
                                  for row in matrix)
            projectors = []
            for index in range(2):
                own, other = root_values[index], root_values[1 - index]
                gap = _cadd(own, _cneg(other))
                entries = tuple(tuple(_cdivide(
                    _cadd(exact_entries[row][column], _cneg(other))
                    if row == column else exact_entries[row][column], gap)
                    for column in range(2)) for row in range(2))
                projectors.append(ProjectorEnclosure(index, roots[index], entries))
            if all(component[1] - component[0] <= width
                   for projector in projectors for row in projector.entries
                   for entry in row for component in entry):
                return tuple(projectors)
        except bridge.CertificationLimit as error:
            if 'coefficient budget' in str(error):
                raise
        if precision == max_precision:
            raise bridge.CertificationLimit('projector enclosure unresolved within precision budget')
        precision = min(2 * precision, max_precision)
