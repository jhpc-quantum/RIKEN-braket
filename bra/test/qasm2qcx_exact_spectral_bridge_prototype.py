#!/usr/bin/env python3
"""Symbolic endpoint spectra for the #634 feasibility study.

The result retains exact eigenvalues and an unexpanded projector fraction.
It does not evaluate non-endpoint phases or emit QCX.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

import qasm2qcx_exact_single_qubit_prototype as exact


@dataclass(frozen=True)
class ScalarSpectrum:
    eigenvalue: exact.Exact

    @property
    def negative_one_multiplicity(self):
        return 2 if self.eigenvalue == -exact.ONE else 0


@dataclass(frozen=True)
class EndpointSpectrum:
    # The -1 eigenphase is +pi under the converter's convention. The second
    # eigenvalue remains exact; its phase has not yet been certified.
    other_eigenvalue: exact.Exact
    projector_numerator: tuple
    projector_denominator: exact.Exact

    @property
    def negative_one_multiplicity(self):
        return 1


def validate_unitary(matrix):
    if (len(matrix) != 2 or any(len(row) != 2 for row in matrix)
            or any(not isinstance(value, exact.Exact) for row in matrix for value in row)):
        raise ValueError('expected a 2x2 matrix of exact values')
    adjoint = tuple(tuple(matrix[column][row].conjugate() for column in range(2))
                    for row in range(2))
    if exact.multiply_matrices(adjoint, matrix) != exact.GATES['id']:
        raise ValueError('expected an exactly unitary matrix')


def endpoint_spectrum(matrix):
    """Return an exact scalar or -1 endpoint spectrum for a unitary matrix.

    For a nonscalar matrix with eigenvalues -1 and lambda, the endpoint
    projector is (G - lambda*I)/(-1 - lambda). The quotient stays symbolic:
    division need not remain in the exact-entry ring.
    """
    validate_unitary(matrix)
    scalar = exact.scalar_value(matrix)
    if scalar is not None:
        return ScalarSpectrum(scalar)
    if not exact.has_negative_one_eigenvalue(matrix):
        raise ValueError('non-scalar matrix has no -1 eigenvalue')
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    other = -determinant
    denominator = -exact.ONE - other
    if denominator == exact.ZERO:
        raise ValueError('repeated endpoint requires an exact scalar matrix')
    numerator = ((matrix[0][0] - other, matrix[0][1]),
                 (matrix[1][0], matrix[1][1] - other))
    return EndpointSpectrum(other, numerator, denominator)


class CertificationLimit(ValueError):
    """An exact input or certified enclosure exceeded the prototype budget."""


@dataclass(frozen=True)
class CertifiedEigenvalue:
    # Rational enclosures contain the exact real and imaginary components.
    real: tuple[Fraction, Fraction]
    imaginary: tuple[Fraction, Fraction]
    # 'above', 'below', 'positive-real', or 'negative-endpoint'.
    branch: str
    exact_value: exact.Exact | None = None


def _sign_surd(a, b):
    """Sign of a + b*sqrt(2), using only integer arithmetic."""
    if not a:
        return (b > 0) - (b < 0)
    if not b or (a > 0) == (b > 0):
        return (a > 0) - (a < 0)
    comparison = (a * a > 2 * b * b) - (a * a < 2 * b * b)
    return comparison if a > 0 else -comparison


def _add(x, y):
    return x[0] + y[0], x[1] + y[1]


def _neg(x):
    return -x[1], -x[0]


def _mul(x, y):
    values = (x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1])
    return min(values), max(values)


def _square(x):
    lower = Fraction(0) if x[0] <= 0 <= x[1] else min(x[0] ** 2, x[1] ** 2)
    return lower, max(x[0] ** 2, x[1] ** 2)


def _sqrt_bounds(x, bits):
    if x[1] < 0:
        raise ValueError('negative square-root enclosure')
    scale = 1 << bits
    def lower(value):
        value = max(Fraction(0), value)
        return Fraction(isqrt((value.numerator << (2 * bits)) // value.denominator), scale)
    lo = lower(x[0])
    hi = lower(x[1])
    if hi * hi < x[1]:
        hi += Fraction(1, scale)
    return lo, hi


def _exact_interval(value, bits):
    root = _sqrt_bounds((Fraction(2), Fraction(2)), bits)
    denominator = 1 << value.k
    def component(a, b):
        return _mul(_add((Fraction(a), Fraction(a)), _mul((Fraction(b), Fraction(b)), root)),
                    (Fraction(1, denominator), Fraction(1, denominator)))
    return component(value.a, value.b), component(value.c, value.d)


def _exact_eigenvalue(value, bits):
    real, imaginary = _exact_interval(value, bits)
    sign = _sign_surd(value.c, value.d)
    if sign:
        branch = 'above' if sign > 0 else 'below'
    elif value == -exact.ONE:
        branch = 'negative-endpoint'
    elif value == exact.ONE:
        branch = 'positive-real'
    else:
        raise ValueError('real eigenvalue of a unitary matrix must be +1 or -1')
    return CertifiedEigenvalue(real, imaginary, branch, value)


def _sqrt_complex(value, real_sign, imag_sign, bits):
    """Enclose the principal square root of an exactly signed complex value."""
    real, imaginary = value
    zero = (Fraction(0), Fraction(0))
    if imag_sign == 0:
        return (_sqrt_bounds(real, bits), zero) if real_sign >= 0 else (zero, _sqrt_bounds(_neg(real), bits))
    magnitude = _sqrt_bounds(_add(_square(real), _square(imaginary)), bits)
    half = (Fraction(1, 2), Fraction(1, 2))
    real_root = _sqrt_bounds(_mul(_add(magnitude, real), half), bits)
    imag_root = _sqrt_bounds(_mul(_add(magnitude, _neg(real)), half), bits)
    return real_root, imag_root if imag_sign > 0 else _neg(imag_root)


def certify_eigenvalue_branches(matrix, *, max_bits=1024, max_input_bits=4096, min_bits=None):
    """Certify branch sides for an exact 2x2 unitary; do not round phases.

    Characteristic roots are enclosed with rational interval arithmetic.
    Exact +/-1 roots are handled separately so refinement need not prove an
    equality. The returned plus/minus root labels refer to the principal
    discriminant square root, not to an eigenbasis or emitted QCX phases.
    """
    if type(max_bits) is not int or max_bits < 8 or type(max_input_bits) is not int or max_input_bits < 1:
        raise ValueError('invalid certification budget')
    if min_bits is not None and (type(min_bits) is not int or not 8 <= min_bits <= max_bits):
        raise ValueError('invalid minimum enclosure precision')
    bits, denominator = exact.representation_size(matrix)
    if max(bits, denominator) > max_input_bits:
        raise CertificationLimit('exact input exceeds coefficient budget')
    validate_unitary(matrix)
    scalar = exact.scalar_value(matrix)
    if scalar is not None:
        eigen = _exact_eigenvalue(scalar, max_bits)
        return eigen, eigen
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if exact.has_negative_one_eigenvalue(matrix):
        return _exact_eigenvalue(-exact.ONE, max_bits), _exact_eigenvalue(-determinant, max_bits)
    plus_one = ((matrix[0][0] - exact.ONE) * (matrix[1][1] - exact.ONE)
                - matrix[0][1] * matrix[1][0]) == exact.ZERO
    if plus_one:
        return _exact_eigenvalue(exact.ONE, max_bits), _exact_eigenvalue(determinant, max_bits)
    trace = matrix[0][0] + matrix[1][1]
    discriminant = trace * trace - exact.Exact(4) * determinant
    if discriminant == exact.ZERO:
        raise ValueError('repeated eigenvalue requires an exact scalar matrix')
    real_sign = _sign_surd(discriminant.a, discriminant.b)
    imag_sign = _sign_surd(discriminant.c, discriminant.d)
    precision = min(32, max_bits) if min_bits is None else min_bits
    while True:
        trace_real, trace_imag = _exact_interval(trace, precision)
        root_real, root_imag = _sqrt_complex(_exact_interval(discriminant, precision),
                                             real_sign, imag_sign, precision)
        half = (Fraction(1, 2), Fraction(1, 2))
        roots = []
        for sign in (1, -1):
            rr, ri = (root_real, root_imag) if sign == 1 else (_neg(root_real), _neg(root_imag))
            real = _mul(_add(trace_real, rr), half)
            imaginary = _mul(_add(trace_imag, ri), half)
            if imaginary[0] > 0:
                branch = 'above'
            elif imaginary[1] < 0:
                branch = 'below'
            else:
                break
            roots.append(CertifiedEigenvalue(real, imaginary, branch))
        if len(roots) == 2:
            return tuple(roots)
        if precision == max_bits:
            raise CertificationLimit('eigenvalue branch unresolved within precision budget')
        precision = min(2 * precision, max_bits)
