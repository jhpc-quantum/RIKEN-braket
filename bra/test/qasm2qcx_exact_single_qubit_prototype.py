#!/usr/bin/env python3
"""Exact fixed-gate entries for the #632 feasibility study.

Each value is (a + b*sqrt(2) + i*(c + d*sqrt(2))) / 2**k.
This experimental module is not imported by the converter.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Exact:
    a: int = 0
    b: int = 0
    c: int = 0
    d: int = 0
    k: int = 0

    def __post_init__(self):
        if any(type(value) is not int for value in (self.a, self.b, self.c, self.d, self.k)):
            raise TypeError('exact coefficients and denominator exponent must be integers')
        if self.k < 0:
            raise ValueError('denominator exponent must be nonnegative')
        a, b, c, d, k = self.a, self.b, self.c, self.d, self.k
        while k and not (a | b | c | d) & 1:
            a, b, c, d, k = a // 2, b // 2, c // 2, d // 2, k - 1
        for name, value in zip(('a', 'b', 'c', 'd', 'k'), (a, b, c, d, k)):
            object.__setattr__(self, name, value)

    def __neg__(self):
        return Exact(-self.a, -self.b, -self.c, -self.d, self.k)

    def __add__(self, other):
        if not isinstance(other, Exact):
            return NotImplemented
        k = max(self.k, other.k)
        left, right = 1 << (k - self.k), 1 << (k - other.k)
        return Exact(*(left * x + right * y for x, y in zip(
            (self.a, self.b, self.c, self.d), (other.a, other.b, other.c, other.d))), k)

    def __sub__(self, other):
        if not isinstance(other, Exact):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, Exact):
            return NotImplemented
        a, b, c, d = self.a, self.b, self.c, self.d
        e, f, g, h = other.a, other.b, other.c, other.d
        return Exact(a * e + 2 * b * f - c * g - 2 * d * h,
                     a * f + b * e - c * h - d * g,
                     a * g + 2 * b * h + c * e + 2 * d * f,
                     a * h + b * g + c * f + d * e,
                     self.k + other.k)

    def conjugate(self):
        return Exact(self.a, self.b, -self.c, -self.d, self.k)

    def to_complex(self):
        return complex(math.ldexp(self.a + self.b * math.sqrt(2), -self.k),
                       math.ldexp(self.c + self.d * math.sqrt(2), -self.k))


ZERO = Exact()
ONE = Exact(1)
I = Exact(c=1)
SQRT_HALF = Exact(b=1, k=1)
OMEGA = Exact(b=1, d=1, k=1)
HALF_PLUS_HALF_I = Exact(a=1, c=1, k=1)
HALF_MINUS_HALF_I = Exact(a=1, c=-1, k=1)

GATES = {
    'id': ((ONE, ZERO), (ZERO, ONE)),
    'x': ((ZERO, ONE), (ONE, ZERO)),
    'y': ((ZERO, -I), (I, ZERO)),
    'z': ((ONE, ZERO), (ZERO, -ONE)),
    'h': ((SQRT_HALF, SQRT_HALF), (SQRT_HALF, -SQRT_HALF)),
    's': ((ONE, ZERO), (ZERO, I)),
    'sdg': ((ONE, ZERO), (ZERO, -I)),
    't': ((ONE, ZERO), (ZERO, OMEGA)),
    'tdg': ((ONE, ZERO), (ZERO, OMEGA.conjugate())),
    'sx': ((HALF_PLUS_HALF_I, HALF_MINUS_HALF_I),
           (HALF_MINUS_HALF_I, HALF_PLUS_HALF_I)),
}


def multiply_matrices(left, right):
    """Return the exact matrix product left * right."""
    return tuple(tuple(sum((left[row][k] * right[k][column] for k in range(2)), ZERO)
                       for column in range(2)) for row in range(2))


def compose_sequence(names):
    """Apply the named gates in execution order, starting at identity."""
    result = GATES['id']
    for name in names:
        result = multiply_matrices(GATES[name], result)
    return result


def scalar_value(matrix):
    """Return the common diagonal for an exact scalar matrix, else None."""
    if matrix[0][1] == ZERO and matrix[1][0] == ZERO and matrix[0][0] == matrix[1][1]:
        return matrix[0][0]
    return None


def has_negative_one_eigenvalue(matrix):
    """Test det(matrix + I) == 0 exactly, without finding eigenvalues."""
    return ((matrix[0][0] + ONE) * (matrix[1][1] + ONE)
            - matrix[0][1] * matrix[1][0]) == ZERO


def representation_size(matrix):
    """Return max numerator bit length and denominator exponent in a matrix."""
    entries = [entry for row in matrix for entry in row]
    return (max(abs(value).bit_length() for entry in entries
                for value in (entry.a, entry.b, entry.c, entry.d)),
            max(entry.k for entry in entries))
