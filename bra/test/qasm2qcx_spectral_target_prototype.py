#!/usr/bin/env python3
"""Explicit target and accuracy descriptors for the #636 study.

This is a mathematical model, not a bra-build detector or converter option.
"""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class BinaryTarget:
    """Binary format with normal exponent range and a stated rounding model.

    precision counts significand bits including the leading bit; emin/emax
    are exponents of the leading bit of normal finite values.
    """
    name: str
    precision: int
    emin: int
    emax: int
    subnormals: bool
    rounding: str = 'nearest-even'

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('target name must be nonempty')
        if any(type(value) is not int for value in (self.precision, self.emin, self.emax)):
            raise TypeError('precision and exponents must be integers')
        if not 2 <= self.precision <= 4096 or not -65536 <= self.emin <= 0 <= self.emax <= 65536:
            raise ValueError('target precision or exponent range outside prototype limits')
        if type(self.subnormals) is not bool:
            raise TypeError('subnormals must be Boolean')
        if self.rounding != 'nearest-even':
            raise ValueError('only nearest-even is modeled')

    @property
    def unit_roundoff(self):
        return Fraction(1, 1 << self.precision)

    @property
    def smallest_positive_exponent(self):
        return self.emin - self.precision + 1 if self.subnormals else self.emin


# These are mathematical profiles, not claims about a particular bra binary.
BINARY32 = BinaryTarget('binary32', 24, -126, 127, True)
BINARY64 = BinaryTarget('binary64', 53, -1022, 1023, True)


@dataclass(frozen=True)
class AccuracyRequest:
    """Requested absolute operator error for the converter's emitted gate.

    This does not certify any output; a future bridge must prove the bound.
    Backend parsing and execution require separate assumptions and analysis.
    """
    target: BinaryTarget
    max_operator_error: Fraction

    def __post_init__(self):
        if not isinstance(self.target, BinaryTarget):
            raise TypeError('an explicit target descriptor is required')
        if not isinstance(self.max_operator_error, Fraction):
            raise TypeError('operator error must be an exact Fraction')
        if not 0 < self.max_operator_error < 1:
            raise ValueError('operator error must be strictly between zero and one')
