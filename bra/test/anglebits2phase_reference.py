"""Independent exact-turn reference for the planned ANGLEBITS2PHASE (#396).

This is a test oracle, not converter/native implementation. See
docs/anglebits2phase-design.md for the Stage 1 interface contract.
"""

import ctypes
from dataclasses import dataclass
from fractions import Fraction
import math
import re


class ChainSyntaxError(ValueError):
    pass


class PhaseDomainError(ValueError):
    pass


class PhaseResourceError(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    uint_width: int = ctypes.sizeof(ctypes.c_uint) * 8
    max_modifiers: int = 1024
    max_literal_chars: int = 4096
    max_exact_bits: int = 65536

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in
               (self.uint_width, self.max_modifiers, self.max_literal_chars, self.max_exact_bits)):
            raise ValueError('reference limits must be positive integers')


@dataclass(frozen=True)
class Modifier:
    kind: str
    exponent: Fraction | None = None


DEFAULT_LIMITS = Limits()
INTEGER = re.compile(r'[+-]?[0-9]+\Z')
FLOATING = re.compile(r'[+-]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)\Z')


def check_exact_size(value: Fraction, limits: Limits) -> None:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > limits.max_exact_bits:
        raise PhaseResourceError('exact numerator/denominator bit limit exceeded')


def decode_exponent(token: str, limits: Limits = DEFAULT_LIMITS) -> Fraction:
    if not isinstance(token, str):
        raise ChainSyntaxError('POW requires a numeric literal')
    if len(token) > limits.max_literal_chars:
        raise PhaseResourceError('exponent literal character limit exceeded')
    if INTEGER.fullmatch(token):
        value = Fraction(int(token))
    elif FLOATING.fullmatch(token):
        stored = float(token)
        if not math.isfinite(stored):
            raise PhaseDomainError('POW exponent must be finite binary64')
        value = Fraction.from_float(stored)
    elif token.lower().lstrip('+-') in ('inf', 'infinity', 'nan'):
        raise PhaseDomainError('POW exponent must be finite binary64')
    else:
        raise ChainSyntaxError('POW requires a decimal integer or floating literal')
    check_exact_size(value, limits)
    return value


def validate_chain(chain: tuple[Modifier, ...], limits: Limits = DEFAULT_LIMITS) -> None:
    if len(chain) > limits.max_modifiers:
        raise PhaseResourceError('modifier count limit exceeded')
    for modifier in chain:
        if not isinstance(modifier, Modifier):
            raise ChainSyntaxError('invalid modifier descriptor')
        if modifier.kind == 'INV':
            if modifier.exponent is not None:
                raise ChainSyntaxError('INV takes no operand')
        elif modifier.kind == 'POW':
            if type(modifier.exponent) is not Fraction:
                raise ChainSyntaxError('POW requires an exact decoded exponent')
            denominator = modifier.exponent.denominator
            if denominator & (denominator - 1):
                raise PhaseDomainError('POW exponent must be dyadic')
            check_exact_size(modifier.exponent, limits)
        else:
            raise ChainSyntaxError('unknown modifier descriptor')


def decode_chain(tokens: list[str] | tuple[str, ...], limits: Limits = DEFAULT_LIMITS) -> tuple[Modifier, ...]:
    chain = []
    position = 0
    while position < len(tokens):
        if len(chain) >= limits.max_modifiers:
            raise PhaseResourceError('modifier count limit exceeded')
        keyword = tokens[position]
        if not isinstance(keyword, str):
            raise ChainSyntaxError('modifier keyword must be text')
        position += 1
        if keyword.upper() == 'INV':
            chain.append(Modifier('INV'))
        elif keyword.upper() == 'POW':
            if position >= len(tokens):
                raise ChainSyntaxError('POW requires an exponent')
            chain.append(Modifier('POW', decode_exponent(tokens[position], limits)))
            position += 1
        else:
            raise ChainSyntaxError('expected INV or POW')
    result = tuple(chain)
    validate_chain(result, limits)
    return result


def principal_turn(value: Fraction) -> Fraction:
    remainder = value % 1
    return remainder - 1 if remainder > Fraction(1, 2) else remainder


def evaluate_turns(bits: int, width: int, chain: tuple[Modifier, ...] = (),
                   limits: Limits = DEFAULT_LIMITS) -> Fraction:
    # Descriptor construction is static; runtime operand checks come later.
    validate_chain(chain, limits)
    if type(width) is not int or not 1 <= width <= limits.uint_width:
        raise PhaseDomainError('width must fit native UINT storage')
    if type(bits) is not int or not 0 <= bits < 1 << width:
        raise PhaseDomainError('angle bits must fit width')
    phase = principal_turn(Fraction(bits, 1 << width))
    check_exact_size(phase, limits)
    for modifier in chain:
        phase = -phase if modifier.kind == 'INV' else principal_turn(phase) * modifier.exponent
        # Canonicalization is exact and preserves the current unitary. It
        # does not distribute, combine, or reorder any powers.
        phase = principal_turn(phase)
        check_exact_size(phase, limits)
    return phase
