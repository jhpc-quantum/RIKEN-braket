"""Bounded exact arithmetic for diagonal-gate certification.

Bounds count explicitly modeled work and integer sizes; they do not constitute
a process-memory or parser budget.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from types import MappingProxyType


class CertificationError(ValueError):
    """Base class for errors in the diagonal-certification foundation."""


class InvalidInput(CertificationError):
    pass


class UnsupportedGate(CertificationError):
    pass


class BudgetExceeded(CertificationError):
    def __init__(self, category):
        self.category = category
        super().__init__(f'{category} budget exceeded')


class AccuracyNotMet(CertificationError):
    pass


@dataclass(frozen=True)
class BudgetLimits:
    # Candidate defaults from the design; production adoption needs review.
    input_bits: int = 4096
    exact_bits: int = 4096
    integer_bits: int = 8192
    denominator_exponent: int = 4096
    rational_bits: int = 8192
    precision_bits: int = 1024
    series_terms: int = 512
    work: int = 100000
    gate_steps: int = 100000
    nesting: int = 128
    decimal_places: int = 200
    output_chars: int = 1000000
    output_lines: int = 100000

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < 0:
                raise InvalidInput(f'{name} limit must be a nonnegative integer')


class Budget:
    """Monotonic counters shared by a complete future source gate call."""

    def __init__(self, limits=None):
        self.limits = BudgetLimits() if limits is None else limits
        if not isinstance(self.limits, BudgetLimits):
            raise InvalidInput('expected BudgetLimits')
        self.work = 0
        self.gate_steps = 0
        self.output_chars = 0
        self.output_lines = 0
        self.series_terms_used = 0

    def charge_work(self, count=1):
        if type(count) is not int or count < 0:
            raise InvalidInput('work charge must be a nonnegative integer')
        if count > self.limits.work - self.work:
            raise BudgetExceeded('work')
        self.work += count

    def charge_gate(self, depth=1):
        self.charge_work()
        if type(depth) is not int or depth < 1:
            raise InvalidInput('gate depth must be a positive integer')
        if depth > self.limits.nesting:
            raise BudgetExceeded('nesting')
        if self.gate_steps >= self.limits.gate_steps:
            raise BudgetExceeded('gate steps')
        self.gate_steps += 1

    def reserve_output(self, *, chars=0, lines=0):
        self.charge_work()
        if any(type(value) is not int or value < 0 for value in (chars, lines)):
            raise InvalidInput('output reservation must be nonnegative integers')
        if chars > self.limits.output_chars - self.output_chars:
            raise BudgetExceeded('output characters')
        if lines > self.limits.output_lines - self.output_lines:
            raise BudgetExceeded('output lines')
        self.output_chars += chars
        self.output_lines += lines

    def check_integer_bits(self, bits):
        if bits > self.limits.integer_bits:
            raise BudgetExceeded('intermediate integer bits')

    def check_rational_bits(self, bits):
        self.check_integer_bits(bits)
        if bits > self.limits.rational_bits:
            raise BudgetExceeded('rational bits')

    def check_precision(self, bits):
        if bits > self.limits.precision_bits:
            raise BudgetExceeded('precision bits')

    def charge_series_term(self):
        self.charge_work()
        self.series_terms_used += 1

    def check_exact(self, values, denominator):
        if denominator > self.limits.denominator_exponent:
            raise BudgetExceeded('exact denominator exponent')
        if any(value.bit_length() > self.limits.exact_bits for value in values):
            raise BudgetExceeded('exact coefficient bits')


def _checked_add(left, right, budget):
    budget.charge_work()
    # The zero case avoids rejecting an unchanged exact-limit operand.
    upper = max(left.bit_length(), right.bit_length()) + bool(left and right)
    budget.check_integer_bits(upper)
    return left + right


def _checked_multiply(left, right, budget):
    budget.charge_work()
    upper = left.bit_length() + right.bit_length() if left and right else 0
    budget.check_integer_bits(upper)
    return left * right


def _checked_shift(value, amount, budget):
    budget.charge_work()
    if amount < 0:
        raise InvalidInput('shift amount must be nonnegative')
    budget.check_integer_bits(value.bit_length() + amount if value else 0)
    return value << amount if value else 0


@dataclass(frozen=True)
class Exact:
    """(a+b*sqrt(2)+i*(c+d*sqrt(2))) / 2**k; operations require normal form."""
    a: int = 0
    b: int = 0
    c: int = 0
    d: int = 0
    k: int = 0

    def __post_init__(self):
        if any(type(value) is not int for value in (self.a, self.b, self.c, self.d, self.k)) or self.k < 0:
            raise InvalidInput('exact components must be integers with nonnegative denominator exponent')


def _components(value):
    return value.a, value.b, value.c, value.d


def _validate_exact(value, budget):
    if not isinstance(value, Exact):
        raise InvalidInput('expected an Exact value')
    budget.check_exact(_components(value), value.k)
    for component in _components(value):
        budget.check_integer_bits(component.bit_length())
    if value.k and not any(component & 1 for component in _components(value)):
        raise InvalidInput('exact value is not normalized')


def _make_exact(values, denominator, budget):
    budget.charge_work()
    if denominator > budget.limits.denominator_exponent:
        raise BudgetExceeded('exact denominator exponent')
    for value in values:
        budget.check_integer_bits(value.bit_length())
    # Compute the common power of two without a loop proportional to k.
    nonzero = [abs(value) for value in values if value]
    common = min((value & -value).bit_length() - 1 for value in nonzero) if nonzero else denominator
    shift = min(denominator, common)
    values = tuple(value >> shift for value in values)
    denominator -= shift
    budget.check_exact(values, denominator)
    return Exact(*values, denominator)


def _exact_add(left, right, budget):
    _validate_exact(left, budget)
    _validate_exact(right, budget)
    denominator = max(left.k, right.k)
    values = tuple(_checked_add(_checked_shift(a, denominator - left.k, budget),
                                _checked_shift(b, denominator - right.k, budget), budget)
                   for a, b in zip(_components(left), _components(right)))
    return _make_exact(values, denominator, budget)


def _exact_multiply(left, right, budget):
    _validate_exact(left, budget)
    _validate_exact(right, budget)
    denominator = left.k + right.k
    if denominator > budget.limits.denominator_exponent:
        raise BudgetExceeded('exact denominator exponent')
    a, b, c, d = _components(left)
    e, f, g, h = _components(right)
    def product(x, y, double=False):
        value = _checked_multiply(x, y, budget)
        return _checked_shift(value, 1, budget) if double else value
    def sum4(first, second, third, fourth):
        return _checked_add(_checked_add(_checked_add(first, second, budget), third, budget), fourth, budget)
    values = (sum4(product(a, e), product(b, f, True), -product(c, g), -product(d, h, True)),
              sum4(product(a, f), product(b, e), -product(c, h), -product(d, g)),
              sum4(product(a, g), product(b, h, True), product(c, e), product(d, f, True)),
              sum4(product(a, h), product(b, g), product(c, f), product(d, e)))
    return _make_exact(values, denominator, budget)


ZERO = Exact()
ONE = Exact(1)
I = Exact(c=1)
MINUS_ONE = Exact(-1)
MINUS_I = Exact(c=-1)
SQRT_HALF = Exact(b=1, k=1)
MINUS_SQRT_HALF = Exact(b=-1, k=1)
OMEGA = Exact(b=1, d=1, k=1)
OMEGA_CONJUGATE = Exact(b=1, d=-1, k=1)
HALF_PLUS_I = Exact(a=1, c=1, k=1)
HALF_MINUS_I = Exact(a=1, c=-1, k=1)

FIXED_GATES = MappingProxyType({
    'id': ((ONE, ZERO), (ZERO, ONE)),
    'x': ((ZERO, ONE), (ONE, ZERO)),
    'y': ((ZERO, MINUS_I), (I, ZERO)),
    'z': ((ONE, ZERO), (ZERO, MINUS_ONE)),
    'h': ((SQRT_HALF, SQRT_HALF), (SQRT_HALF, MINUS_SQRT_HALF)),
    's': ((ONE, ZERO), (ZERO, I)),
    'sdg': ((ONE, ZERO), (ZERO, MINUS_I)),
    't': ((ONE, ZERO), (ZERO, OMEGA)),
    'tdg': ((ONE, ZERO), (ZERO, OMEGA_CONJUGATE)),
    'sx': ((HALF_PLUS_I, HALF_MINUS_I), (HALF_MINUS_I, HALF_PLUS_I)),
})


def multiply_matrices(left, right, budget):
    """Compose two exact 2x2 matrices with checked intermediate operations."""
    if not isinstance(budget, Budget):
        raise InvalidInput('expected a shared Budget')
    if any(not isinstance(matrix, (tuple, list)) or len(matrix) != 2
           or any(not isinstance(row, (tuple, list)) or len(row) != 2 for row in matrix)
           for matrix in (left, right)):
        raise InvalidInput('expected two 2x2 matrices')
    for matrix in (left, right):
        for row in matrix:
            for value in row:
                _validate_exact(value, budget)
    return tuple(tuple(_exact_add(_exact_multiply(left[row][0], right[0][column], budget),
                                  _exact_multiply(left[row][1], right[1][column], budget), budget)
                       for column in range(2)) for row in range(2))


def adjoint_matrix(matrix, budget):
    """Return the exact conjugate transpose using the caller's shared budget."""
    if not isinstance(budget, Budget):
        raise InvalidInput('expected a shared Budget')
    if (not isinstance(matrix, (tuple, list)) or len(matrix) != 2
            or any(not isinstance(row, (tuple, list)) or len(row) != 2 for row in matrix)):
        raise InvalidInput('expected a 2x2 matrix')
    for row in matrix:
        for value in row:
            _validate_exact(value, budget)
    # Negation does not increase an integer's bit length; _make_exact charges
    # each entry and verifies the output limits before constructing it.
    return tuple(tuple(_make_exact((value.a, value.b, -value.c, -value.d), value.k, budget)
                       for value in (matrix[column][row] for column in range(2)))
                 for row in range(2))


def compose_fixed_gates(names, budget):
    """Compose a streamed sequence in execution order using the shared budget."""
    if not isinstance(budget, Budget):
        raise InvalidInput('expected a shared Budget')
    matrix = FIXED_GATES['id']
    budget.charge_work()
    for row in matrix:
        for value in row:
            _validate_exact(value, budget)
    for name in names:
        budget.charge_gate()
        if not isinstance(name, str) or name not in FIXED_GATES:
            raise UnsupportedGate('unsupported fixed gate')
        matrix = multiply_matrices(FIXED_GATES[name], matrix, budget)
    return matrix


def classify_matrix(matrix, budget):
    """Return 'non-diagonal', 'diagonal', or 'scalar' using exact zeros."""
    if not isinstance(budget, Budget):
        raise InvalidInput('expected a shared Budget')
    budget.charge_work()
    if (not isinstance(matrix, (tuple, list)) or len(matrix) != 2
            or any(not isinstance(row, (tuple, list)) or len(row) != 2 for row in matrix)):
        raise InvalidInput('expected a 2x2 matrix')
    for row in matrix:
        for value in row:
            _validate_exact(value, budget)
    if matrix[0][1] != ZERO or matrix[1][0] != ZERO:
        return 'non-diagonal'
    return 'scalar' if matrix[0][0] == matrix[1][1] else 'diagonal'


def _validate_rational(value, budget):
    if not isinstance(value, Fraction):
        raise InvalidInput('expected an exact Fraction')
    budget.check_rational_bits(value.numerator.bit_length())
    budget.check_rational_bits(value.denominator.bit_length())


def _fraction(numerator, denominator, budget):
    budget.charge_work()  # Includes normalization/GCD with bounded operands.
    if denominator <= 0:
        raise InvalidInput('rational denominator must be positive')
    budget.check_rational_bits(numerator.bit_length())
    budget.check_rational_bits(denominator.bit_length())
    return Fraction(numerator, denominator)


def _qinteger_product(left, right, budget):
    upper = left.bit_length() + right.bit_length() if left and right else 0
    budget.check_rational_bits(upper)
    return _checked_multiply(left, right, budget)


def _qinteger_sum(left, right, budget):
    upper = max(left.bit_length(), right.bit_length()) + bool(left and right)
    budget.check_rational_bits(upper)
    return _checked_add(left, right, budget)


def _qadd(left, right, budget):
    _validate_rational(left, budget)
    _validate_rational(right, budget)
    if left.numerator == 0:
        budget.charge_work()
        return right
    if right.numerator == 0:
        budget.charge_work()
        return left
    first = _qinteger_product(left.numerator, right.denominator, budget)
    second = _qinteger_product(right.numerator, left.denominator, budget)
    numerator = _qinteger_sum(first, second, budget)
    denominator = _qinteger_product(left.denominator, right.denominator, budget)
    return _fraction(numerator, denominator, budget)


def _qneg(value, budget):
    _validate_rational(value, budget)
    budget.charge_work()
    return -value


def _qsub(left, right, budget):
    return _qadd(left, _qneg(right, budget), budget)


def _qmul(left, right, budget):
    _validate_rational(left, budget)
    _validate_rational(right, budget)
    if not left.numerator or not right.numerator:
        budget.charge_work()
        return Fraction(0)
    numerator = _qinteger_product(left.numerator, right.numerator, budget)
    denominator = _qinteger_product(left.denominator, right.denominator, budget)
    return _fraction(numerator, denominator, budget)


def _qdiv(left, right, budget):
    _validate_rational(left, budget)
    _validate_rational(right, budget)
    if not right.numerator:
        raise InvalidInput('rational division by zero')
    if not left.numerator:
        budget.charge_work()
        return Fraction(0)
    numerator = _qinteger_product(left.numerator, right.denominator, budget)
    denominator = _qinteger_product(left.denominator, abs(right.numerator), budget)
    if right.numerator < 0:
        numerator = -numerator
    return _fraction(numerator, denominator, budget)


def _qcompare(left, right, budget):
    _validate_rational(left, budget)
    _validate_rational(right, budget)
    first = _qinteger_product(left.numerator, right.denominator, budget)
    second = _qinteger_product(right.numerator, left.denominator, budget)
    budget.charge_work()
    return (first > second) - (first < second)


def _pow2(exponent, budget):
    if exponent < 0:
        raise InvalidInput('power exponent must be nonnegative')
    budget.check_rational_bits(exponent + 1)
    return _checked_shift(1, exponent, budget)


def _sqrt2_interval(bits, budget):
    budget.check_precision(bits)
    scaled = _checked_shift(2, 2 * bits, budget)
    budget.charge_work()  # isqrt input is bounded before allocation.
    root = isqrt(scaled)
    scale = _pow2(bits, budget)
    return _fraction(root, scale, budget), _fraction(_checked_add(root, 1, budget), scale, budget)


def _exact_component_interval(a, b, denominator, root, budget):
    first = _fraction(a, 1, budget)
    coefficient = _fraction(b, 1, budget)
    bounds = root if b >= 0 else (root[1], root[0])
    divisor = _fraction(denominator, 1, budget)
    return tuple(_qdiv(_qadd(first, _qmul(coefficient, bound, budget), budget), divisor, budget)
                 for bound in bounds)


def _sign_surd(a, b, budget):
    budget.charge_work()
    if not a:
        return (b > 0) - (b < 0)
    if not b or (a > 0) == (b > 0):
        return (a > 0) - (a < 0)
    first = _checked_multiply(a, a, budget)
    second = _checked_shift(_checked_multiply(b, b, budget), 1, budget)
    comparison = (first > second) - (first < second)
    return comparison if a > 0 else -comparison


def _atan_series(value, bits, budget):
    _validate_rational(value, budget)
    budget.check_precision(bits)
    if value.numerator < 0:
        lo, hi = _atan_series(_qneg(value, budget), bits, budget)
        return _qneg(hi, budget), _qneg(lo, budget)
    if _qcompare(value, Fraction(1, 2), budget) > 0:
        raise InvalidInput('atan series argument outside reduced range')
    if not value.numerator:
        return Fraction(0), Fraction(0)
    tolerance = _fraction(1, _pow2(bits, budget), budget)
    squared = _qmul(value, value, budget)
    power = value
    total = Fraction(0)
    index = 0
    while index < budget.limits.series_terms:
        budget.charge_series_term()
        term = _qdiv(power, _fraction(2 * index + 1, 1, budget), budget)
        total = _qadd(total, term, budget) if index % 2 == 0 else _qsub(total, term, budget)
        power = _qmul(power, squared, budget)
        next_term = _qdiv(power, _fraction(2 * index + 3, 1, budget), budget)
        if _qcompare(next_term, tolerance, budget) <= 0:
            next_total = _qsub(total, next_term, budget) if index % 2 == 0 else _qadd(total, next_term, budget)
            return (next_total, total) if index % 2 == 0 else (total, next_total)
        index += 1
    raise BudgetExceeded('series terms')


def _pi_interval(bits, budget):
    # Machin identity. No cross-call cache: all arithmetic is charged.
    budget.check_precision(bits + 8)
    a = _atan_series(Fraction(1, 5), bits + 8, budget)
    b = _atan_series(Fraction(1, 239), bits + 8, budget)
    return (_qsub(_qmul(Fraction(16), a[0], budget), _qmul(Fraction(4), b[1], budget), budget),
            _qsub(_qmul(Fraction(16), a[1], budget), _qmul(Fraction(4), b[0], budget), budget))


def _atan(value, bits, pi, budget):
    _validate_rational(value, budget)
    if value.numerator < 0:
        lo, hi = _atan(_qneg(value, budget), bits, pi, budget)
        return _qneg(hi, budget), _qneg(lo, budget)
    if _qcompare(value, Fraction(1), budget) > 0:
        lo, hi = _atan(_qdiv(Fraction(1), value, budget), bits, pi, budget)
        return (_qsub(_qdiv(pi[0], Fraction(2), budget), hi, budget),
                _qsub(_qdiv(pi[1], Fraction(2), budget), lo, budget))
    if _qcompare(value, Fraction(1, 2), budget) > 0:
        reduced = _qdiv(_qsub(value, Fraction(1), budget), _qadd(value, Fraction(1), budget), budget)
        lo, hi = _atan_series(reduced, bits, budget)
        return (_qadd(_qdiv(pi[0], Fraction(4), budget), lo, budget),
                _qadd(_qdiv(pi[1], Fraction(4), budget), hi, budget))
    return _atan_series(value, bits, budget)


class _NeedPrecision(Exception):
    pass


def _interval_multiply(left, right, budget):
    values = tuple(_qmul(a, b, budget) for a in left for b in right)
    lower = upper = values[0]
    for value in values[1:]:
        if _qcompare(value, lower, budget) < 0:
            lower = value
        if _qcompare(value, upper, budget) > 0:
            upper = value
    return lower, upper


def _principal_phase(value, bits, root2, pi, budget):
    _validate_exact(value, budget)
    sign = _sign_surd(value.c, value.d, budget)
    if not sign:
        if value == MINUS_ONE:
            return 'negative-endpoint', pi
        if value == ONE:
            return 'positive-real', (Fraction(0), Fraction(0))
        raise InvalidInput('real unit eigenvalue must be +1 or -1')
    denominator = _pow2(value.k, budget)
    x = _exact_component_interval(value.a, value.b, denominator, root2, budget)
    y = _exact_component_interval(value.c, value.d, denominator, root2, budget)
    if y[0].numerator <= 0 <= y[1].numerator:
        raise _NeedPrecision
    reciprocal = (_qdiv(Fraction(1), y[1], budget), _qdiv(Fraction(1), y[0], budget))
    ratio = _interval_multiply(x, reciprocal, budget)
    lower_atan = _atan(ratio[0], bits, pi, budget)[0]
    upper_atan = _atan(ratio[1], bits, pi, budget)[1]
    if sign > 0:
        return 'above', (_qsub(_qdiv(pi[0], Fraction(2), budget), upper_atan, budget),
                         _qsub(_qdiv(pi[1], Fraction(2), budget), lower_atan, budget))
    return 'below', (_qsub(_qneg(_qdiv(pi[1], Fraction(2), budget), budget), upper_atan, budget),
                     _qsub(_qneg(_qdiv(pi[0], Fraction(2), budget), budget), lower_atan, budget))


def _serialize_decimal(value, places, budget):
    _validate_rational(value, budget)
    if type(places) is not int or places < 0:
        raise InvalidInput('decimal places must be a nonnegative integer')
    if places > budget.limits.decimal_places:
        raise BudgetExceeded('decimal places')
    # 10**places < 2**(4*places), with a conservative extra bit.
    budget.check_rational_bits(4 * places + 1)
    budget.charge_work()
    scale = 10**places
    scaled = _qinteger_product(value.numerator, scale, budget)
    budget.charge_work()  # divmod on bounded integers
    quotient, remainder = divmod(abs(scaled), value.denominator)
    budget.check_rational_bits(remainder.bit_length() + 1 if remainder else 0)
    twice = _checked_shift(remainder, 1, budget)
    if twice > value.denominator or (twice == value.denominator and quotient & 1):
        quotient = _checked_add(quotient, 1, budget)
    budget.charge_work()  # divmod on bounded integers
    whole, fractional = divmod(quotient, scale)
    negative = scaled < 0 and quotient != 0
    # Upper bound on decimal digits avoids constructing an unbounded string.
    whole_digits = whole.bit_length() * 30103 // 100000 + 1
    import sys
    python_limit = getattr(sys, 'get_int_max_str_digits', lambda: 0)()
    if python_limit and (whole_digits > python_limit or places > python_limit):
        raise BudgetExceeded('decimal integer digits')
    chars = whole_digits + (places + 1 if places else 0) + int(negative)
    budget.reserve_output(chars=chars)
    text = ('-' if negative else '') + str(whole)
    if places:
        text += f'.{fractional:0{places}d}'
    serialized = _fraction(-quotient if negative else quotient, scale, budget)
    return text, serialized


@dataclass(frozen=True)
class DiagonalCertificate:
    angles: tuple[str, str]
    branches: tuple[str, str]
    error_bound: Fraction


def certify_diagonal_power(matrix, exponent, goal, budget, *, phase_bits=48, decimal_places=24):
    """Certify exact-real serialized angles for an exactly diagonal unitary.

    Exponent and error goal are exact Fractions. The bound excludes bra parsing
    and execution. Resource checks are conservative and may reject a result
    that would fit after cancellation. No source or target format is inferred.
    """
    if not isinstance(budget, Budget):
        raise InvalidInput('expected a shared Budget')
    _validate_rational(exponent, budget)
    _validate_rational(goal, budget)
    if any(value.bit_length() > budget.limits.input_bits for value in
           (exponent.numerator, exponent.denominator, goal.numerator, goal.denominator)):
        raise BudgetExceeded('input bits')
    if goal.numerator <= 0 or _qcompare(goal, Fraction(1), budget) >= 0:
        raise InvalidInput('error goal must be strictly between zero and one')
    if type(phase_bits) is not int or phase_bits < 1:
        raise InvalidInput('phase bits must be a positive integer')
    if type(decimal_places) is not int or decimal_places < 0:
        raise InvalidInput('decimal places must be a nonnegative integer')
    if decimal_places > budget.limits.decimal_places:
        raise BudgetExceeded('decimal places')
    if classify_matrix(matrix, budget) == 'non-diagonal':
        raise InvalidInput('expected an exactly diagonal matrix')
    for value in (matrix[0][0], matrix[1][1]):
        conjugate = _make_exact((value.a, value.b, -value.c, -value.d), value.k, budget)
        if _exact_multiply(value, conjugate, budget) != ONE:
            raise InvalidInput('expected an exactly unitary matrix')

    # Guard digits count against the precision budget as well.
    maximum = budget.limits.precision_bits - 16
    if maximum < 8 or phase_bits > maximum:
        raise BudgetExceeded('precision bits')
    precision = min(maximum, max(8, phase_bits + 16))
    width = _fraction(1, _pow2(phase_bits, budget), budget)
    while True:
        budget.charge_work()  # Every refinement, including failed attempts.
        try:
            root2 = _sqrt2_interval(precision, budget)
            pi = _pi_interval(precision + 8, budget)
            phase0 = _principal_phase(matrix[0][0], precision + 8, root2, pi, budget)
            phase1 = _principal_phase(matrix[1][1], precision + 8, root2, pi, budget)
            phases = (phase0, phase1)
            if all(_qcompare(_qsub(bounds[1], bounds[0], budget), width, budget) <= 0
                   for _, bounds in phases):
                break
        except _NeedPrecision:
            pass
        if precision >= maximum:
            raise BudgetExceeded('precision bits')
        precision = min(precision * 2, maximum)

    angles = []
    errors = []
    absolute_exponent = _qneg(exponent, budget) if exponent.numerator < 0 else exponent
    for _, (lower, upper) in phases:
        midpoint = _qdiv(_qadd(lower, upper, budget), Fraction(2), budget)
        radius = _qdiv(_qsub(upper, lower, budget), Fraction(2), budget)
        ideal_approximation = _qmul(exponent, midpoint, budget)
        text, serialized = _serialize_decimal(ideal_approximation, decimal_places, budget)
        serialization = _qsub(serialized, ideal_approximation, budget)
        if serialization.numerator < 0:
            serialization = _qneg(serialization, budget)
        errors.append(_qadd(_qmul(absolute_exponent, radius, budget), serialization, budget))
        angles.append(text)
    bound = errors[0] if _qcompare(errors[0], errors[1], budget) >= 0 else errors[1]
    if _qcompare(bound, goal, budget) > 0:
        raise AccuracyNotMet('operator error bound exceeds requested goal')
    return DiagonalCertificate(tuple(angles), tuple(branch for branch, _ in phases), bound)
