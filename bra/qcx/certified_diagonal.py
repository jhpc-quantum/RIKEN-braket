"""Bounded exact arithmetic for future diagonal-gate certification.

This module is not yet used by qasm2qcx. Bounds count explicitly modeled work
and integer sizes; they do not constitute a process-memory or parser budget.
"""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class BudgetLimits:
    # Candidate defaults from the design; production adoption needs review.
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
