#!/usr/bin/env python3
"""Conditional operation-error bounds for the #636 precision study.

Inputs are assumed proven by a caller; this module does not certify them or
produce QCX. Arithmetic in the bound itself uses exact rational numbers.
"""

from dataclasses import dataclass
from fractions import Fraction

from qasm2qcx_spectral_target_prototype import AccuracyRequest


PI_UPPER = Fraction(22, 7)  # strictly greater than pi
BRANCHES = frozenset(('above', 'below', 'positive-real', 'negative-endpoint'))


def _nonnegative(value, name):
    if not isinstance(value, Fraction):
        raise TypeError(f'{name} must be an exact Fraction')
    if value < 0:
        raise ValueError(f'{name} must be nonnegative')


@dataclass(frozen=True)
class PhaseErrorAssumption:
    """Caller-supplied bounds for a labeled, already-certified branch.

    phase_error bounds the chosen unpowered principal phase; output_error
    bounds all later angle arithmetic and serialization for that phase.
    """
    label: str
    branch: str
    phase_error: Fraction
    output_error: Fraction

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label:
            raise ValueError('eigenvalue label must be nonempty')
        if self.branch not in BRANCHES:
            raise ValueError('unsupported branch label')
        _nonnegative(self.phase_error, 'phase_error')
        _nonnegative(self.output_error, 'output_error')


@dataclass(frozen=True)
class SpectralErrorAssumptions:
    """Conditional inputs for a V diag(exp(i alpha_j)) V-dagger bound.

    basis_labels identify the columns of V in order. Errors in beta and eta
    include all basis calculation and serialization error. The exponent is
    the value actually used to form the emitted angles; exponent_error bounds
    its distance from the ideal exponent. Labels and all bounds must already
    have been certified externally.
    """
    phases: tuple[PhaseErrorAssumption, PhaseErrorAssumption]
    basis_labels: tuple[str, str]
    exponent: Fraction
    exponent_error: Fraction
    beta_error: Fraction
    eta_error: Fraction

    def __post_init__(self):
        if (not isinstance(self.phases, tuple) or len(self.phases) != 2
                or not all(isinstance(phase, PhaseErrorAssumption) for phase in self.phases)):
            raise TypeError('two phase-error assumptions are required')
        labels = tuple(phase.label for phase in self.phases)
        if labels[0] == labels[1] or self.basis_labels != labels:
            raise ValueError('distinct eigenvalue labels must match the basis column order')
        if not isinstance(self.exponent, Fraction):
            raise TypeError('exponent must be an exact Fraction')
        for name in ('exponent_error', 'beta_error', 'eta_error'):
            _nonnegative(getattr(self, name), name)


@dataclass(frozen=True)
class ConditionalBound:
    basis: Fraction
    phases: tuple[Fraction, Fraction]
    total: Fraction
    meets_request: bool


def assess_bound(request: AccuracyRequest, assumptions: SpectralErrorAssumptions) -> ConditionalBound:
    """Bound operator error if every supplied assumption is independently true.

    For the conventional U3(beta, eta, 0) eigenbasis, a telescoping bound is
    ||V~-V|| <= |delta beta|/2 + |delta eta|. Both V and V-dagger contribute.
    Each diagonal error is at most its angle error. Ideal principal phases
    have magnitude <= pi, so using 22/7 gives an exact rational bound on the
    exponent-error contribution. This includes full phase, hence it also
    bounds a controlled operation whose inactive block is exactly identity.
    """
    if not isinstance(request, AccuracyRequest) or not isinstance(assumptions, SpectralErrorAssumptions):
        raise TypeError('an accuracy request and spectral assumptions are required')
    basis = assumptions.beta_error + 2 * assumptions.eta_error
    phases = tuple(abs(assumptions.exponent) * phase.phase_error
                   + PI_UPPER * assumptions.exponent_error + phase.output_error
                   for phase in assumptions.phases)
    total = basis + max(phases)
    return ConditionalBound(basis, phases, total, total <= request.max_operator_error)
