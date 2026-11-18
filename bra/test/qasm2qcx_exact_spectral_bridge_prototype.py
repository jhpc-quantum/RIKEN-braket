#!/usr/bin/env python3
"""Symbolic endpoint spectra for the #634 feasibility study.

The result retains exact eigenvalues and an unexpanded projector fraction.
It does not evaluate non-endpoint phases or emit QCX.
"""

from dataclasses import dataclass

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


def endpoint_spectrum(matrix):
    """Return an exact scalar or -1 endpoint spectrum for a unitary matrix.

    For a nonscalar matrix with eigenvalues -1 and lambda, the endpoint
    projector is (G - lambda*I)/(-1 - lambda). The quotient stays symbolic:
    division need not remain in the exact-entry ring.
    """
    if (len(matrix) != 2 or any(len(row) != 2 for row in matrix)
            or any(not isinstance(value, exact.Exact) for row in matrix for value in row)):
        raise ValueError('expected a 2x2 matrix of exact values')
    adjoint = tuple(tuple(matrix[column][row].conjugate() for column in range(2))
                    for row in range(2))
    if exact.multiply_matrices(adjoint, matrix) != exact.GATES['id']:
        raise ValueError('expected an exactly unitary matrix')
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
