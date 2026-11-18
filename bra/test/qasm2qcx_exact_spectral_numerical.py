#!/usr/bin/env python3
"""Experimental bridge from certified branches to QCX spectral angles.

Only modest, well-separated test matrices are supported. Angle and basis
calculations below use ordinary binary64 and are NOT certified error bounds.
The exact endpoint decision is retained rather than rediscovered from floats.
"""

import argparse
import cmath
import math
import pathlib
import re
import resource
import subprocess

import qasm2qcx_exact_single_qubit_prototype as exact
import qasm2qcx_exact_spectral_bridge_prototype as bridge


def _midpoint(interval):
    return float((interval[0] + interval[1]) / 2)


def approximate_eigenvalue(root):
    if root.branch == 'negative-endpoint':
        return -1 + 0j
    if root.branch == 'positive-real':
        return 1 + 0j
    value = complex(_midpoint(root.real), _midpoint(root.imaginary))
    # The branch is certified independently; do not use a rounded zero or
    # opposite sign to invent an angle on one side of the cut.
    if (root.branch == 'above' and value.imag <= 0) or (root.branch == 'below' and value.imag >= 0):
        raise ValueError('binary64 eigenvalue approximation lost its certified side')
    return value


def approximate_powered_spectrum(matrix, exponent):
    """Return beta, eta and two phases; this is not certified lowering."""
    if not math.isfinite(exponent):
        raise ValueError('power must be finite')
    roots = bridge.certify_eigenvalue_branches(matrix, max_bits=128)
    values = tuple(approximate_eigenvalue(root) for root in roots)
    phases = tuple(exponent * (math.pi if root.branch == 'negative-endpoint' else cmath.phase(value))
                   for root, value in zip(roots, values))
    if not all(math.isfinite(phase) for phase in phases):
        raise ValueError('powered phase is not finite')
    if exact.scalar_value(matrix) is not None:
        return 0.0, 0.0, *phases
    # Reject nearly degenerate approximate eigenvectors; this threshold limits
    # the experiment and does not classify or merge eigenvalues.
    if abs(values[0] - values[1]) < 1e-7:
        raise ValueError('eigenvalue separation too small for the binary64 experiment')
    (a, b), (c, d) = tuple(tuple(value.to_complex() for value in row) for row in matrix)
    first = (b, values[0] - a)
    second = (values[0] - d, c)
    vector = max((first, second), key=lambda v: abs(v[0]) ** 2 + abs(v[1]) ** 2)
    if vector == (0j, 0j):
        raise ValueError('binary64 eigenvector is unavailable')
    beta = 2 * math.atan2(abs(vector[1]), abs(vector[0]))
    eta = cmath.phase(vector[1]) - cmath.phase(vector[0]) if vector[0] and vector[1] else 0.0
    return beta, eta, *phases


def powered_matrix(matrix, exponent):
    beta, eta, phase0, phase1 = approximate_powered_spectrum(matrix, exponent)
    c, s = math.cos(beta / 2), math.sin(beta / 2)
    e = cmath.exp(1j * eta)
    basis = ((c, -s), (e * s, e * c))
    diagonal = ((cmath.exp(1j * phase0), 0j), (0j, cmath.exp(1j * phase1)))
    adjoint = tuple(tuple(basis[column][row].conjugate() for column in range(2)) for row in range(2))
    def multiply(left, right):
        return tuple(tuple(sum(left[row][k] * right[k][column] for k in range(2))
                           for column in range(2)) for row in range(2))
    return multiply(multiply(basis, diagonal), adjoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Four T gates are exactly Z. The control is in superposition, so an
    # incorrect global phase or branch changes the measured amplitudes.
    matrix = exact.compose_sequence(('t',) * 4)
    beta, eta, phase0, phase1 = approximate_powered_spectrum(matrix, 0.5)
    qcx = '\n'.join(('QUBITS 5', 'H 0', 'H 1', f'U3+ 1 {beta} {eta} 0.0',
                     f'CU1 0 1#0 {phase0}', f'CU1 0 1 {phase1}',
                     f'U3 1 {beta} {eta} 0.0', 'DO AMPLITUDES', ''))
    result = subprocess.run([str(bra)], input=qcx, text=True, capture_output=True, check=True, timeout=30)
    actual = {int(index, 2): complex(float(real), float(imag)) for index, real, imag in
              re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
    expected = [0j] * 32
    expected[0] = expected[1] = expected[2] = 0.5
    expected[3] = 0.5j
    if set(actual) != set(range(32)) or any(
            not cmath.isclose(actual[index], value, rel_tol=1e-5, abs_tol=2e-6)
            for index, value in enumerate(expected)):
        raise RuntimeError(f'Controlled endpoint mismatch\nQCX:\n{qcx}\n{result.stdout}')


if __name__ == '__main__':
    main()
