#!/usr/bin/env python3
"""Run controlled QCX examples from the exact-diagonal bound prototype."""

import argparse
import cmath
from fractions import Fraction
import math
import pathlib
import re
import resource
import subprocess

import qasm2qcx_certified_diagonal_prototype as diagonal
import qasm2qcx_exact_single_qubit_prototype as exact
import qasm2qcx_spectral_target_prototype as target


def run_case(bra, matrix, active_phases):
    request = target.AccuracyRequest(target.BINARY64, Fraction(1, 10**12))
    result = diagonal.certify_diagonal_power(matrix, Fraction(1, 2), request)
    qcx = '\n'.join(('QUBITS 5', 'H 0', 'H 1', f'CU1 0 1#0 {result.angles[0]}',
                     f'CU1 0 1 {result.angles[1]}', 'DO AMPLITUDES', ''))
    process = subprocess.run([str(bra)], input=qcx, text=True, capture_output=True, check=True, timeout=30)
    actual = {int(index, 2): complex(float(real), float(imag)) for index, real, imag in
              re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', process.stdout, re.MULTILINE)}
    expected = [0j] * 32
    expected[0] = expected[2] = 0.5
    expected[1], expected[3] = (0.5 * value for value in active_phases)
    if set(actual) != set(range(32)) or any(
            not cmath.isclose(actual[index], value, rel_tol=1e-5, abs_tol=2e-6)
            for index, value in enumerate(expected)):
        raise RuntimeError(f'Controlled diagonal mismatch\nQCX:\n{qcx}\n{process.stdout}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    run_case(bra, exact.compose_sequence(('t',) * 4), (1, 1j))
    scalar = ((exact.I, exact.ZERO), (exact.ZERO, exact.I))
    root = cmath.exp(0.25j * math.pi)
    run_case(bra, scalar, (root, root))
    # Distinct exact non-endpoint phases on opposite sides of the cut.
    for value, sign in ((-exact.OMEGA, -1), (-exact.OMEGA.conjugate(), 1)):
        matrix = ((exact.ONE, exact.ZERO), (exact.ZERO, value))
        run_case(bra, matrix, (1, cmath.exp(sign * 0.375j * math.pi)))


if __name__ == '__main__':
    main()
