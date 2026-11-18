#!/usr/bin/env python3
"""Run controlled QCX examples using the unintegrated maintained foundation."""

import argparse
import cmath
from fractions import Fraction
import math
from pathlib import Path
import re
import resource
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qcx'))
import certified_diagonal as diagonal


def check_case(bra, names, expected_active):
    budget = diagonal.Budget()
    matrix = diagonal.compose_fixed_gates(names, budget)
    certificate = diagonal.certify_diagonal_power(matrix, Fraction(1, 2), Fraction(1, 10**12), budget)
    # In production, the caller must reserve these output lines on the same
    # budget before emitting them; this script illustrates that boundary.
    budget.reserve_output(lines=2)
    qcx = '\n'.join(('QUBITS 5', 'H 0', 'H 1', f'CU1 0 1#0 {certificate.angles[0]}',
                     f'CU1 0 1 {certificate.angles[1]}', 'DO AMPLITUDES', ''))
    result = subprocess.run([str(bra)], input=qcx, text=True, capture_output=True, check=True, timeout=30)
    actual = {int(index, 2): complex(float(real), float(imag)) for index, real, imag in
              re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
    expected = [0j] * 32
    expected[0] = expected[2] = 0.5
    expected[1], expected[3] = (0.5 * value for value in expected_active)
    if set(actual) != set(range(32)) or any(
            not cmath.isclose(actual[index], value, rel_tol=1e-5, abs_tol=2e-6)
            for index, value in enumerate(expected)):
        raise RuntimeError(f'Controlled diagonal mismatch\nQCX:\n{qcx}\n{result.stdout}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    check_case(bra, ('t',) * 4, (1, 1j))
    root = cmath.exp(0.25j * math.pi)
    check_case(bra, ('z', 'y', 'x'), (root, root))
    check_case(bra, ('t',), (1, cmath.exp(0.125j * math.pi)))
    check_case(bra, ('tdg',), (1, cmath.exp(-0.125j * math.pi)))


if __name__ == '__main__':
    main()
