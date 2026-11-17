#!/usr/bin/env python3
"""Execute the experimental controlled spectral sequence using bra."""

import argparse
import cmath
import math
import pathlib
import re
import resource
import subprocess

from qasm2qcx_single_qubit_composition_prototype import compose_sequence, native_principal_argument
from test_qasm2qcx_single_qubit_power_reference import sequence_matrix
from test_qasm2qcx_single_qubit_spectral_prototype import direct_matrix_power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    names = ('h', 's')
    unitary = compose_sequence(names)
    x, y, z = unitary.vector
    radial = math.hypot(x, y)
    length = math.hypot(radial, z)
    beta = math.atan2(radial, z) if length else 0.0
    eta = math.atan2(y, x) if radial else 0.0
    plus = unitary.phase * complex(unitary.scalar, -length)
    minus = unitary.phase * complex(unitary.scalar, length)
    alpha_plus = 0.5 * native_principal_argument(plus)
    alpha_minus = 0.5 * native_principal_argument(minus)
    qcx = '\n'.join(('QUBITS 5', 'H 0', 'H 1', f'U3+ 1 {beta} {eta} 0.0',
                     f'CU1 0 1#0 {alpha_plus}', f'CU1 0 1 {alpha_minus}',
                     f'U3 1 {beta} {eta} 0.0', 'DO AMPLITUDES', ''))
    result = subprocess.run([str(bra)], input=qcx, text=True, capture_output=True, check=True, timeout=30)
    actual = {int(index, 2): complex(float(real), float(imag)) for index, real, imag in
              re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
    matrix = direct_matrix_power(sequence_matrix(names), 0.5)
    expected = [0j] * 32
    expected[0] = expected[2] = 0.5
    expected[1] = (matrix[0][0] + matrix[0][1]) / 2
    expected[3] = (matrix[1][0] + matrix[1][1]) / 2
    if set(actual) != set(range(32)) or any(
            not cmath.isclose(actual[index], value, rel_tol=1e-5, abs_tol=2e-6)
            for index, value in enumerate(expected)):
        raise RuntimeError(f'Controlled spectral result mismatch\nQCX:\n{qcx}\n{result.stdout}')


if __name__ == '__main__':
    main()
