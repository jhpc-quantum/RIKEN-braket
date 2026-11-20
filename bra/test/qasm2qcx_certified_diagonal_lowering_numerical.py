#!/usr/bin/env python3
"""Check converter-generated diagonal powers against independent amplitudes.

Run with: python3 bra/test/qasm2qcx_certified_diagonal_lowering_numerical.py --bra bra/bin/bra
These checks exercise one bra build; they do not certify its numerical error.
"""

import argparse
import cmath
import math
from pathlib import Path
import re
import resource
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qcx'))
import qasm2qcx


DEFINITIONS = '''
gate t2 a { t a; t a; }
gate z_nested a { t2 a; t2 a; }
gate minus_i a { t2 a; t2 a; t2 a; }
gate scalar_i a { z a; y a; x a; }
gate identity a { h a; h a; }
gate empty a {}
'''


def check_case(bra, call, phases, exponent, controls, targets, fused):
    source = ('OPENQASM 3; include "stdgates.inc"; ' + DEFINITIONS
              + 'qubit[5] q; h q; ' + call)
    lines = qasm2qcx.convert(source, diagonal_power_error='1e-12')
    if fused:
        lines.insert(1, 'BEGIN FUSION')
        lines.append('END FUSION')
    qcx = '\n'.join(lines + ['DO AMPLITUDES', ''])
    result = subprocess.run([str(bra)], input=qcx, text=True, capture_output=True,
                            check=True, timeout=30)
    actual = {int(index, 2): complex(float(real), float(imag)) for index, real, imag in
              re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
    if set(actual) != set(range(32)):
        raise RuntimeError(f'Missing amplitudes for {call}\nQCX:\n{qcx}\n{result.stdout}')
    for index in range(32):
        enabled = all((index >> wire) & 1 == value for wire, value in controls)
        phase = sum(phases[(index >> target) & 1] * exponent for target in targets) if enabled else 0
        expected = cmath.exp(1j * phase) / math.sqrt(32)
        if not cmath.isclose(actual[index], expected, rel_tol=1e-5, abs_tol=2e-6):
            raise RuntimeError(f'Amplitude {index} mismatch for {call}, fusion={fused}: '
                               f'{actual[index]} != {expected}\nQCX:\n{qcx}\n{result.stdout}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cases = (
        ('pow(0.5) @ z_nested q[1];', (0, math.pi), 0.5, (), (1,)),
        ('ctrl @ pow(0.5) @ z_nested q[0],q[1];', (0, math.pi), 0.5, ((0, 1),), (1,)),
        ('negctrl @ pow(0.5) @ scalar_i q[0],q[1];',
         (math.pi / 2, math.pi / 2), 0.5, ((0, 0),), (1,)),
        ('ctrl @ negctrl @ pow(-0.5) @ minus_i q[0],q[1],q[2];',
         (0, -math.pi / 2), -0.5, ((0, 1), (1, 0)), (2,)),
        ('pow(0.1) @ t2 q[{1,3}];', (0, math.pi / 2), 0.1, (), (1, 3)),
        ('pow(0.0) @ z_nested q[1];', (0, math.pi), 0.0, (), (1,)),
        ('pow(-0.5) @ identity q[1];', (0, 0), -0.5, (), (1,)),
        ('pow(2.0) @ empty q[1];', (0, 0), 2.0, (), (1,)),
    )
    for case in cases:
        for fused in (False, True):
            check_case(bra, *case, fused)
    print(f'Passed {2 * len(cases)} converter-generated diagonal power cases')


if __name__ == '__main__':
    main()
