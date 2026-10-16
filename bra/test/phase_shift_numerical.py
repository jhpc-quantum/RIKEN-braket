#!/usr/bin/env python3

"""Check phase-exponent gates independently of the state-sized coefficient cache.

Example:
  python3 bra/test/phase_shift_numerical.py --bra bra/bin/bra
"""

import argparse
import cmath
import ctypes
import math
import pathlib
import re
import subprocess


def check_phase(bra: pathlib.Path, qubits: int, instruction: str,
                operated: tuple[int, ...], exponent: int, adjoint: bool = False,
                fused: bool = False, declaration: str = '') -> None:
    lines = [f'QUBITS {qubits}']
    for index in range(qubits):
        lines.extend((f'H {index}', f'U1 {index} {0.17 * (index + 1)}'))
    if declaration:
        lines.extend(('VAR K INT', f'LET K := {declaration}'))
    if fused:
        lines.append('BEGIN FUSION')
    lines.append(instruction)
    if fused:
        lines.append('END FUSION')
    lines.append('DO AMPLITUDES')
    result = subprocess.run([str(bra)], input='\n'.join(lines) + '\n', check=True,
                            text=True, capture_output=True, timeout=30)
    actual = {int(index, 2): complex(float(real), float(imaginary)) for index, real, imaginary in
              re.findall(r'^([01]+) => (\S+) \+ (\S+) i$', result.stdout, re.MULTILINE)}
    # A negative exponent means the adjoint phase, not 2*pi*2**abs(k).
    angle = math.ldexp(2 * math.pi, -abs(exponent))
    if (exponent < 0) != adjoint:
        angle = -angle
    mask = sum(1 << index for index in operated)
    expected = []
    for basis in range(1 << qubits):
        preparation = sum(0.17 * (index + 1) for index in range(qubits) if basis & (1 << index))
        phase = preparation + (angle if basis & mask == mask else 0)
        expected.append(cmath.exp(1j * phase) / math.sqrt(1 << qubits))
    if set(actual) != set(range(len(expected))) or any(
            not cmath.isclose(actual[index], value, rel_tol=1e-5, abs_tol=2e-6)
            for index, value in enumerate(expected)):
        raise RuntimeError(f'Phase-shift mismatch: qubits={qubits}, fused={fused}\n'
                           f'{instruction}\nexpected: {expected}\nstdout:\n{result.stdout}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    width = ctypes.sizeof(ctypes.c_int) * 8
    minimum, maximum = -(1 << (width - 1)), (1 << (width - 1)) - 1
    for qubits in (1, 2, 3, 4):
        for fused in (False, True):
            for count in range(1, min(qubits, 3) + 1):
                operated = tuple(range(qubits - count, qubits))
                operands = ' '.join(map(str, operated))
                prefix = '' if count == 1 else 'C' if count == 2 else f'C{count - 1}'
                for name, exponent in (('S', 2), ('T', 3)):
                    for adjoint in (False, True):
                        check_phase(bra, qubits, prefix + name + ('+' if adjoint else '') + ' ' + operands,
                                    operated, exponent, adjoint, fused)
                # Cover both cache hits and misses, signed exponents, the
                # integer-shift boundary, underflow, and native INT endpoints.
                exponents = sorted({0, 1, -1, 2, -2, 3, -3, qubits, qubits + 1, -(qubits + 1),
                                    9, -9, 63, -63, 64, -64, 1075, -1075, minimum, maximum})
                for exponent in exponents:
                    for adjoint in (False, True):
                        name = prefix + 'R' + ('+' if adjoint else '')
                        check_phase(bra, qubits, f'{name} {operands} {exponent}', operated,
                                    exponent, adjoint, fused)
                        check_phase(bra, qubits, f'{name} {operands} K', operated,
                                    exponent, adjoint, fused, str(exponent))
    print('Phase-shift numerical checks passed')


if __name__ == '__main__':
    main()
