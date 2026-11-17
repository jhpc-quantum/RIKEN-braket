#!/usr/bin/env python3
"""Check runtime-powered user gates against phase-sensitive state vectors."""

import argparse
import math
import pathlib
import resource

from qasm2qcx_inverse_numerical import apply_gate, check_amplitudes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for exponent in (-3, 0, 2):
        state = [0j] * 32
        state[0] = 1
        state = apply_gate(state, 'h', (), (0,))
        for _ in range(abs(exponent)):
            sequence = (('h', ()), ('p', (math.pi / 4,))) if exponent > 0 else (
                ('p', (-math.pi / 4,)), ('h', ()))
            for gate, parameters in sequence:
                state = apply_gate(state, gate, parameters, (0,))
        check_amplitudes(bra, f'''gate g(t) a {{ h a; p(t) a; }}
            qubit[5] q; angle[3] a=pi/4; int n={exponent};
            h q[0]; pow(n) @ g(a) q[0];''', state)


if __name__ == '__main__':
    main()
