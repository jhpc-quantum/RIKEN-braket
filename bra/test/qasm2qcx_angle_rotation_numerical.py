#!/usr/bin/env python3
"""Check angle-valued rotations against independent matrices."""

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
    for gate in ('rx', 'ry', 'rz'):
        for value, angle in (('0.0', 0), ('pi', math.pi), ('7*pi/4', 7*math.pi/4)):
            state = [0j] * 32
            state[0] = 1
            state = apply_gate(state, 'h', (), (0,))
            state = apply_gate(state, gate, (angle,), (0,))
            check_amplitudes(bra, f'''qubit[5] q; angle[3] a={value};
                h q[0]; {gate}(a) q[0];''', state)


if __name__ == '__main__':
    main()
