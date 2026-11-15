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
    state = [0j] * 32
    state[0] = 1
    for gate, angle in (('rx', math.pi / 4), ('ry', 7 * math.pi / 4),
                        ('rz', 5 * math.pi / 4)):
        state = apply_gate(state, gate, (angle,), (0,))
    check_amplitudes(bra, '''gate inner(theta,phi) q {
        rx(theta+phi) q; ry(-(theta+phi)) q; rz(theta-phi) q; }
        gate outer(a,b) q { inner(a,b) q; }
        qubit[5] q; angle[3] x=7*pi/4; angle[5] y=pi/2;
        outer(x,y) q[0];''', state)


if __name__ == '__main__':
    main()
