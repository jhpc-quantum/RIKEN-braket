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
    for gate in ('crx', 'cry', 'crz'):
        for preparation in ('', 'x', 'h'):
            state = [0j] * 32
            state[0] = 1
            if preparation:
                state = apply_gate(state, preparation, (), (0,))
            state = apply_gate(state, 'h', (), (1,))
            state = apply_gate(state, gate, (7 * math.pi / 4,), (0, 1))
            prepare = f'{preparation} q[0];' if preparation else ''
            check_amplitudes(bra, f'''qubit[5] q; angle[3] a=7*pi/4;
                {prepare} h q[1]; {gate}(a) q[0],q[1];''', state)
    state = [0j] * 32
    state[0] = 1
    state = apply_gate(state, 'h', (), (0,))
    for gate, angle in (('crx', math.pi / 4), ('cry', 7 * math.pi / 4),
                        ('crz', 5 * math.pi / 4)):
        state = apply_gate(state, gate, (angle,), (0, 1))
    check_amplitudes(bra, '''gate inner(theta,phi) c,t {
        crx(theta+phi) c,t; cry(-(theta+phi)) c,t; crz(theta-phi) c,t; }
        gate outer(a,b) c,t { inner(a,b) c,t; }
        qubit[5] q; angle[3] x=7*pi/4; angle[5] y=pi/2;
        h q[0]; outer(x,y) q[0],q[1];''', state)
    state = [0j] * 32
    state[0] = 1
    for control, target in ((0, 2), (1, 3)):
        state = apply_gate(state, 'h', (), (control,))
        state = apply_gate(state, 'crx', (math.pi / 2,), (control, target))
    check_amplitudes(bra, '''qubit[2] c; qubit[2] t; qubit spare;
        angle[3] a=pi/2; h c; crx(a) c,t;''', state)


if __name__ == '__main__':
    main()
