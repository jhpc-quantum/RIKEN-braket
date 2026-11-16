#!/usr/bin/env python3
"""Check angle-valued rotations against independent matrices."""

import argparse
import cmath
import ctypes
import math
import pathlib
import resource

from qasm2qcx_inverse_numerical import apply_gate, check_amplitudes
from anglebits2phase_reference import decode_chain, evaluate_turns


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
            inverse = apply_gate(apply_gate([1] + [0j] * 31, 'h', (), (0,)),
                                 gate, (angle,), (0,), inverse=True)
            check_amplitudes(bra, f'''qubit[5] q; angle[3] a={value};
                h q[0]; inv @ {gate}(a) q[0];''', inverse)
        state = [0j] * 32
        state[0] = 1
        state = apply_gate(state, 'h', (), (0,))
        state = apply_gate(state, 'c' + gate, (2 * math.pi,), (0, 1))
        check_amplitudes(bra, f'''qubit[5] q; angle[3] a=pi; h q[0];
            pow(2) @ c{gate}(a) q[0],q[1];''', state)
        for exponent in (-3, 0, 19, 1048577):
            state = [0j] * 32
            state[0] = 1
            state = apply_gate(state, 'h', (), (0,))
            state = apply_gate(state, gate, ((exponent % 16) * math.pi / 4,), (0,))
            check_amplitudes(bra, f'''qubit[5] q; angle[3] a=pi/4; h q[0];
                pow({exponent}) @ {gate}(a) q[0];''', state)
    width = ctypes.sizeof(ctypes.c_uint) * 8
    state = [0j] * 32
    state[0] = 1
    state = apply_gate(state, 'h', (), (0,))
    expected = [1j * amplitude for amplitude in state]
    check_amplitudes(bra, '''qubit[5] q; angle[3] a=pi; h q[0];
        pow(0.5) @ pow(2) @ rx(a) q[0];''', expected)
    state = apply_gate(state, 'crz', (2 * math.pi,), (0, 1))
    check_amplitudes(bra, f'''qubit[5] q; angle[{width}] a=pi; h q[0];
        pow(2) @ crz(a) q[0],q[1];''', state)
    for exponent in (-3, 0, 2, 5):
        state = [0j] * 32
        state[0] = 1
        state = apply_gate(state, 'h', (), (0,))
        state = apply_gate(state, 'crx', ((exponent % 16) * math.pi / 4,), (0, 1))
        check_amplitudes(bra, f'''qubit[5] q; angle[3] a=pi/4; int k={exponent};
            h q[0]; pow(k) @ crx(a) q[0],q[1];''', state)
    state = [0j] * 32
    state[0] = 1
    state = apply_gate(state, 'h', (), (0,))
    state = apply_gate(state, 'crz', (2 * math.pi,), (0, 1))
    check_amplitudes(bra, f'''qubit[5] q; angle[{width}] a=pi; int k=2;
        h q[0]; pow(k) @ crz(a) q[0],q[1];''', state)
    state = [0j] * 32
    state[0] = 1
    state = apply_gate(state, 'h', (), (0,))
    state = apply_gate(state, 'crx', (9 * math.pi / 4,), (0, 1))
    check_amplitudes(bra, f'''qubit[5] q; angle[{width}] a=7*pi/4;
        uint k={(1 << width) - 1}; h q[0]; pow(k) @ crx(a) q[0],q[1];''', state)
    int_width = ctypes.sizeof(ctypes.c_int) * 8
    int_max = (1 << (int_width - 1)) - 1
    angle = -math.ldexp(2 * math.pi, int_width - 1 - width)
    state = [0j] * 32
    state[0] = 1
    state = apply_gate(state, 'h', (), (0,))
    state = apply_gate(state, 'cry', (angle,), (0, 1))
    check_amplitudes(bra, f'''qubit[5] q; angle[{width}] a=2*pi/{1 << width}.0;
        int k=-{int_max}-1; h q[0]; pow(k) @ cry(a) q[0],q[1];''', state)
    # Each factor fits UINT, but their product exceeds native UINT width.
    first = (1 << (width // 2)) - 1
    second = (1 << (width // 2)) + 3
    powered_angle = ((first * second) % 16) * math.pi / 4
    state = [0j] * 32
    state[0] = 1
    state = apply_gate(state, 'h', (), (0,))
    state = apply_gate(state, 'crz', (powered_angle,), (0, 1))
    check_amplitudes(bra, f'''qubit[5] q; angle[3] a=pi/4;
        uint first={first}; uint second={second}; h q[0];
        pow(first) @ pow(second) @ crz(a) q[0],q[1];''', state)
    state = [0j] * 64
    state[0] = 1
    for qubit in (0, 1, 2):
        state = apply_gate(state, 'h', (), (qubit,))
    rotated = apply_gate(state, 'rx', (3 * math.pi / 2,), (2,))
    expected = [rotated[index] if index & 3 == 2 else value
                for index, value in enumerate(state)]
    check_amplitudes(bra, '''qubit[6] q; angle[3] a=pi/4; int k=-3; uint u=2;
        h q[0]; h q[1]; h q[2];
        negctrl @ inv @ pow(k) @ pow(u) @ crx(a) q[0],q[1],q[2];''', expected)
    state = [0j] * 64
    state[0] = 1
    for qubit in (0, 1, 2):
        state = apply_gate(state, 'h', (), (qubit,))
    rotated = apply_gate(state, 'ry', (3 * math.pi / 4,), (2,))
    expected = [rotated[index] if index & 3 == 2 else value
                for index, value in enumerate(state)]
    check_amplitudes(bra, '''gate inner(t,u) a,b,c {
        negctrl @ inv @ pow(-3) @ cry(t+u) a,b,c; }
        gate outer(t,u) a,b,c { inner(t,u) a,b,c; }
        qubit[6] q; angle[3] x=7*pi/4; angle[5] y=pi/2;
        h q[0]; h q[1]; h q[2]; outer(x,y) q[0],q[1],q[2];''', expected)
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
    chain = decode_chain(('POW', '2', 'POW', '0.5'))
    for width, bits in ((4, 7), (4, 8), (4, 9), (width, (1 << (width - 1)) + 1)):
        plus = evaluate_turns(bits, width, chain, half_sign=-1)
        minus = evaluate_turns(bits, width, chain, half_sign=1)
        eigenplus, eigenminus = cmath.exp(2j * math.pi * float(plus)), cmath.exp(2j * math.pi * float(minus))
        for gate, pauli in (('crx', 'x'), ('cry', 'y'), ('crz', 'z')):
            state = [0j] * 64
            state[0] = 1
            for qubit in (0, 1, 2):
                state = apply_gate(state, 'h', (), (qubit,))
            transformed = apply_gate(state, pauli, (), (2,))
            expected = [((eigenplus + eigenminus) * value + (eigenplus - eigenminus) * transformed[index]) / 2
                        if index & 3 == 2 else value for index, value in enumerate(state)]
            bit_string = format(bits, f'0{width}b')
            check_amplitudes(bra, f'''gate inner(t) a,b,c {{
                pow(0.5) @ pow(2) @ negctrl @ {gate}(t) a,b,c; }}
                gate outer(t) a,b,c {{ inner(t) a,b,c; }}
                qubit[6] q; angle[{width}] v=angle[{width}]("{bit_string}");
                h q[0]; h q[1]; h q[2]; outer(v) q[0],q[1],q[2];''', expected)
    for gate, base in (('crx', 'rx'), ('cry', 'ry'), ('crz', 'rz')):
        for basis in (*range(8), None):
            state = [0j] * 128
            state[0 if basis is None else basis] = 1
            if basis is None:
                for qubit in (0, 1, 2):
                    state = apply_gate(state, 'h', (), (qubit,))
                preparation = 'h q[0]; h q[1]; h q[2];'
            else:
                preparation = ' '.join(f'x q[{qubit}];' for qubit in (0, 1, 2)
                                       if basis & (1 << qubit))
            state = apply_gate(state, 'h', (), (3,))
            transformed = apply_gate(state, base, (7 * math.pi / 4,), (3,))
            expected = [transformed[index] if index & 7 == 6 else value
                        for index, value in enumerate(state)]
            check_amplitudes(bra, f'''qubit[7] q; angle[3] a=7*pi/4;
                {preparation} h q[3];
                negctrl @ ctrl @ {gate}(a) q[0],q[1],q[2],q[3];''', expected)
        state = [0j] * 128
        state[0] = 1
        for qubit in (0, 1, 2, 3):
            state = apply_gate(state, 'h', (), (qubit,))
        transformed = apply_gate(state, base, (7 * math.pi / 4,), (3,), inverse=True)
        expected = [transformed[index] if index & 7 == 6 else value
                    for index, value in enumerate(state)]
        check_amplitudes(bra, f'''gate inner(theta) a,b,c,d {{
            negctrl @ inv @ ctrl @ {gate}(theta) a,b,c,d; }}
            gate outer(value) a,b,c,d {{ inner(value) a,b,c,d; }}
            qubit[7] q; angle[3] a=7*pi/4; h q[0]; h q[1]; h q[2]; h q[3];
            outer(a) q[0],q[1],q[2],q[3];''', expected)
        forward = apply_gate(state, base, (7 * math.pi / 4,), (3,))
        expected = [forward[index] if index & 7 == 6 else value
                    for index, value in enumerate(state)]
        check_amplitudes(bra, f'''qubit[7] q; angle[3] a=7*pi/4;
            h q[0]; h q[1]; h q[2]; h q[3];
            inv @ negctrl @ inv @ ctrl @ {gate}(a) q[0],q[1],q[2],q[3];''', expected)


if __name__ == '__main__':
    main()
