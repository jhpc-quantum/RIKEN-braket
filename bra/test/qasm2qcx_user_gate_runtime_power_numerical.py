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

    # Mixed powers multiply to -2; the outer inverse makes the effective
    # power +2. Exercise nested definitions, negative control, and broadcast.
    state = [0j] * 32
    state[0] = 1
    for qubit in (0, 1, 2):
        state = apply_gate(state, 'h', (), (qubit,))
    for target in (1, 2):
        transformed = state
        for _ in range(2):
            transformed = apply_gate(transformed, 'h', (), (target,))
            transformed = apply_gate(transformed, 'p', (math.pi / 4,), (target,))
        state = [transformed[index] if not index & 1 else value for index, value in enumerate(state)]
    check_amplitudes(bra, '''gate inner(t) a { h a; p(t) a; }
        gate outer(t) a { inner(t) a; }
        qubit[5] q; angle[3] a=pi/4; int n=-1; uint[3] u=2;
        h q[0]; h q[1]; h q[2];
        negctrl @ inv @ pow(n) @ pow(u) @ outer(a) q[0],q[{1,2}];''', state)

    # The exponent is evaluated again each time the outer runtime loop reaches
    # the gate call, rather than being captured when conversion takes place.
    state = [0j] * 32
    state[0] = 1
    state = apply_gate(state, 'h', (), (0,))
    for exponent in (1, 2):
        for _ in range(exponent):
            state = apply_gate(state, 'h', (), (0,))
            state = apply_gate(state, 'p', (math.pi / 4,), (0,))
    check_amplitudes(bra, '''gate g(t) a { h a; p(t) a; }
        qubit[5] q; angle[3] a=pi/4; int n=1; int stop=1; h q[0];
        for int i in [0:stop] { pow(n) @ g(a) q[0]; n += 1; }''', state)


if __name__ == '__main__':
    main()
