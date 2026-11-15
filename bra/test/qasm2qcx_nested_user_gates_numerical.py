#!/usr/bin/env python3
"""Execute nested user-defined gates and check the resulting amplitudes."""

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
    expected = [0j] * 32
    expected[0] = expected[3] = 1 / math.sqrt(2)
    check_amplitudes(bra, '''gate entangle a,b { h a; cx a,b; }
        gate outer b,a { entangle a,b; } qubit[5] q; outer q[1],q[0];''', expected)
    state = [0j] * 32
    state[0] = 1
    for gate, parameters, qubits in (('h', (), (1,)), ('cx', (), (1, 3)),
                                      ('U', (math.pi / 2, 0, math.pi), (3,)),
                                      ('h', (), (3,)), ('cx', (), (3, 1)),
                                      ('U', (math.pi / 2, 0, math.pi), (1,))):
        state = apply_gate(state, gate, parameters, qubits)
    check_amplitudes(bra, '''gate inner a,b { h a; ctrl @ x a,b; U(pi/2,0,pi) b; }
        gate middle b,a { inner a,b; } gate empty x {} gate outer a,b { empty a; middle b,a; }
        qubit[5] q; outer q[1],q[3]; outer q[3],q[1];''', state)


if __name__ == '__main__':
    main()
