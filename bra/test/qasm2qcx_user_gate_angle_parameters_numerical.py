#!/usr/bin/env python3
"""Execute user-defined gates with angle-valued phase parameters."""

import argparse
import cmath
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
    expected[0] = 1 / math.sqrt(2)
    expected[1] = cmath.exp(1j * math.pi / 4) / math.sqrt(2)
    check_amplitudes(bra, '''gate phase_gate(theta) a { p(theta) a; }
        qubit[5] q; angle[8] value=pi/4; h q[0]; phase_gate(value) q[0];''', expected)
    state = [0j] * 32
    state[0] = 1
    for gate, parameters, qubits in (('h', (), (0,)), ('h', (), (1,)),
                                      ('p', (math.pi / 2,), (0,)), ('cp', (math.pi / 4,), (0, 1)),
                                      ('p', (math.pi / 2,), (1,)), ('p', (math.pi,), (0,)),
                                      ('cp', (math.pi / 4,), (0, 1)), ('p', (math.pi,), (1,))):
        state = apply_gate(state, gate, parameters, qubits)
    check_amplitudes(bra, '''gate phases(theta,phi) a,b { p(theta) a; cp(phi) a,b; p(theta) b; }
        qubit[5] q; angle[3] a=pi/2; angle[4] b=pi/4; h q[0]; h q[1];
        phases(a,b) q[0],q[1]; a=pi; phases(a,b) q[0],q[1];''', state)
    idle = [0j] * 32
    idle[0] = 1
    check_amplitudes(bra, '''gate g(theta) a { p(theta) a; } qubit[5] q;
        array[angle[3],2] values={pi/4,pi/2}; int index=9;
        if (false) { g(values[index]) q[0]; }''', idle)


if __name__ == '__main__':
    main()
