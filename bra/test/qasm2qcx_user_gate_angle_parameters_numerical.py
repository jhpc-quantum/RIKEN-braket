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
    document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
    marker = '<!-- user-gate-angle-parameters-example -->'
    if document.count(marker) != 1:
        raise RuntimeError('Expected exactly one user-gate angle parameter example')
    example = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
    example = example.replace('OPENQASM 3.0;', '', 1).replace('include "stdgates.inc";', '', 1)
    check_amplitudes(bra, example, expected)
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
    state = [0j] * 32
    state[0] = 1
    for gate, parameters, qubits in (
            ('h', (), (0,)), ('h', (), (1,)),
            ('p', (math.pi / 4,), (0,)), ('cp', (math.pi / 4,), (0, 1)),
            ('p', (5 * math.pi / 4,), (1,)),
            ('p', (math.pi,), (0,)), ('cp', (3 * math.pi / 2,), (0, 1))):
        state = apply_gate(state, gate, parameters, qubits)
    check_amplitudes(bra, '''gate combined(theta,phi) a,b {
        p(theta+phi) a; cp(-theta) a,b; p(theta-phi) b; }
        qubit[5] q; angle[3] a=7*pi/4; angle[5] b=pi/2; h q[0]; h q[1];
        combined(a,b) q[0],q[1]; a=pi/2; combined(a,b) q[0],q[1];''', state)
    state = [0j] * 32
    state[0] = 1
    for qubit in (0, 1):
        state = apply_gate(state, 'h', (), (qubit,))
    for angle in (math.pi / 4, math.pi / 2):
        for control, target in ((0, 2), (1, 3)):
            state = apply_gate(state, 'p', (angle,), (control,))
            state = apply_gate(state, 'cx', (), (control, target))
    check_amplitudes(bra, '''gate inner(theta) a,b { p(theta) a; cx a,b; }
        gate outer(theta) a,b { inner(theta) a,b; }
        qubit[2] q; qubit[2] r; qubit spare; angle[3] a=pi/4;
        h q; outer(a) q,r; a=pi/2; outer(a) q,r;''', state)
    state = [0j] * 32
    state[0] = 1
    for gate, parameters, qubits in (
            ('h', (), (0,)), ('h', (), (1,)),
            ('p', (math.pi / 4,), (0,)), ('cp', (math.pi / 2,), (0, 1)),
            ('p', (math.pi / 4,), (1,)),
            ('p', (3 * math.pi / 4,), (0,)), ('cp', (math.pi / 2,), (0, 1)),
            ('p', (7 * math.pi / 4,), (1,))):
        state = apply_gate(state, gate, parameters, qubits)
    check_amplitudes(bra, '''gate inner(theta,phi) a,b {
        p(theta) a; cp(phi) a,b; }
        gate middle(phi,theta) a,b { inner(theta+phi,theta) a,b; p(-phi) b; }
        gate outer(x,y) a,b { middle(x,y) a,b; }
        qubit[5] q; angle[3] x=7*pi/4; angle[5] y=pi/2;
        h q[0]; h q[1]; outer(x,y) q[0],q[1];
        x=pi/4; outer(x,y) q[0],q[1];''', state)
    idle = [0j] * 32
    idle[0] = 1
    check_amplitudes(bra, '''gate g(theta) a { p(theta) a; } qubit[5] q;
        array[angle[3],2] values={pi/4,pi/2}; int index=9;
        if (false) { g(values[index]) q[0]; }''', idle)


if __name__ == '__main__':
    main()
