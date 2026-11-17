#!/usr/bin/env python3
"""Check floating powers of user-gate wrappers against independent matrices."""

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

    state = [0j] * 32
    state[0] = 1
    state = apply_gate(state, 'sx', (), (0,))
    check_amplitudes(bra, '''gate g a { x a; }
        qubit[5] q; pow(0.5) @ g q[0];''', state)

    # The intermediate phase -pi is mapped to the principal endpoint +pi.
    # Multiplying the exponents into a single power would give the opposite
    # phase and fail this check.
    state = [0j] * 32
    state[0] = 1
    state = apply_gate(state, 'h', (), (0,))
    state = apply_gate(state, 's', (), (0,))
    check_amplitudes(bra, '''gate inner a { inv @ s a; }
        gate outer a { pow(2) @ inner a; }
        qubit[5] q; h q[0]; pow(0.5) @ outer q[0];''', state)

    # Test the relative phase of a negative-controlled powered gate, including
    # the ordering of a broadcast over the selected target qubits.
    state = [0j] * 32
    state[0] = 1
    for qubit in (0, 1, 2):
        state = apply_gate(state, 'h', (), (qubit,))
    for target in (2, 1):
        transformed = apply_gate(state, 'sx', (), (target,))
        state = [transformed[index] if not index & 1 else value for index, value in enumerate(state)]
    check_amplitudes(bra, '''gate inner a { x a; } gate outer a { inner a; }
        qubit[5] q; h q[0]; h q[1]; h q[2];
        negctrl @ pow(0.5) @ outer q[0],q[{2,1}];''', state)

    # The forwarded -theta wraps to pi/4 at angle[3] precision.
    state = [0j] * 32
    state[0] = 1
    state = apply_gate(state, 'h', (), (0,))
    state = apply_gate(state, 'p', (math.pi / 8,), (0,))
    check_amplitudes(bra, '''gate inner(t) a { p(t) a; }
        gate outer(t) a { inner(-t) a; }
        qubit[5] q; angle[3] theta=7*pi/4; h q[0];
        pow(0.5) @ outer(theta) q[0];''', state)


if __name__ == '__main__':
    main()
