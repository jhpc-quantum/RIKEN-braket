#!/usr/bin/env python3
"""Execute basic user-defined gates and check the resulting amplitudes."""

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
    scale = 1 / math.sqrt(2)
    expected = [0j] * 32
    expected[0] = expected[3] = scale
    document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
    marker = '<!-- basic-user-gate-example -->'
    if document.count(marker) != 1:
        raise RuntimeError('Expected exactly one basic user-defined gate example')
    example = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
    example = example.replace('OPENQASM 3.0;', '', 1).replace('include "stdgates.inc";', '', 1)
    check_amplitudes(bra, example, expected)
    check_amplitudes(bra, '''gate bell a,b { h a; cx a,b; }
        qubit[5] q; bell q[0],q[1];''', expected)
    state = [0j] * 32
    state[0] = 1
    state = apply_gate(state, 'h', (), (1,))
    state = apply_gate(state, 'cx', (), (1, 3))
    state = apply_gate(state, 'U', (math.pi / 2, 0, math.pi), (3,))
    state = apply_gate(state, 'h', (), (3,))
    state = apply_gate(state, 'cx', (), (3, 1))
    state = apply_gate(state, 'U', (math.pi / 2, 0, math.pi), (1,))
    check_amplitudes(bra, '''gate mixed a,b { h a; ctrl @ x a,b; U(pi/2,0,pi) b; }
        gate idle a {} qubit[5] q; mixed q[1],q[3]; idle q[0]; mixed q[3],q[1];''', state)


if __name__ == '__main__':
    main()
