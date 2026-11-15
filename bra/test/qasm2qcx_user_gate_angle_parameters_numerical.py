#!/usr/bin/env python3
"""Execute user-defined gates with angle-valued phase parameters."""

import argparse
import cmath
import math
import pathlib
import resource

from qasm2qcx_inverse_numerical import check_amplitudes


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


if __name__ == '__main__':
    main()
