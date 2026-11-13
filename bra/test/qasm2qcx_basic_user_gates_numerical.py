#!/usr/bin/env python3
"""Execute basic user-defined gates and check the resulting amplitudes."""

import argparse
import math
import pathlib
import resource

from qasm2qcx_inverse_numerical import check_amplitudes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    scale = 1 / math.sqrt(2)
    expected = [0j] * 32
    expected[0] = expected[3] = scale
    check_amplitudes(bra, '''gate bell a,b { h a; cx a,b; }
        qubit[5] q; bell q[0],q[1];''', expected)


if __name__ == '__main__':
    main()
