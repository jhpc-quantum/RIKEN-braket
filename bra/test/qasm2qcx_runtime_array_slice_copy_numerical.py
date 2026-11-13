#!/usr/bin/env python3
"""Execute assignments between runtime-dependent array slices."""

import argparse
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    check_program(bra, '''array[int,4] a={1,2,3,4}; int d=1; int e=3; int s=0; int t=2;
        a[d:e]=a[s:t];''', ('A1:0', 'A1:1', 'A1:2', 'A1:3'), ['1', '1', '2', '3'])
    check_error(bra, '''array[int,4] a; int d=0; int e=2; int s=0; int t=1;
        a[d:e]=a[s:t];''', 'assertion failed')


if __name__ == '__main__':
    main()
