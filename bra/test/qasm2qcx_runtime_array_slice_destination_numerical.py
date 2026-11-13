#!/usr/bin/env python3
"""Check runtime array-slice destinations against bra."""

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
    check_program(bra, '''array[int,4] a={1,2,3,4}; int first=1; int last=3;
        a[first:last]=a[0:2];''', ('A1:0', 'A1:1', 'A1:2', 'A1:3'), ['1', '1', '2', '3'])
    check_error(bra, '''array[int,4] a; array[int,5] b; int first=-4; int last=0;
        a[first:last]=b[0:4];''', 'assertion failed')


if __name__ == '__main__':
    main()
