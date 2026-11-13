#!/usr/bin/env python3
"""Execute runtime slice-to-whole-array assignments."""

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
    check_program(bra, '''array[int,3] a={1,2,3}; int first=2; int step=-1; int last=0;
        a=a[first:step:last];''', ('A1:0', 'A1:1', 'A1:2'), ['3', '2', '1'])
    check_error(bra, '''array[int,4] a; array[int,2] b; int first=0; int last=2;
        b=a[first:last];''', 'assertion failed')


if __name__ == '__main__':
    main()
