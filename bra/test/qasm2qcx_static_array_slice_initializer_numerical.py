#!/usr/bin/env python3
"""Execute classical array initializers from static slices."""

import argparse
import pathlib
import resource

from qasm2qcx_runtime_for_numerical import check_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    check_program(bra, '''array[int,4] a={1,2,3,4}; array[int,2] b=a[1:2];
        a[1]=9; b[1]=8;''', ('A1:1', 'A1:2', 'B1:0', 'B1:1'), ['9', '3', '2', '8'])


if __name__ == '__main__':
    main()
