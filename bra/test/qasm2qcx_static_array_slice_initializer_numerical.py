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
    check_program(bra, '''array[int,4] a={1,2,3,4}; array[int,5] b=a[-4:0];''',
                  ('B1:0', 'B1:1', 'B1:2', 'B1:3', 'B1:4'), ['1', '2', '3', '4', '1'])
    check_program(bra, '''array[uint[4],4] a={1,2,3,15}; array[uint[4],2] b=a[-1:-2:-3];''',
                  ('B1:0', 'B1:1'), ['15', '2'])
    check_program(bra, '''array[bool,3] a={true,false,true}; array[bool,2] b=a[1:2];''',
                  ('B1:0', 'B1:1'), ['0', '1'])
    check_program(bra, '''array[float,3] a={1.5,2.5,3.5}; array[float,2] b=a[2:-1:1];''',
                  ('B1:0', 'B1:1'), ['3.5', '2.5'])
    check_program(bra, '''array[complex,2] a={1+2im,3+4im}; array[complex,1] b=a[1:1];
        int r=int(real(b[0])); int i=int(imag(b[0]));''', ('R1', 'I1'), ['3', '4'])
    check_program(bra, '''array[angle[4],3] a={angle[4]("0001"),angle[4]("0010"),angle[4]("0011")};
        array[angle[4],2] b=a[1:2];''', ('B1:0', 'B1:1'), ['2', '3'])


if __name__ == '__main__':
    main()
