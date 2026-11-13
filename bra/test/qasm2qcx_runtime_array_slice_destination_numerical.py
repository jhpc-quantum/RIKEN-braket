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
    check_program(bra, '''array[uint[4],3] a={1,2,15}; int first=2; int step=-1; int last=0;
        a[first:step:last]=a[0:2];''', ('A1:0', 'A1:1', 'A1:2'), ['15', '2', '1'])
    check_program(bra, '''array[bool,3] a={true,false,true}; int first=-2; int last=-1;
        a[first:last]=a[0:1];''', ('A1:0', 'A1:1', 'A1:2'), ['1', '1', '0'])
    check_program(bra, '''array[float,3] a={1.5,2.5,3.5}; int first=0; int last=1;
        a[first:last]=a[1:2];''', ('A1:0', 'A1:1', 'A1:2'), ['2.5', '3.5', '3.5'])
    check_program(bra, '''array[complex,2] a={1+2im,3+4im}; int first=0; int last=0;
        a[first:last]=a[1:1]; int r=int(real(a[0])); int i=int(imag(a[0]));''',
        ('R1', 'I1'), ['3', '4'])
    check_program(bra, '''array[angle[4],3] a={angle[4]("0001"),angle[4]("0010"),angle[4]("0011")};
        int first=0; int last=1; a[first:last]=a[1:2];''', ('A1:0', 'A1:1', 'A1:2'), ['2', '3', '3'])
    check_program(bra, '''array[int,3] a={1,2,3}; int first=2; int last=1;
        a[first:last]=a[2:1];''', ('A1:0', 'A1:1', 'A1:2'), ['1', '2', '3'])
    check_program(bra, '''array[int,3] a={1,2,3}; int first=99; int last=99;
        if (false) { a[first:last]=a[0:0]; }''', ('A1:0', 'A1:1', 'A1:2'), ['1', '2', '3'])
    for source in (
            '''array[int,4] a; array[int,5] b; int first=-4; int last=0;
               a[first:last]=b[0:4];''',
            '''array[int,4] a; int first=0; int last=3; a[first:last]=a[0:1];''',
            '''array[int,4] a; int first=3; int last=4; a[first:last]=a[0:1];''',
            '''array[int,4] a; int first=0; int step=0; int last=1;
               a[first:step:last]=a[0:1];'''):
        check_error(bra, source, 'assertion failed')


if __name__ == '__main__':
    main()
