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
    document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
    marker = '<!-- runtime-slice-to-whole-array-example -->'
    if document.count(marker) != 1:
        raise RuntimeError('Expected exactly one runtime slice-to-whole-array example')
    example = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
    check_program(bra, example.replace('OPENQASM 3.0;', '', 1), ('RESULT63:0', 'RESULT63:1'), ['2', '3'])
    check_program(bra, '''array[int,3] a={1,2,3}; int first=2; int step=-1; int last=0;
        a=a[first:step:last];''', ('A1:0', 'A1:1', 'A1:2'), ['3', '2', '1'])
    check_program(bra, '''array[int,4] a={1,2,3,4}; array[int,5] b; int first=-4; int last=0;
        b=a[first:last];''', ('B1:0','B1:1','B1:2','B1:3','B1:4'), ['1','2','3','4','1'])
    check_program(bra, '''array[uint[4],4] a={1,2,3,15}; array[uint[4],2] b; int first=-1; int step=-2; int last=-3;
        b=a[first:step:last];''', ('B1:0','B1:1'), ['15','2'])
    check_program(bra, '''array[float,3] a={1.5,2.5,3.5}; array[float,2] b; int first=1; int last=2;
        b=a[first:last];''', ('B1:0','B1:1'), ['2.5','3.5'])
    check_program(bra, '''array[bool,3] a={true,false,true}; array[bool,2] b; int first=1; int last=2;
        b=a[first:last];''', ('B1:0','B1:1'), ['0','1'])
    check_program(bra, '''array[complex,2] a={1+2im,3+4im}; array[complex,1] b; int first=1; int last=1;
        b=a[first:last]; int r=int(real(b[0])); int i=int(imag(b[0]));''', ('R1','I1'), ['3','4'])
    check_program(bra, '''array[angle[4],3] a={angle[4]("0001"),angle[4]("0010"),angle[4]("0011")};
        array[angle[4],2] b; uint first=1; uint last=2; b=a[first:last];''', ('B1:0','B1:1'), ['2','3'])
    check_program(bra, '''array[int,3] a={1,2,3}; array[int,2] b={4,5}; int first=99; int last=100;
        if (false) { b=a[first:last]; }''', ('B1:0','B1:1'), ['4','5'])
    for source in (
            '''array[int,4] a; array[int,2] b; int first=0; int last=2; b=a[first:last];''',
            '''array[int,4] a; array[int,2] b; int first=3; int last=4; b=a[first:last];''',
            '''array[int,4] a; array[int,2] b; int first=0; int step=0; int last=1; b=a[first:step:last];''',
            '''array[int,4] a; array[int,2] b; int first=2; int last=1; b=a[first:last];'''):
        check_error(bra, source, 'assertion failed')


if __name__ == '__main__':
    main()
