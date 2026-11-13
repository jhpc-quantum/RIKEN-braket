#!/usr/bin/env python3
"""Execute whole-array assignments."""

import argparse
import pathlib
import resource

from qasm2qcx_runtime_for_numerical import check_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    check_program(bra, '''array[int,3] a={1,2,3}; array[int,3] b={4,5,6};
        b=a; a[1]=8; b[2]=9; b=b;''', ('A1:0','A1:1','A1:2','B1:0','B1:1','B1:2'),
        ['1','8','3','1','2','9'])
    check_program(bra, '''array[uint[4],3] a={1,2,15}; array[uint[4],3] b; b=a;''',
                  ('B1:0','B1:1','B1:2'), ['1','2','15'])
    check_program(bra, '''array[float,2] a={1.5,2.5}; array[float,2] b; b=a;''',
                  ('B1:0','B1:1'), ['1.5','2.5'])
    check_program(bra, '''array[bool,2] a={true,false}; array[bool,2] b; b=a;''',
                  ('B1:0','B1:1'), ['1','0'])
    check_program(bra, '''array[complex,1] a={3+4im}; array[complex,1] b; b=a;
        int r=int(real(b[0])); int i=int(imag(b[0]));''', ('R1','I1'), ['3','4'])
    check_program(bra, '''array[angle[4],2] a={angle[4]("0001"),angle[4]("0010")};
        array[angle[4],2] b; b=a;''', ('B1:0','B1:1'), ['1','2'])
    check_program(bra, '''array[int,2] a={1,2}; array[int,2] b={3,4};
        if (false) { b=a; }''', ('B1:0','B1:1'), ['3','4'])


if __name__ == '__main__':
    main()
