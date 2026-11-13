#!/usr/bin/env python3
"""Execute static one-dimensional array-slice copies (#565)."""

import argparse
import pathlib
import resource

from qasm2qcx_runtime_for_numerical import check_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    check_program(bra, '''array[int,4] values={1,2,3,4}; values[1:3]=values[0:2];''',
                  ('VALUES63:0','VALUES63:1','VALUES63:2','VALUES63:3'), ['1','1','2','3'])
    check_program(bra, '''array[uint[4],3] values={1,2,15}; values[2:-1:0]=values[0:2];''',
                  ('VALUES63:0','VALUES63:1','VALUES63:2'), ['15','2','1'])
    check_program(bra, '''array[bool,3] values={true,false,true}; values[0:1]=values[1:2];''',
                  ('VALUES63:0','VALUES63:1','VALUES63:2'), ['0','1','1'])
    check_program(bra, '''array[float,3] values={1.5,2.5,3.5}; values[1:2]=values[0:1];''',
                  ('VALUES63:0','VALUES63:1','VALUES63:2'), ['1.5','1.5','2.5'])
    check_program(bra, '''array[complex,2] values={1+2im,3+4im}; values[0:0]=values[1:1];
        int r=int(real(values[0])); int i=int(imag(values[0]));''', ('R1','I1'), ['3','4'])
    check_program(bra, '''array[angle[4],3] values={angle[4]("0001"),angle[4]("0010"),angle[4]("0011")};
        values[0:1]=values[1:2];''', ('VALUES63:0','VALUES63:1','VALUES63:2'), ['2','3','3'])
    print('Static array slice assignment numerical checks passed')


if __name__ == '__main__':
    main()
