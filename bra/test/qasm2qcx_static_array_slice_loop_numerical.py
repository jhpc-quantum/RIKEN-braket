#!/usr/bin/env python3
"""Execute static one-dimensional array slice loops (#561)."""

import argparse
import pathlib
import resource

from qasm2qcx_runtime_for_numerical import check_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    check_program(bra, '''array[int,5] values={1,2,3,4,5}; int total=0;
        for int value in values[1:3] { total+=value; }
        for int value in values[4:-2:0] { total+=value; }
        for int value in values[-5:2:3] { total+=value; }
        for int value in values[3:1] { total+=100; }''', ('TOTAL31',), ['33'])
    check_program(bra, '''array[uint[4],3] values={1,2,3}; uint total=0; int i=0;
        for uint[8] value in values[0:2] { total+=value; values[i]=0; i+=1;
            if(i==1) { values[1]=7; continue; } if(i==2) { break; } }''',
        ('TOTAL31','I1','VALUES63:0','VALUES63:1','VALUES63:2'), ['8','2','0','0','3'])
    print('Static array slice loop numerical checks passed')


if __name__ == '__main__':
    main()
