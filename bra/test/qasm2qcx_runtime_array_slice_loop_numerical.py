#!/usr/bin/env python3
"""Execute runtime-dependent array slice loops (#563)."""

import argparse
import pathlib
import resource

from qasm2qcx_runtime_for_numerical import check_program
from qasm2qcx_integer_array_numerical import check_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    check_program(bra, '''array[int,3] values={1,2,3}; int first=0; int last=2;
        int step=1; int total=0; int visits=0;
        for int value in values[first:step:last] { total+=value; visits+=1;
            if(visits==1) { first=99; last=0; step=0; values[1]=7; } }''',
        ('TOTAL31','VISITS63'), ['11','3'])
    check_program(bra, '''array[angle[4],3] values={angle[4]("0001"),angle[4]("0010"),angle[4]("0011")};
        int first=2; int last=1; int step=-1; angle[8] total=0.0;
        for angle[8] value in values[first:step:last] { total+=value; }''', ('TOTAL31',), ['80'])
    check_program(bra, '''array[int,3] values={1,2,3}; int first=2; int last=0;
        int total=0; for int value in values[first:last] { total+=value; }''', ('TOTAL31',), ['0'])
    for source in ('''array[int,3] values; int first=0; int step=0; int last=2;
                       for int value in values[first:step:last] {}''',
                   '''array[int,3] values; int first=0; int last=3;
                       for int value in values[first:last] {}'''):
        check_error(bra, source, 'assertion failed')
    print('Runtime array slice loop numerical checks passed')


if __name__ == '__main__':
    main()
