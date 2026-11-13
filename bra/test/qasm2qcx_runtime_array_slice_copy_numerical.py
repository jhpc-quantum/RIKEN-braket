#!/usr/bin/env python3
"""Execute assignments between runtime-dependent array slices."""

import argparse
import pathlib
import random
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    check_program(bra, '''array[int,4] a={1,2,3,4}; int d=1; int e=3; int s=0; int t=2;
        a[d:e]=a[s:t];''', ('A1:0', 'A1:1', 'A1:2', 'A1:3'), ['1', '1', '2', '3'])
    check_program(bra, '''array[int,4] a={1,2,3,4}; int d=0; int e=2; int s=1; int t=3;
        a[d:e]=a[s:t];''', ('A1:0', 'A1:1', 'A1:2', 'A1:3'), ['2', '3', '4', '4'])
    check_program(bra, '''array[uint[4],3] a={1,2,15}; int d=2; int e=0; int p=-1; int s=0; int t=2;
        a[d:p:e]=a[s:t];''', ('A1:0', 'A1:1', 'A1:2'), ['15', '2', '1'])
    check_program(bra, '''array[bool,3] a={true,false,true}; int d=-2; int e=-1; int s=0; int t=1;
        a[d:e]=a[s:t];''', ('A1:0', 'A1:1', 'A1:2'), ['1', '1', '0'])
    check_program(bra, '''array[float,3] a={1.5,2.5,3.5}; int d=0; int e=1; int s=1; int t=2;
        a[d:e]=a[s:t];''', ('A1:0', 'A1:1', 'A1:2'), ['2.5', '3.5', '3.5'])
    check_program(bra, '''array[complex,2] a={1+2im,3+4im}; int d=0; int e=0; int s=1; int t=1;
        a[d:e]=a[s:t]; int r=int(real(a[0])); int i=int(imag(a[0]));''', ('R1', 'I1'), ['3', '4'])
    check_program(bra, '''array[angle[4],3] a={angle[4]("0001"),angle[4]("0010"),angle[4]("0011")};
        int d=0; int e=1; int s=1; int t=2; a[d:e]=a[s:t];''', ('A1:0', 'A1:1', 'A1:2'), ['2', '3', '3'])
    check_program(bra, '''array[int,3] a={1,2,3}; int d=2; int e=1; int s=2; int t=1;
        a[d:e]=a[s:t];''', ('A1:0', 'A1:1', 'A1:2'), ['1', '2', '3'])
    check_program(bra, '''array[int,3] a={1,2,3}; int d=99; int e=99; int s=99; int t=99;
        if (false) { a[d:e]=a[s:t]; }''', ('A1:0', 'A1:1', 'A1:2'), ['1', '2', '3'])
    check_program(bra, '''array[int,3] a={1,2,3}; array[int,2] b={5,6};
        int d=0; int e=2; int s=-2; int t=0; a[d:e]=b[s:t];''', ('A1:0', 'A1:1', 'A1:2'), ['5', '6', '5'])
    for source in (
            '''array[int,4] a; int d=0; int e=2; int s=0; int t=1; a[d:e]=a[s:t];''',
            '''array[int,4] a; array[int,5] b; int d=-4; int e=0; int s=0; int t=4; a[d:e]=b[s:t];''',
            '''array[int,4] a; int d=3; int e=4; int s=0; int t=1; a[d:e]=a[s:t];''',
            '''array[int,4] a; int d=0; int e=1; int s=3; int t=4; a[d:e]=a[s:t];''',
            '''array[int,4] a; int d=0; int e=1; int s=0; int t=1; int step=0; a[d:step:e]=a[s:t];''',
            '''array[int,4] a; int d=0; int e=1; int s=0; int t=1; int step=0; a[d:e]=a[s:step:t];'''):
        check_error(bra, source, 'assertion failed')

    # Compare captured positions and overlapping copies with an independent model.
    choices = []
    for start in range(-4, 4):
        for stop in range(-4, 4):
            for step in (-3, -2, -1, 1, 2, 3):
                indices = tuple(index % 4 for index in range(start, stop + (1 if step > 0 else -1), step))
                choices.append((start, step, stop, indices))
    destinations = [case for case in choices if len(set(case[3])) == len(case[3])]
    rng = random.Random(571)
    for _ in range(20):
        d, p, e, destination = rng.choice(destinations)
        s, q, t, source = rng.choice([case for case in choices if len(case[3]) == len(destination)])
        expected = [10, 20, 30, 40]
        snapshot = [expected[index] for index in source]
        for index, value in zip(destination, snapshot):
            expected[index] = value
        check_program(bra, f'''array[int,4] a={{10,20,30,40}};
            int d={d}; int p={p}; int e={e}; int s={s}; int q={q}; int t={t};
            a[d:p:e]=a[s:q:t];''', ('A1:0', 'A1:1', 'A1:2', 'A1:3'), list(map(str, expected)))


if __name__ == '__main__':
    main()
