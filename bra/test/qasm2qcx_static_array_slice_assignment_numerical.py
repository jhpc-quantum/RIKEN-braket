#!/usr/bin/env python3
"""Execute static one-dimensional array-slice copies (#565)."""

import argparse
import pathlib
import random
import resource

from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
    marker = '<!-- static-array-slice-assignment-example -->'
    if document.count(marker) != 1:
        raise RuntimeError('Expected exactly one static array slice assignment example')
    example = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
    check_program(bra, example.replace('OPENQASM 3.0;', '', 1),
                  ('VALUES63:0','VALUES63:1','VALUES63:2','VALUES63:3'), ['1','1','2','3'])
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

    # Compare overlapping and distinct arrays against an independent copy model.
    choices = []
    for first in range(-4, 4):
        for last in range(-4, 4):
            for step in (-3, -2, -1, 1, 2, 3):
                positions = tuple(index % 4 for index in
                                  range(first, last + (1 if step > 0 else -1), step))
                choices.append((first, step, last, positions))
    destinations = [item for item in choices if len(set(item[3])) == len(item[3])]
    grouped = {length: [item for item in choices if len(item[3]) == length] for length in range(5)}
    rng = random.Random(565)
    for case in range(300):
        first, step, last, dst = rng.choice(destinations)
        src_first, src_step, src_last, src = rng.choice(grouped[len(dst)])
        original = [10,20,30,40]
        other = [5,6,7,8]
        same_array = bool(case % 2)
        source_values = original if same_array else other
        expected = original.copy()
        snapshot = [source_values[index] for index in src]
        for index, value in zip(dst, snapshot):
            expected[index] = value
        source_name = 'a' if same_array else 'b'
        check_program(bra, f'''array[int,4] a={{10,20,30,40}}; array[int,4] b={{5,6,7,8}};
            a[{first}:{step}:{last}]={source_name}[{src_first}:{src_step}:{src_last}];''',
            tuple(f'A1:{i}' for i in range(4)) + tuple(f'B1:{i}' for i in range(4)),
            list(map(str, expected + other)))

    # Exercise raw UINT and ANGLE representations at every supported width.
    native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
    for width in range(1, native + 1):
        mask = (1 << width) - 1
        check_program(bra, f'''array[uint[{width}],3] values={{0,1,uint[{width}](-1)}};
            values[1:2]=values[0:1];''',
            ('VALUES63:0','VALUES63:1','VALUES63:2'), ['0','0','1'])
        check_program(bra, f'''array[angle[{width}],3] values={{angle[{width}]("{mask:0{width}b}"),
            angle[{width}]("{1:0{width}b}"),angle[{width}]("{0:0{width}b}")}};
            values[0:1]=values[1:2];''',
            ('VALUES63:0','VALUES63:1','VALUES63:2'), ['1','0','0'])

    # Empty and unreachable copies do not read or modify source arrays.
    check_program(bra, '''array[int,4] a={1,2,3,4}; array[int,4] b={5,6,7,8};
        a[3:1]=b[2:0]; if(false) { a[0:2]=b[1:3]; }
        for int i in [1:0] { a[0:2]=b[1:3]; }
        for int i in [0:1] { a[0:1]=a[2:3]; break; a[0:2]=b[1:3]; }''',
        ('A1:0','A1:1','A1:2','A1:3','B1:0'), ['3','4','3','4','5'])

    check_program(bra, '''array[complex,3] a={1+2im,3+4im,5+6im}; a[2:-1:0]=a[0:2];
        int r=int(real(a[0])); int i=int(imag(a[2]));''', ('R1','I1'), ['5','2'])

    check_program(bra, '''include "stdgates.inc"; qubit q; h q;
        array[angle[4],2] values={0.0,pi}; values[0:0]=values[1:1];
        p(values[0]) q; h q; bit result=measure q; int n=int(result);''', ('N1',), ['1'])
    print('Static array slice assignment numerical checks passed')


if __name__ == '__main__':
    main()
