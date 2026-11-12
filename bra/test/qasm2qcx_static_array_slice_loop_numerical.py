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
    document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
    marker = '<!-- static-array-slice-loop-example -->'
    if document.count(marker) != 1:
        raise RuntimeError('Expected exactly one static array slice loop example')
    example = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
    check_program(bra, example.replace('OPENQASM 3.0;', '', 1), ('TOTAL31',), ['9'])
    check_program(bra, '''array[int,5] values={1,2,3,4,5}; int total=0;
        for int value in values[1:3] { total+=value; }
        for int value in values[4:-2:0] { total+=value; }
        for int value in values[-5:2:3] { total+=value; }
        for int value in values[3:1] { total+=100; }''', ('TOTAL31',), ['33'])
    check_program(bra, '''array[uint[4],3] values={1,2,3}; uint total=0; int i=0;
        for uint[8] value in values[0:2] { total+=value; values[i]=0; i+=1;
            if(i==1) { values[1]=7; continue; } if(i==2) { break; } }''',
        ('TOTAL31','I1','VALUES63:0','VALUES63:1','VALUES63:2'), ['8','2','0','0','3'])

    # The reference uses the written range, normalizing each index separately.
    source_values = [1, 2, 3, 4, 5]
    for first in range(-5, 5):
        for last in range(-5, 5):
            for step in (-3, -2, -1, 1, 2, 3):
                positions = range(first, last + (1 if step > 0 else -1), step)
                selected = [source_values[index] for index in positions]
                record = 0
                for value in selected:
                    record = (record * 10 + value) % 1000000
                check_program(bra, f'''array[int,5] values={{1,2,3,4,5}};
                    int visits=0; int record=0;
                    for int value in values[{first}:{step}:{last}] {{
                        record=(record*10+value)%1000000; visits+=1; }}''',
                    ('VISITS63','RECORD63'), [str(len(selected)), str(record)])

    # Every supported element type, including widening and exact angle bits.
    check_program(bra, '''array[float,4] values={0.5,1.0,1.5,2.0}; float total=0.0;
        for float value in values[3:-2:0] { total+=value; }''', ('TOTAL31',), ['3'])
    check_program(bra, '''array[bool,4] values={true,false,true,true}; int total=0;
        for bool value in values[3:-1:1] { if(value) { total+=1; } }''', ('TOTAL31',), ['2'])
    check_program(bra, '''array[complex,3] values={1+2im,2+3im,3+4im}; complex total=0;
        for complex value in values[2:-1:0] { total+=value; }
        int r=int(real(total)); int i=int(imag(total));''', ('R1','I1'), ['6','9'])
    check_program(bra, '''array[angle[4],3] values={angle[4]("0001"),angle[4]("0010"),angle[4]("0011")};
        angle[8] total=0.0; for angle[8] value in values[2:-1:1] { total+=value; }''',
        ('TOTAL31',), ['80'])
    check_program(bra, '''array[uint[4],3] values={1,2,15}; uint total=0;
        for uint[8] value in values[2:-1:0] { total+=value+uint[8](1); }''', ('TOTAL31',), ['21'])

    # Repeated aliases see later mutations; copied iterators do not change.
    check_program(bra, '''array[int,3] values={1,2,3}; int total=0; int i=0;
        for int value in values[-3:2] { total=total*10+value; i+=1;
            if(i==1) { values[0]=7; values[1]=8; } }
    ''', ('TOTAL31','I1','VALUES63:0','VALUES63:1'), ['183783','6','7','8'])

    check_program(bra, '''array[int,4] values={1,2,3,4}; int total=0; int i=0;
        for int values in values[3:-1:0] { i+=1;
            if(i==2) { continue; } if(i==4) { break; } total+=values; }
        total+=values[0];
        for int outer in [0:1] { for int value in values[1:2] { total+=value; } }
    ''', ('TOTAL31','I1'), ['17','4'])

    # A skipped outer loop still validates the slice without reading the body.
    check_program(bra, '''array[int,3] values={1,2,3}; int total=0; int bad=99;
        if(false) { for int value in values[0:2] { total+=values[bad]; } }
        for int outer in [1:0] { for int value in values[0:2] { total+=values[bad]; } }
        for int value in values[2:1] { total+=values[bad]; }''', ('TOTAL31',), ['0'])

    check_program(bra, '''include "stdgates.inc"; qubit q; h q;
        array[angle[4],3] phases={0.0,pi,0.0};
        for angle[8] phase in phases[1:1] { p(phase) q; }
        h q; bit result=measure q; int n=int(result);''', ('N1',), ['1'])
    print('Static array slice loop numerical checks passed')


if __name__ == '__main__':
    main()
