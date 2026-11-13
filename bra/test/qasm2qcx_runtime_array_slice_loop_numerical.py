#!/usr/bin/env python3
"""Execute runtime-dependent array slice loops (#563)."""

import argparse
import pathlib
import resource
import subprocess

from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx
from qasm2qcx_integer_array_numerical import check_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
    marker = '<!-- runtime-array-slice-loop-example -->'
    if document.count(marker) != 1:
        raise RuntimeError('Expected exactly one runtime array slice loop example')
    example = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
    check_program(bra, example.replace('OPENQASM 3.0;', '', 1), ('TOTAL31', 'LAST15'), ['9', '0'])
    def fail_before_body(source):
        lines = qasm2qcx.convert('OPENQASM 3; ' + source)
        body = lines.index('@QASM2QCX_LOOP_0_BODY')
        lines.insert(body + 1, 'PRINTLN 999')
        result = subprocess.run([str(bra)], input='\n'.join(lines)+'\n', text=True,
                                capture_output=True, timeout=30)
        if result.returncode == 0 or 'assertion failed' not in result.stderr or result.stdout.strip():
            raise RuntimeError(f'Expected pre-body slice failure: {result}')
        if 'runtime error:' in result.stderr:
            raise RuntimeError(f'UBSan reported a failure: {result.stderr}')
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
    fail_before_body('''array[int,3] values; int first=0; int last=3;
                        for int value in values[first:last] {}''')

    # Compare runtime ranges with independently normalized Python indices.
    source_values = [1, 2, 3, 4, 5]
    for first in range(-5, 5):
        for last in range(-5, 5):
            for step in (-3, -2, -1, 1, 2, 3):
                selected = [source_values[index] for index in
                            range(first, last + (1 if step > 0 else -1), step)]
                record = 0
                for value in selected:
                    record = (record * 10 + value) % 1000000
                check_program(bra, f'''array[int,5] values={{1,2,3,4,5}};
                    int first={first}; int last={last}; int step={step}; int visits=0; int record=0;
                    for int value in values[first:step:last] {{
                        record=(record*10+value)%1000000; visits+=1; }}''',
                    ('VISITS63','RECORD63'), [str(len(selected)), str(record)])

    # Cover supported element types and both kinds of iterator widening.
    check_program(bra, '''array[float,4] values={0.5,1.0,1.5,2.0}; int first=3; int last=0; int step=-2;
        float total=0.0; for float value in values[first:step:last] { total+=value; }''', ('TOTAL31',), ['3'])
    check_program(bra, '''array[bool,4] values={true,false,true,true}; int first=3; int last=1; int step=-1;
        int total=0; for bool value in values[first:step:last] { if(value) { total+=1; } }''', ('TOTAL31',), ['2'])
    check_program(bra, '''array[complex,3] values={1+2im,2+3im,3+4im}; int first=2; int last=0; int step=-1;
        complex total=0; for complex value in values[first:step:last] { total+=value; }
        int r=int(real(total)); int i=int(imag(total));''', ('R1','I1'), ['6','9'])
    check_program(bra, '''array[uint[4],3] values={1,2,15}; int first=2; int last=0; int step=-1; uint total=0;
        for uint[8] value in values[first:step:last] { total+=value+uint[8](1); }''', ('TOTAL31',), ['21'])

    # Crossing zero can revisit a position; each visit reads its current value.
    check_program(bra, '''array[int,3] values={1,2,3}; int first=-3; int last=2; int total=0; int i=0;
        for int value in values[first:last] { total=total*10+value; i+=1;
            if(i==1) { first=9; last=9; values[0]=7; values[1]=8; } }''',
        ('TOTAL31','I1','VALUES63:0','VALUES63:1'), ['183783','6','7','8'])
    check_program(bra, '''array[int,4] values={1,2,3,4}; int first=3; int last=0; int step=-1;
        int total=0; int i=0; for int values in values[first:step:last] { i+=1;
            if(i==2) { continue; } if(i==4) { break; } total+=values; } total+=values[0];
        for int outer in [0:1] { first=1; last=2; for int value in values[first:last] { total+=value; } }''',
        ('TOTAL31','I1'), ['17','4'])
    check_program(bra, '''array[int,3] values={1,2,3}; int first=0; int last=2; int zero=0; int total=0; int bad=99;
        if(false) { for int value in values[first/zero:zero:99] { total+=value; } }
        for int outer in [1:0] { for int value in values[first/zero:zero:99] { total+=value; } }
        for int value in values[first:last] { continue; total+=values[bad]; }''', ('TOTAL31',), ['0'])

    # Endpoint steps must not overflow after a singleton selection.
    for step_type, step in (('int',str(qasm2qcx.QASM2QCXConverter.QCX_INT_MIN)),
                            ('int',str(qasm2qcx.QASM2QCXConverter.QCX_INT_MAX)), ('uint','uint(-1)')):
        check_program(bra, f'''array[int,1] values={{7}}; int first=0; int last=0;
            {step_type} step={step}; int total=0;
            for int value in values[first:step:last] {{ total+=value; }}''', ('TOTAL31',), ['7'])

    check_program(bra, '''include "stdgates.inc"; qubit q; h q;
        array[angle[4],3] phases={0.0,pi,0.0}; int first=1; int last=1;
        for angle[8] phase in phases[first:last] { p(phase) q; }
        h q; bit result=measure q; int n=int(result);''', ('N1',), ['1'])

    # The selection budget fails before the first body execution.
    previous_limit = qasm2qcx.QASM2QCXConverter.MAX_RUNTIME_BIT_RANGE_POSITIONS
    try:
        qasm2qcx.QASM2QCXConverter.MAX_RUNTIME_BIT_RANGE_POSITIONS = 2
        fail_before_body('''array[int,2] values={1,2}; int first=-1; int last=1;
            for int value in values[first:last] { }''')
    finally:
        qasm2qcx.QASM2QCXConverter.MAX_RUNTIME_BIT_RANGE_POSITIONS = previous_limit
    print('Runtime array slice loop numerical checks passed')


if __name__ == '__main__':
    main()
