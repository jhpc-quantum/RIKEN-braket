#!/usr/bin/env python3
"""Execute runtime array-slice source assignments (#567)."""

import argparse
import pathlib
import random
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
    marker = '<!-- runtime-array-slice-source-example -->'
    if document.count(marker) != 1:
        raise RuntimeError('Expected exactly one runtime array-slice source example')
    example = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
    check_program(bra, example.replace('OPENQASM 3.0;', '', 1),
                  ('VALUES63:0','VALUES63:1','VALUES63:2','VALUES63:3'), ['1','1','2','3'])
    check_program(bra, '''array[int,4] a={1,2,3,4}; int first=0; int last=2;
        a[1:3]=a[first:last];''', ('A1:0','A1:1','A1:2','A1:3'), ['1','1','2','3'])
    check_program(bra, '''array[int,5] a={0,0,0,0,0}; array[int,4] b={1,2,3,4};
        int first=-4; int last=0; a[0:4]=b[first:last];''',
        ('A1:0','A1:1','A1:2','A1:3','A1:4'), ['1','2','3','4','1'])
    check_program(bra, '''array[uint[4],3] a={1,2,15}; int first=0; int last=2; int step=1;
        a[2:-1:0]=a[first:step:last];''', ('A1:0','A1:1','A1:2'), ['15','2','1'])
    check_program(bra, '''array[bool,3] a={true,false,true}; int first=1; int last=2;
        a[0:1]=a[first:last];''', ('A1:0','A1:1','A1:2'), ['0','1','1'])
    check_program(bra, '''array[float,3] a={1.5,2.5,3.5}; int first=0; int last=1;
        a[1:2]=a[first:last];''', ('A1:0','A1:1','A1:2'), ['1.5','1.5','2.5'])
    check_program(bra, '''array[complex,2] a={1+2im,3+4im}; int first=1; int last=1;
        a[0:0]=a[first:last]; int r=int(real(a[0])); int i=int(imag(a[0]));''',
        ('R1','I1'), ['3','4'])
    check_program(bra, '''array[angle[4],3] a={angle[4]("0001"),angle[4]("0010"),angle[4]("0011")};
        int first=1; int last=2; a[0:1]=a[first:last];''', ('A1:0','A1:1','A1:2'), ['2','3','3'])
    check_program(bra, '''array[int,3] a={1,2,3}; int first=2; int last=1;
        a[2:1]=a[first:last];''', ('A1:0','A1:1','A1:2'), ['1','2','3'])
    for source in ('''array[int,3] a; int first=0; int last=2; a[0:1]=a[first:last];''',
                   '''array[int,3] a; int first=0; int last=3; a[0:1]=a[first:last];''',
                   '''array[int,3] a; int first=0; int step=0; int last=1; a[0:1]=a[first:step:last];'''):
        check_error(bra, source, 'assertion failed')

    # Independent snapshot model for overlapping and distinct array copies.
    choices = []
    for first in range(-4, 4):
        for last in range(-4, 4):
            for step in (-3, -2, -1, 1, 2, 3):
                indices = tuple(index % 4 for index in
                                range(first, last + (1 if step > 0 else -1), step))
                choices.append((first, step, last, indices))
    destinations = [item for item in choices if len(set(item[3])) == len(item[3])]
    grouped = {length: [item for item in choices if len(item[3]) == length] for length in range(5)}
    rng = random.Random(567)
    for case in range(300):
        first, step, last, destination = rng.choice(destinations)
        src_first, src_step, src_last, source = rng.choice(grouped[len(destination)])
        original, other = [10,20,30,40], [5,6,7,8]
        same = bool(case % 2)
        source_values = original if same else other
        expected = original.copy()
        snapshot = [source_values[index] for index in source]
        for index, value in zip(destination, snapshot):
            expected[index] = value
        source_name = 'a' if same else 'b'
        check_program(bra, f'''array[int,4] a={{10,20,30,40}}; array[int,4] b={{5,6,7,8}};
            int first={src_first}; int step={src_step}; int last={src_last};
            a[{first}:{step}:{last}]={source_name}[first:step:last];''',
            tuple(f'A1:{i}' for i in range(4)) + tuple(f'B1:{i}' for i in range(4)),
            list(map(str, expected + other)))

    native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
    for width in range(1, native + 1):
        mask = (1 << width) - 1
        check_program(bra, f'''array[uint[{width}],3] a={{0,1,uint[{width}](-1)}};
            int first=0; int last=1; a[1:2]=a[first:last];''',
            ('A1:0','A1:1','A1:2'), ['0','0','1'])
        check_program(bra, f'''array[angle[{width}],3] a={{angle[{width}]("{mask:0{width}b}"),
            angle[{width}]("{1:0{width}b}"),angle[{width}]("{0:0{width}b}")}};
            int first=1; int last=2; a[0:1]=a[first:last];''', ('A1:0','A1:1','A1:2'), ['1','0','0'])

    check_program(bra, '''array[complex,3] a={1+2im,3+4im,5+6im}; int first=0; int last=2;
        a[2:-1:0]=a[first:last]; int r=int(real(a[0])); int i=int(imag(a[2]));''',
        ('R1','I1'), ['5','2'])
    check_program(bra, '''array[int,4] a={2,3,0,1}; a[1:2]=a[a[2]:a[3]];''',
                  ('A1:0','A1:1','A1:2','A1:3'), ['2','2','3','1'])
    check_program(bra, '''array[int,4] a={1,2,3,4}; int first=0; int last=1;
        for int i in [0:1] { a[2:3]=a[first:last]; first=1; last=2; a[1]=9; }''',
        ('A1:0','A1:1','A1:2','A1:3'), ['1','9','9','1'])
    check_program(bra, '''array[int,4] a={1,2,3,4}; array[int,4] b={5,6,7,8};
        int first=0; int last=2; int zero=0;
        if(false) { a[0:1]=b[first/zero:0:99]; }
        for int i in [1:0] { a[0:1]=b[first/zero:0:99]; }
        for int i in [0:1] { a[1:3]=a[first:last]; break; a[0:1]=b[first/zero:0:99]; }''',
        ('A1:0','A1:1','A1:2','A1:3','B1:0'), ['1','1','2','3','5'])

    # Reached failures must precede both source snapshots and destination writes.
    def fail_before_snapshot(source, expected_finish):
        lines = qasm2qcx.convert('OPENQASM 3; ' + source)
        instrumented = []
        for line in lines:
            if line.startswith('LET QASM2QCX_ARRAY_SLICE_') and ':0 :=' in line:
                instrumented.append('PRINTLN 222')
            if line.startswith('LET A1:'):
                instrumented.append('PRINTLN 333')
            instrumented.append(line)
            if line.startswith('@QASM2QCX_ARRAY_SLICE_COPY_') and line.endswith('_FINISH'):
                instrumented.append('PRINTLN 111')
        result = subprocess.run([str(bra)], input='\n'.join(instrumented)+'\n',
                                text=True, capture_output=True, timeout=30)
        if (result.returncode == 0 or 'assertion failed' not in result.stderr
                or result.stdout.split() != (['111'] if expected_finish else [])
                or 'runtime error:' in result.stderr):
            raise RuntimeError(f'Expected pre-snapshot failure: {result}')
    fail_before_snapshot('array[int,3] a; int first=0; int last=2; a[0:1]=a[first:last];', True)
    fail_before_snapshot('array[int,3] a; int first=0; int last=3; a[0:1]=a[first:last];', False)
    fail_before_snapshot('array[int,3] a; int first=0; int last=1; int step=0; a[0:1]=a[first:step:last];', False)
    fail_before_snapshot('array[int,3] a; int first=0; int last=0; a[2:1]=a[first:last];', True)
    fail_before_snapshot('array[int,3] a; int first=2; int last=1; int step=0; a[2:1]=a[first:step:last];', False)
    previous_limit = qasm2qcx.QASM2QCXConverter.MAX_RUNTIME_BIT_RANGE_POSITIONS
    try:
        qasm2qcx.QASM2QCXConverter.MAX_RUNTIME_BIT_RANGE_POSITIONS = 2
        fail_before_snapshot('array[int,3] a; int first=0; int last=2; a[0:2]=a[first:last];', False)
    finally:
        qasm2qcx.QASM2QCXConverter.MAX_RUNTIME_BIT_RANGE_POSITIONS = previous_limit

    # Singleton selections must handle extreme native step values.
    for step_type, step in (('int',str(qasm2qcx.QASM2QCXConverter.QCX_INT_MIN)),
                            ('int',str(qasm2qcx.QASM2QCXConverter.QCX_INT_MAX)), ('uint','uint(-1)')):
        check_program(bra, f'''array[int,2] a={{7,0}}; int first=0; int last=0; {step_type} step={step};
            a[1:1]=a[first:step:last];''', ('A1:0','A1:1'), ['7','7'])

    check_program(bra, '''include "stdgates.inc"; qubit q; h q;
        array[angle[4],2] phases={0.0,pi}; int first=1; int last=1;
        phases[0:0]=phases[first:last]; p(phases[0]) q; h q;
        bit result=measure q; int n=int(result);''', ('N1',), ['1'])
    print('Runtime array slice source numerical checks passed')


if __name__ == '__main__':
    main()
