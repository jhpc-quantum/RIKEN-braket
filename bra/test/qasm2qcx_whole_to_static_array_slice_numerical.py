#!/usr/bin/env python3
"""Execute whole-array to static-slice assignments."""

import argparse
import pathlib
import resource

from qasm2qcx_runtime_for_numerical import check_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
    marker = '<!-- whole-to-static-array-slice-example -->'
    if document.count(marker) != 1:
        raise RuntimeError('Expected exactly one whole-array to static-slice example')
    example = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
    check_program(bra, example.replace('OPENQASM 3.0;', '', 1),
                  ('RESULT63:0', 'RESULT63:1', 'RESULT63:2', 'RESULT63:3'), ['0', '1', '2', '3'])
    check_program(bra, '''array[int,3] a={1,2,3}; a[2:-1:0]=a;''',
                  ('A1:0', 'A1:1', 'A1:2'), ['3', '2', '1'])
    check_program(bra, '''array[int,4] a={0,0,0,0}; array[int,2] b={1,2}; a[-1:-2:-3]=b;''',
                  ('A1:0','A1:1','A1:2','A1:3'), ['0','2','0','1'])
    check_program(bra, '''array[uint[4],3] a; array[uint[4],2] b={1,15}; a[1:2]=b;''',
                  ('A1:1','A1:2'), ['1','15'])
    check_program(bra, '''array[float,3] a; array[float,2] b={1.5,2.5}; a[2:-1:1]=b;''',
                  ('A1:1','A1:2'), ['2.5','1.5'])
    check_program(bra, '''array[bool,3] a; array[bool,2] b={true,false}; a[1:2]=b;''',
                  ('A1:1','A1:2'), ['1','0'])
    check_program(bra, '''array[complex,2] a; array[complex,1] b={3+4im}; a[1:1]=b;
        int r=int(real(a[1])); int i=int(imag(a[1]));''', ('R1','I1'), ['3','4'])
    check_program(bra, '''array[angle[4],3] a; array[angle[4],2] b={angle[4]("0001"),angle[4]("0010")};
        a[1:2]=b;''', ('A1:1','A1:2'), ['1','2'])
    check_program(bra, '''array[int,3] a={1,2,3}; array[int,2] b={4,5};
        if (false) { a[0:1]=b; }''', ('A1:0','A1:1','A1:2'), ['1','2','3'])


if __name__ == '__main__':
    main()
