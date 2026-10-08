#!/usr/bin/env python3

"""Execute runtime-indexed bit-register reads with bra."""

import argparse
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    for size in (1, 2, 3):
        for pattern in range(1 << size):
            for index in range(-size, size):
                value = (pattern >> (index % size)) & 1
                check_program(bra, f'''bit[{size}] flags = "{pattern:0{size}b}";
                    int i = {index}; bit out = flags[i]; bool ready = flags[i];
                    int total = int(flags[i]);''', ('OUT7', 'READY31', 'TOTAL31', 'I1'),
                              [str(value), str(value), str(value), str(index)])
        for index in (-size - 1, size, qasm2qcx.QASM2QCXConverter.QCX_INT_MIN,
                      qasm2qcx.QASM2QCXConverter.QCX_INT_MAX):
            check_error(bra, f'bit[{size}] flags; int i = {index}; bit out = flags[i];',
                        'assertion failed in ASSERT')

    check_program(bra, '''bit[3] flags = "101"; int i = -1;
        array[int, 2] a = {0, 1}; bit out = flags[int(flags[i])];
        int total = int(flags[a[0]]) + int(flags[a[1]]);
        out = flags[a[1]];''', ('OUT7', 'TOTAL31', 'I1'), ['0', '1', '-1'])
    check_program(bra, '''bit[3] flags = "101"; array[int, 3] a = {0, 1, 2}; int total = 0;
        for int i in a { if (flags[i]) { total += 1; } }
        for bit value in flags { total += int(flags[int(value)]); }
        int i = 0; while (i < 3) { total += int(flags[i]); i += 1; }''',
                  ('TOTAL31',), ['5'])
    check_program(bra, '''bit[2] flags = "01"; int bad = 2; int zero = 0;
        bool ready = false && flags[bad]; ready = true || flags[1 / zero];
        if (false) { ready = flags[bad]; }
        while (false) { ready = flags[bad]; }
        int total = int(ready);''', ('TOTAL31',), ['1'])
    check_error(bra, 'bit[2] flags; int zero = 0; bit out = flags[1 / zero];',
                'integer division by zero in LET')
    print('Runtime bit-register read numerical tests passed')


if __name__ == '__main__':
    main()
