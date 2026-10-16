#!/usr/bin/env python3

"""Execute Boolean array iteration and compare it with Python reference loops.

Example:
  python3 bra/test/qasm2qcx_bool_array_loop_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    for size in (1, 2, 4):
        for pattern in range(1 << size):
            initial = [bool(pattern & (1 << i)) for i in range(size)]
            for mode in ('all', 'continue', 'break'):
                flags = initial.copy()
                count = encoded = position = 0
                for index in range(size):
                    flag = flags[index]
                    position += 1
                    if mode == 'continue' and not flag:
                        continue
                    if mode == 'break' and flag:
                        break
                    flags[index] = not flag
                    if position < size:
                        flags[position] = not flags[position]
                    count += int(flag)
                    encoded = encoded * 2 + int(flag)
                transfer = ('if (!flag) { continue; }' if mode == 'continue'
                            else 'if (flag) { break; }' if mode == 'break' else '')
                literal = ', '.join(str(flag).lower() for flag in initial)
                check_program(bra, f'''array[bool, {size}] flags = {{{literal}}};
                    int count = 0; int encoded = 0; int i = 0;
                    for bool flag in flags {{ i += 1; {transfer}
                        flags[i - 1] = !flag;
                        if (i < {size}) {{ flags[i] = !flags[i]; }}
                        count += int(flag); encoded = encoded * 2 + int(flag);
                    }}''', ('COUNT31', 'ENCODED127', 'I1') + tuple(f'FLAGS31:{i}' for i in range(size)),
                              [str(count), str(encoded), str(position)] + [str(int(flag)) for flag in flags])

    check_program(bra, '''array[bool, 3] flags = {true, false, true}; int count = 0;
        for bool flag in flags { for bool other in flags {
            if (!other) { continue; } if (flag) { count += 1; } break;
        } }''', ('COUNT31',), ['2'])
    for outer in ('for int outer in [0:2]', 'for int outer in [0:stop]',
                  'for int outer in {0, stop, 1}', 'while (visits < 3)'):
        check_program(bra, 'array[bool, 3] flags = {true, true, false}; int count = 0; '
                      'int stop = 2; int visits = 0; ' + outer + ''' {
                          visits += 1; for bool flag in flags { if (!flag) { break; } count += 1; }
                      }''', ('COUNT31', 'VISITS63'), ['6', '3'])

    check_program(bra, '''array[bool, 2] flags = {false, true}; int count = 0;
        for bool flag in flags { for int j in [0:int(flag)] { count += j; }
            for int j in {int(flag), 2} { count += j; } }''', ('COUNT31',), ['6'])
    check_program(bra, '''const int N = 2; array[bool, N] flags = {true, false}; int count = 0;
        for bool N in flags { count += int(N); } count += N;
        for bool flags in flags { count += int(flags); } count += int(flags[0]);''',
                  ('COUNT31',), ['5'])

    check_program(bra, '''array[bool, 2] flags = {true, false}; bit b = 0; bit[2] bits = "00";
        int i = 0; for bool flag in flags { b = flag; bits[i] = flag;
            flags[i] = bits[i]; i += 1; }''',
                  ('B1', 'BITS15:0', 'BITS15:1', 'FLAGS31:0', 'FLAGS31:1'), ['0', '1', '0', '1', '0'])
    check_program(bra, '''array[bool, 2] flags = {true, false}; bool flag = false;
        array[uint[4], 1] values = {1}; array[float, 1] floats = {0.5}; bit[1] bits = "1";
        int count = 0; float total = 0;
        for bool flag in flags { for uint[4] flag in values { count += int(flag); }
            for float flag in floats { total += flag; }
            for bit flag in bits { count += int(flag); }
            if (flag) { count += 1; } }
        count += int(flag);''', ('COUNT31', 'TOTAL31', 'FLAG15'), ['5', '1', '0'])

    check_program(bra, '''array[bool, 2] flags = {true, false}; bool ready = false;
        int bad = 2; int zero = 0;
        if (false) { for bool flag in flags { ready = flags[bad]; } }
        while (false) { for bool flag in flags { ready = flags[1 / zero]; } }
        for int outer in [1:0] { for bool flag in flags { ready = flags[bad]; } }
        for bool flag in flags { ready = true || flags[bad];
            ready = false && flags[1 / zero]; continue; ready = flags[bad]; }''',
                  ('READY31',), ['0'])

    check_program(bra, '''array[bool, 2] flags = {true, false}; int count = 0; uint u = 0;
        float total = 0; complex z;
        for bool flag in flags { count += int(flag); u += uint[4](flag);
            z = complex(flag); total += float(z); if (flag != false) { count += 1; } }''',
                  ('COUNT31', 'U1', 'TOTAL31'), ['2', '1', '1'])
    for gate in ('rx(float(flag) * pi)', 'ry(-float(flag) * pi)', 'U(int(flag) * pi, 0, 0)'):
        check_program(bra, 'include "stdgates.inc"; qubit q; array[bool, 2] flags = {true, false}; '
                      f'for bool flag in flags {{ {gate} q; gphase(int(flag)); }} '
                      'bit result = measure q; int n = int(result);', ('N1', 'FLAGS31:0'), ['1', '1'])

    check_error(bra, '''array[bool, 1] flags = {false}; bool ready = false; int bad = 1;
        for bool flag in flags { ready = flag || flags[bad]; }''', 'assertion failed in ASSERT')
    check_error(bra, '''array[bool, 1] flags = {true}; bool ready = false; int zero = 0;
        for bool flag in flags { ready = flag && flags[1 / zero]; }''', 'integer division by zero in LET')

    print('Boolean array loop numerical checks passed')


if __name__ == '__main__':
    main()
