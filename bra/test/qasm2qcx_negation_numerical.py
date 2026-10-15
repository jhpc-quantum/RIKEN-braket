#!/usr/bin/env python3

"""Check native INT unary negation, including failures and skipped paths.

Example:
  python3 bra/test/qasm2qcx_negation_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def check_integration(bra: pathlib.Path) -> None:
    minimum = qasm2qcx.QASM2QCXConverter.QCX_INT_MIN
    check_program(bra, '''int a = -3; int x = -(-a); int y = 2;
        y += -a; y -= -a;
        array[int, 2] v = {2, 3}; int i = 0;
        v[i + 1] = -v[i]; v[i] += -v[i + 1];
        bool c = -a == 3 && -(-a) == a;''',
                  ('X1', 'Y1', 'V1:0', 'V1:1', 'C1'), ['-3', '2', '4', '-2', '1'])
    check_program(bra, '''int a = -2; int step = -1; int x = 0;
        for int i in [0:-step:2] { x += -i; }
        for int j in {-a, -step} { x += -j; }
        while (-step > 0) { step = -step; if (-a == 2) { break; } }
        array[int, 2] v = {5, 7}; int y = -v[-step + 1];
        bit[2] flags = "11"; for bit b in flags { x += -int(b); }
        for int k in v { y += -k; }''', ('X1', 'Y1', 'STEP15'), ['-8', '-17', '1'])
    check_program(bra, f'''int a = {minimum}; uint u = -uint(a);
        float f = -float(a); complex z = -complex(a); float g = float(z);
        bool c = f == {-minimum}.0; bool d = g == {-minimum}.0;
        const int N = {qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH};
        int[N] native = -2; int x = 0;
        for int N in [0:1] {{ x += -native; }}''',
                  ('U1', 'C1', 'D1', 'X1'), [str(-minimum), '1', '1', '4'])
    for expression in ('uint(-a)', 'float(-a)', 'complex(-a)', '-(-a)'):
        target = 'complex' if expression.startswith('complex') else 'float'
        check_error(bra, f'int a = {minimum}; {target} out = {expression};', 'assertion failed in ASSERT')
    check_program(bra, f'''int a = {minimum}; array[int, 1] v = {{a}}; int i = 0; int bad = 99;
        bool c = false && -v[i] > 0; bool d = true || -int(a) == 0;
        bool e = false && bool(-v[bad]);
        int x = 0; while (false && bool(-a)) {{ x = -a; }}
        if (false) {{ x = -v[i]; }} else {{ x = -x; }}''',
                  ('C1', 'D1', 'E1', 'X1'), ['0', '1', '0', '0'])
    check_program(bra, '''include "stdgates.inc"; qubit q; int n = -1;
        rx(float(-n) * pi) q; bit b = measure q;
        int x = 3; if (b) { x = -x; }''', ('B1', 'X1'), ['1', '-3'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    converter = qasm2qcx.QASM2QCXConverter
    minimum, maximum = converter.QCX_INT_MIN, converter.QCX_INT_MAX
    for value in [minimum + 1, minimum + 2, maximum - 1, maximum, *range(-16, 17)]:
        check_program(bra, f'''const int C = -({value}); int x = C;
            int a = {value}; int y = -a;''', ('X1', 'Y1'), [str(-value), str(-value)])
    check_program(bra, f'''int x = {minimum}; const int C = -0x{-minimum:x};
        int y = C; int z = -(-(-1));''', ('X1', 'Y1', 'Z1'), [str(minimum), str(minimum), '-1'])
    for source in (f'int a = {minimum}; int x = -a;', f'int x = -({minimum});',
                   f'const int C = {minimum}; int x = -C;',
                   f'array[int, 1] v = {{{minimum}}}; int i = 0; int x = -v[i];'):
        check_error(bra, source, 'assertion failed in ASSERT')
    for source in (f'const int C = -({minimum});', f'const int C = -({maximum} + 1);'):
        try:
            qasm2qcx.convert('OPENQASM 3.0; ' + source)
        except qasm2qcx.InvalidNegationException:
            pass
        else:
            raise RuntimeError('Invalid constant negation was accepted')
    check_program(bra, f'''int a = {minimum}; int x = 7;
        bool c = false && bool(-a); bool d = true || bool(-a);
        if (false) {{ x = -({minimum}); }}
        while (false) {{ x = -a; }}
        const int C = {minimum}; const bool OFF = false && bool(-C);
        bool e = OFF;''', ('X1', 'C1', 'D1', 'E1'), ['7', '0', '1', '0'])
    check_program(bra, '''array[int, 2] v = {-3, 5}; int i = 1;
        int x = -v[i]; int y = -(x + v[i - 1]);
        uint[4] u = 1; uint[4] w = -u; uint n = uint(-1); uint m = -n;''',
                  ('X1', 'Y1', 'W1', 'M1'), ['-5', '8', '15', '1'])
    check_integration(bra)
    print('Checked INT negation numerical tests passed')


if __name__ == '__main__':
    main()
