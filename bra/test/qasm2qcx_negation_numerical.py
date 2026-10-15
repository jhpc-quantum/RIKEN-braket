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
    print('Checked INT negation numerical tests passed')


if __name__ == '__main__':
    main()
