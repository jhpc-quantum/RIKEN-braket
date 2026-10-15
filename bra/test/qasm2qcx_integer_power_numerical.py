#!/usr/bin/env python3

"""Compare folded and runtime integer powers against bounded Python results.

Example:
  python3 bra/test/qasm2qcx_integer_power_numerical.py --bra bra/bin/bra
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
    minimum, maximum, width = converter.QCX_INT_MIN, converter.QCX_INT_MAX, converter.QCX_UINT_WIDTH
    signed_cases = [(base, exponent) for base in range(-8, 9) for exponent in range(10)]
    signed_cases.extend(((minimum, 0), (minimum, 1), (minimum, 2), (maximum, 1),
                         (maximum, 2), (-2, width - 1), (2, width - 1),
                         (-2, width), (-1, maximum), (0, maximum), (1, maximum)))
    for base, exponent in signed_cases:
        expected = pow(base, exponent)
        runtime = f'int a = {base}; int n = {exponent}; int x = a ** n;'
        if not minimum <= expected <= maximum:
            check_error(bra, runtime, 'assertion failed in ASSERT')
            continue
        check_program(bra, runtime + f'const int C = ({base}) ** {exponent}; int y = C;',
                      ('X1', 'Y1'), [str(expected), str(expected)])
    for bits in (1, 4, 8, width):
        mask = (1 << bits) - 1
        for base in sorted({0, 1, 2 & mask, mask, 1 << (bits - 1)}):
            for exponent in sorted({0, 1, 2 & mask, 3 & mask, 4 & mask, 9 & mask, mask}):
                expected = pow(base, exponent, 1 << bits)
                check_program(bra, f'''uint[{bits}] a = {base}; uint[{bits}] n = {exponent};
                    uint[{bits}] x = a ** n;
                    const uint[{bits}] C = uint[{bits}]({base}) ** uint[{bits}]({exponent});
                    uint[{bits}] y = C;''', ('X1', 'Y1'), [str(expected), str(expected)])
    check_program(bra, '''int a = -2; uint[4] n = 3;
        int x = a ** n; uint[4] b = 3; int e = 4; int y = b ** e;
        uint u = 3; uint z = a ** u;''',
                  ('X1', 'Y1', 'Z1'), ['-8', '81', str((-8) & converter.QCX_UINT_MAX)])
    check_program(bra, f'''int n = -1; int a = 2; int z = 0;
        bool c = false && bool(a ** n);
        bool d = true || bool({maximum} ** 2);
        bool e = false && bool(a ** (1 / z));
        const bool OFF = false && bool(2 ** -1);
        bool f = OFF; if (false) {{ a = 2 ** -1; }}''',
                  ('A1', 'C1', 'D1', 'E1', 'F1'), ['2', '0', '1', '0', '0'])
    check_program(bra, '''int a = 2; int n = 3; int x = (a ** n) ** (n ** a);
        array[int, 2] values = {2, 3}; int i = 0; int y = values[i] ** values[i + 1];''',
                  ('X1', 'Y1'), [str(8 ** 9), '8'])
    for source in ('int n = -1; int x = 2 ** n;',
                   'int n = -1; uint x = uint(2) ** n;',
                   'int x = 2 ** -1;', 'uint x = uint(2) ** -1;',
                   f'int x = {maximum} ** 2;'):
        check_error(bra, source, 'assertion failed in ASSERT')
    check_error(bra, f'int a = {maximum}; int n = 2; int z = 0; int x = (a ** n) ** (1 / z);',
                'assertion failed in ASSERT')
    check_error(bra, 'int a = 0; int z = 0; int x = a ** (1 / z);',
                'integer division by zero in LET')
    print('Integer exponentiation numerical tests passed')


if __name__ == '__main__':
    main()
