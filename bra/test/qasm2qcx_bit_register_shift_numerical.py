#!/usr/bin/env python3

"""Compare folded and runtime whole-register shifts with Python bit patterns.

Example:
  python3 bra/test/qasm2qcx_bit_register_shift_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import resource

from qasm2qcx_bit_register_bitwise_numerical import bits, outputs
from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    native = qasm2qcx.QASM2QCXConverter
    for width in (1, 2, 4, 33, 65):
        mask = (1 << width) - 1
        values = range(1 << width) if width <= 4 else (0, 1, 3, 1 << (width - 1), mask)
        counts = sorted({0, 1, width - 1, width, width + 1, native.QCX_INT_MAX, native.QCX_UINT_MAX})
        for value in values:
            for count in counts:
                for operator in ('<<', '>>'):
                    expected = 0 if count >= width else ((value << count) if operator == '<<' else (value >> count)) & mask
                    for count_type in ('int', 'uint'):
                        if count_type == 'int' and count > native.QCX_INT_MAX:
                            continue
                        check_program(bra, f'''bit[{width}] a = "{value:0{width}b}";
                            {count_type} n = {count};
                            bit[{width}] b = "{value:0{width}b}" {operator} {count_type}({count});
                            bit[{width}] c = a {operator} n; bit[{width}] d = a;
                            d {operator}= n; a = a {operator} n;''',
                                      outputs('A1', width) + outputs('B1', width) + outputs('C1', width) + outputs('D1', width) + ('N1',),
                                      bits(expected, width) * 4 + [str(count)])
    check_program(bra, '''bit[4] a = "1011"; int i = 0; array[int, 2] counts = {1, 2};
        a = a << int(a[i]); a = a >> counts[i + 1];''', outputs('A1', 4) + ('I1',), bits(1, 4) + ['0'])
    check_program(bra, '''bit[4] a = "1011"; int i = 0; array[int, 2] counts = {1, 2};
        a <<= int(a[i]); a >>= counts[i + 1];''', outputs('A1', 4) + ('I1',), bits(1, 4) + ['0'])
    check_program(bra, '''bit[4] a = "1011"; bit[4] b = "1100"; int n = 1;
        bit[4] c = ~(a << n) ^ (b >> n);
        a = (a << n) >> n; b = (b >> n) << n;
        if (false) { a = "1011" << -1; }
        while (false) { b = b >> -1; }''',
                  outputs('A1', 4) + outputs('B1', 4) + outputs('C1', 4), bits(3, 4) + bits(12, 4) + bits(15, 4))
    check_program(bra, '''bit[4] a = "1111"; array[int, 2] counts = {0, 1}; int n = 2;
        for int count in counts { a >>= count; }
        for int j in [0:n] { if (j == 1) { continue; } a <<= uint(j); }
        while (n > 0) { a >>= 1; n -= 1; if (n == 1) { break; } }
        for int j in {n, n + 1} { a >>= j; }
        if (false) { a <<= -1; a >>= 1 / 0; }
        for int j in [1:0] { a <<= counts[999]; }
        bit[1] one = "1"; uint huge = uint(-1); one >>= huge;''',
                  outputs('A1', 4) + ('N1', 'ONE7'), bits(0, 4) + ['1', '0'])
    check_program(bra, '''include "stdgates.inc"; qubit q; x q;
        bit measured = measure q; bit[4] a = "1011";
        if (measured) { a <<= int(measured); }
        for bit value in a { if (value) { a >>= int(value); } }''',
                  outputs('A1', 4), bits(3, 4))
    # Execute the guide's example, including self-referencing count capture.
    check_program(bra, '''bit[4] a = "1011"; uint count = 1;
        bit[4] b = a << count; bit[4] c = a >> count;
        a <<= int(a[0]); a >>= 4;''',
                  outputs('A1', 4) + outputs('B1', 4) + outputs('C1', 4),
                  bits(0, 4) + bits(6, 4) + bits(5, 4))
    for source in ('bit[4] a = "1011"; int n = -1; a = a << n;',
                   'bit[4] a = "0000"; a = a >> -1;',
                   f'bit[1] a = "1"; int n = {native.QCX_INT_MIN}; a = a << n;',
                   'bit[4] a = "1011"; a <<= -1;',
                   'bit[4] a = "0000"; int n = -1; a >>= n;'):
        check_error(bra, source, 'assertion failed in ASSERT')
    print('Whole-bit-register shift numerical tests passed')


if __name__ == '__main__':
    main()
