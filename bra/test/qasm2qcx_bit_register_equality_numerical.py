#!/usr/bin/env python3

"""Compare literal and runtime bit-register equality against Python results.

Example:
  python3 bra/test/qasm2qcx_bit_register_equality_numerical.py --bra bra/bin/bra
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
    for width in (1, 2, 4, 33, 65):
        mask = (1 << width) - 1
        values = range(1 << width) if width <= 4 else (0, 1, 3, 1 << (width - 1), mask)
        for left in values:
            for right in values:
                check_program(bra, f'''bit[{width}] a = "{left:0{width}b}";
                    bit[{width}] b = "{right:0{width}b}";
                    const bool SAME = "{left:0{width}b}" == "{right:0{width}b}";
                    const bool DIFF = "{left:0{width}b}" != "{right:0{width}b}";
                    bool c = a == b; bool d = a != b; bool e = SAME; bool f = DIFF;
                    bool g = (~a ^ b) == (a | b); bool h = "{left:0{width}b}" != b;
                    bool j = (a << 1) == (b >> 1);''',
                              ('C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'J1'),
                              [str(int(result)) for result in (left == right, left != right, left == right,
                               left != right, ((~left ^ right) & mask) == (left | right), left != right,
                               ((left << 1) & mask) == (right >> 1))])
    check_program(bra, '''bit[4] a = "1010"; bit[4] b = "0101"; int n = 1;
        bool same = a[3:-1:0] == b; bool diff = a[{3, 0}] != a[0:1];
        bool shifted = (a[0:1] << n) == (b[{3, 0}] >> n);
        bit[1] one = "1"; bool single = a[1:1] == one;
        const bool OFF = false && (("01" << -1) == "00"); bool folded = OFF;''',
                  ('SAME15', 'DIFF15', 'SHIFTED127', 'SINGLE63', 'FOLDED63'), ['1', '1', '0', '1', '0'])
    # A known mismatching first bit must not skip evaluating the RHS shift.
    check_error(bra, 'bit[4] a = "0001"; int n = -1; bool same = "0000" == (a << n);',
                'assertion failed in ASSERT')
    check_program(bra, '''bit[4] a = "0011"; int x = 0;
        while (!(a == "0000")) { a <<= 1; x += 1; }
        bool c = (a == "0000") == true;
        bool d = bool(a != "0001");
        bool e = (a << int(a == "0000")) == a;''',
                  ('X1', 'C1', 'D1', 'E1'), ['4', '1', '1', '1'])
    check_program(bra, '''bit[4] a = "1010"; bit[4] b = "0101";
        int x = 0; int y = 0;
        for int i in [0:2] {
            if (a[3:-1:0] == b && !(a == b)) { x += 1; }
            while (~a == b) {
                y += 1;
                if (y < 2 && a != b) { continue; }
                break;
            }
        }''', ('X1', 'Y1'), ['3', '4'])
    # Runtime failures inside the equality operand are skipped only by the
    # surrounding logical expression, not by an earlier mismatching bit.
    check_program(bra, '''bit[2] a = "01"; int n = -1; int z = 0;
        array[int, 1] counts = {0}; int k = 9;
        bool c = false && ((a << n) == "00");
        bool d = true || ((a >> n) != "11");
        bool e = false && ((a << (1 / z)) == "00");
        bool f = true || ((a << counts[k]) == "00");
        if (c || !(a != "00")) { z = 1; } else { z = 2; }
        while (false && ((a << n) == a)) { z = 3; }''',
                  ('C1', 'D1', 'E1', 'F1', 'Z1'), ['0', '1', '0', '1', '2'])
    for condition in ('true && ((a << n) == "00")',
                      'false || ((a >> n) != "11")'):
        check_error(bra, f'bit[2] a = "01"; int n = -1; bool c = {condition};',
                    'assertion failed in ASSERT')
    print('Bit-register equality numerical tests passed')


if __name__ == '__main__':
    main()
