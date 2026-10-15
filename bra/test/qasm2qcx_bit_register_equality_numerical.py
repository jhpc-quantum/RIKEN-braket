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
    print('Bit-register equality numerical tests passed')


if __name__ == '__main__':
    main()
