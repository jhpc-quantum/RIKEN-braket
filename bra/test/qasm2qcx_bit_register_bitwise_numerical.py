#!/usr/bin/env python3

"""Execute whole-register bitwise expressions and compare every result bit.

Example:
  python3 bra/test/qasm2qcx_bit_register_bitwise_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib

from qasm2qcx_runtime_for_numerical import check_program


def outputs(name: str, width: int) -> tuple[str, ...]:
    return (name,) if width == 1 else tuple(f'{name}:{index}' for index in range(width))


def bits(value: int, width: int) -> list[str]:
    return [str((value >> index) & 1) for index in range(width)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    for width in (1, 2, 3, 4):
        mask = (1 << width) - 1
        for a in range(1 << width):
            for b in range(1 << width):
                check_program(bra, f'''bit[{width}] a = "{a:0{width}b}";
                    bit[{width}] b = "{b:0{width}b}";
                    bit[{width}] c = a & b; bit[{width}] d = a | b;
                    bit[{width}] e = a ^ b; bit[{width}] f = ~a;
                    bit[{width}] g = ~(a ^ b) & (a | "{b:0{width}b}");
                    bit[{width}] h = ~("{a:0{width}b}" ^ "{b:0{width}b}");
                    a = ~(a & b) ^ a; b = a | ~b;''',
                              sum((outputs(name, width) for name in ('C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'A1', 'B1')), ()),
                              sum((bits(value & mask, width) for value in
                                   (a & b, a | b, a ^ b, ~a, ~(a ^ b) & (a | b), ~(a ^ b),
                                    ~(a & b) ^ a, (~(a & b) ^ a) | ~b)), []))
    # Widths larger than native integers remain bit arrays, not packed INT values.
    for width in (33, 65):
        a = (1 << (width - 1)) | 3
        b = (1 << (width - 2)) | 5
        check_program(bra, f'''bit[{width}] a = "{a:0{width}b}";
            bit[{width}] b = "{b:0{width}b}"; a = ~(a ^ b);''',
                      outputs('A1', width), bits(~(a ^ b), width))
    check_program(bra, '''bit[4] a = "1010"; bit[4] b = "1100";
        if (false) { a = ~b; } else { b = a ^ b; }
        a = a | "0001";''', outputs('A1', 4) + outputs('B1', 4), bits(11, 4) + bits(6, 4))
    print('Whole-bit-register bitwise expression numerical tests passed')


if __name__ == '__main__':
    main()
