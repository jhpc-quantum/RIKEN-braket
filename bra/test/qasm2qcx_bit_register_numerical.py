#!/usr/bin/env python3

"""Execute converted bit-register loops with bra.

Example:
  python3 bra/test/qasm2qcx_bit_register_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib

from qasm2qcx_runtime_for_numerical import check_program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra

    # Bit zero is the rightmost bit of the source bit string.
    for width in (1, 2, 3):
        for pattern in range(1 << width):
            bits = [(pattern >> index) & 1 for index in range(width)]
            ordered = 0
            for value in bits:
                ordered = ordered * 10 + value
            check_program(bra, f'''bit[{width}] flags = "{pattern:0{width}b}";
                int total = 0; int ordered = 0; int visits = 0;
                for bit value in flags {{
                    total += int(value); ordered = ordered * 10 + int(value); visits += 1;
                }}''', ('TOTAL31', 'ORDERED127', 'VISITS63'),
                          [str(sum(bits)), str(ordered), str(width)])

    check_program(bra, '''bit[3] flags = "111"; int total = 0;
        for bit value in flags { flags[1] = 0; total += int(value); }''', ('TOTAL31',), ['2'])
    check_program(bra, '''bit[3] flags = "111"; int total = 0;
        for bit value in flags { flags[0] = 0; total += int(value); }''', ('TOTAL31',), ['3'])
    check_program(bra, '''bit[3] flags = "101"; bit out = 0; bool ready = false; int total = 0;
        for bit value in flags {
            out = value; ready = value && !ready;
            if (out && value) { total += int(bool(value)); }
        }''', ('TOTAL31', 'OUT7', 'READY31'), ['2', '1', '1'])
    check_program(bra, '''bit[3] flags = "101"; int total = 0;
        for bit flags in flags { total += int(flags); } total += int(flags[0]);''', ('TOTAL31',), ['3'])
    check_program(bra, '''bit[3] flags = "101"; int visits = 0; int total = 0;
        for bit value in flags {
            visits += 1; if (!value) { continue; }
            total += 1; if (visits == 3) { break; }
        }''', ('VISITS63', 'TOTAL31'), ['3', '2'])
    check_program(bra, '''bit[2] flags = "01"; array[int, 2] values = {2, 3}; int total = 0;
        for bit value in flags {
            for int i in values { total += i * int(value); }
            for int j in {int(value), int(!value)} { total += j; }
            for bit other in flags { if (other) { total += int(value); } }
        }''', ('TOTAL31',), ['8'])
    check_program(bra, '''bit[2] flags = "01"; int total = 0; int tick = 0;
        while (tick < 2) {
            for bit value in flags { total += int(value); }
            flags[1] = 1; tick += 1;
        }''', ('TOTAL31',), ['3'])
    check_program(bra, '''const int N = 2; bit[N] flags = "01"; int total = 0;
        for bit N in flags { for bit value in flags { total += int(value); } }''', ('TOTAL31',), ['2'])
    check_program(bra, '''include "stdgates.inc"; bit[3] flags = "101"; qubit q; bit outcome = 0;
        int visits = 0;
        for bit value in flags {
            visits += 1; if (value) { x q; }
            outcome = measure q; if (outcome) { break; }
        }''', ('VISITS63', 'OUTCOME127'), ['1', '1'])


if __name__ == '__main__':
    main()
