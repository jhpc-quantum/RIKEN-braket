#!/usr/bin/env python3

"""Execute static bit-selection operands in register expressions.

Example:
  python3 bra/test/qasm2qcx_static_bit_selection_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import resource

from qasm2qcx_bit_register_bitwise_numerical import bits, outputs
from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program


def selected(value: int, indices: tuple[int, ...]) -> int:
    return sum(((value >> index) & 1) << position for position, index in enumerate(indices))


def assigned(value: int, indices: tuple[int, ...], pattern: int) -> int:
    for position, index in enumerate(indices):
        value = (value & ~(1 << index)) | (((pattern >> position) & 1) << index)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    selections = (('a[0:1]', (0, 1)), ('a[3:-1:2]', (3, 2)),
                  ('a[{3, 0}]', (3, 0)), ('a[-2:-1]', (2, 3)),
                  ('a[2:2]', (2,)), ('a[{1}]', (1,)), ('a[{1, 1}]', (1, 1)),
                  ('a[{3, 0, 2}]', (3, 0, 2)), ('a[3:-1:0]', (3, 2, 1, 0)))
    for expression, indices in selections:
        width = len(indices)
        mask = (1 << width) - 1
        for value in range(16):
            view = selected(value, indices)
            other = mask ^ (mask >> 1)
            for count in (0, 1, width, width + 1):
                left = 0 if count >= width else (view << count) & mask
                right = 0 if count >= width else view >> count
                check_program(bra, f'''bit[4] a = "{value:04b}"; uint n = {count};
                    bit[{width}] b = "{other:0{width}b}";
                    bit[{width}] c = {expression} & b;
                    bit[{width}] d = {expression} | b;
                    bit[{width}] e = {expression} ^ b;
                    bit[{width}] f = ~{expression};
                    bit[{width}] g = {expression} << n;
                    bit[{width}] h = {expression} >> n;
                    bit[{width}] j = ~({expression} << n) ^ ({expression} >> 1);''',
                              sum((outputs(name, width) for name in ('C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'J1')), ()) + outputs('A1', 4),
                              sum((bits(result & mask, width) for result in
                                   (view & other, view | other, view ^ other, ~view, left, right, ~left ^ (view >> 1))), []) + bits(value, 4))
    check_program(bra, '''const int LOW = 1; const uint HIGH = 3;
        bit[4] a = "1010"; bit[2] b = a[{HIGH, LOW}] ^ "01";
        a = ~a[3:-1:0]; if (false) { b = a[0:1] << -1; }
        for int i in [0:1] { b = ~(a[0:1] ^ a[{3, 1}]); }''',
                  outputs('A1', 4) + outputs('B1', 2), bits(10, 4) + bits(2, 2))
    check_program(bra, '''bit[1] one = "1"; bit[1] a = ~one[0:0];
        bit[1] b = one[{0}] << uint(-1);''', ('A1', 'B1'), ['0', '0'])
    # Each operation starts from the same original register. Python computes
    # its entire source pattern before updating any selected destination bit.
    targets = (('a[0:1]', (0, 1)), ('a[1:2]', (1, 2)),
               ('a[{3, 0}]', (3, 0)), ('a[3:-1:2]', (3, 2)), ('a[{2}]', (2,)))
    sources = (('a[1:2]', (1, 2)), ('a[0:1]', (0, 1)),
               ('a[{0, 3}]', (0, 3)), ('a[{1, 1}]', (1, 1)), ('a[0:0]', (0,)))
    for target, target_indices in targets:
        for source, source_indices in sources:
            if len(target_indices) != len(source_indices):
                continue
            width = len(target_indices)
            mask = (1 << width) - 1
            for value in range(16):
                lhs, rhs = selected(value, target_indices), selected(value, source_indices)
                operations = (('C1', f'{target} = {source};', rhs),
                              ('D1', f'{target} = ~{source};', rhs ^ mask),
                              ('E1', f'{target} &= {source};', lhs & rhs),
                              ('F1', f'{target} |= {source};', lhs | rhs),
                              ('G1', f'{target} ^= {source};', lhs ^ rhs),
                              ('H1', f'{target} <<= n;', (lhs << 1) & mask),
                              ('J1', f'{target} >>= n;', lhs >> 1))
                program = f'bit[4] a; uint n = 1; '
                result_names, expected = (), []
                for name, statement, pattern in operations:
                    variable = name[0].lower()
                    program += f'a = "{value:04b}"; {statement} bit[4] {variable} = a; '
                    result_names += outputs(name, 4)
                    expected += bits(assigned(value, target_indices, pattern), 4)
                check_program(bra, program, result_names, expected)
    check_program(bra, '''bit[4] a = "1011"; a = a[3:-1:0];
        a[{0, 2}] <<= int(a[0]); a[1:2] = ~a[0:1];
        for int i in [0:1] { a[{3, 0}] ^= a[1:2]; }
        if (false) { a[0:1] >>= -1; }''', outputs('A1', 4), bits(14, 4))
    check_error(bra, 'bit[4] a = "1111"; int n = -1; bit[2] b = a[{3, 0}] << n;',
                'assertion failed in ASSERT')
    check_error(bra, 'bit[4] a = "1111"; a[0:1] >>= -1;', 'assertion failed in ASSERT')
    print('Static bit-selection numerical tests passed')


if __name__ == '__main__':
    main()
