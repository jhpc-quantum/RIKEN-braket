#!/usr/bin/env python3

"""Execute width-preserving UINT array reads and assignments with bra.

Example:
  python3 bra/test/qasm2qcx_uint_array_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def result_for(operator: str, lhs: int, rhs: int, width: int,
               rhs_width: int | None, native: int) -> int:
    mask = (1 << width) - 1
    if operator == '=':
        return rhs & mask
    if operator in ('<<=', '>>='):
        return (lhs << rhs if operator == '<<=' else lhs >> rhs) & mask
    unsigned = rhs_width is not None or width == native
    if unsigned:
        result_mask = (1 << max(width, rhs_width or native)) - 1
        lhs, rhs = lhs & result_mask, rhs & result_mask
    if operator == '+=':
        value = lhs + rhs
    elif operator == '-=':
        value = lhs - rhs
    elif operator == '*=':
        value = lhs * rhs
    elif operator in ('/=', '%='):
        quotient = abs(lhs) // abs(rhs)
        if (lhs < 0) != (rhs < 0):
            quotient = -quotient
        value = quotient if operator == '/=' else lhs - quotient * rhs
    elif operator == '&=':
        value = lhs & rhs
    elif operator == '|=':
        value = lhs | rhs
    elif operator == '^=':
        value = lhs ^ rhs
    elif operator == '**=':
        value = pow(lhs, rhs)
    else:
        raise ValueError(operator)
    return value & mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH

    for width in range(1, native + 1):
        mask = (1 << width) - 1
        check_program(bra, f'array[uint[{width}], 3] values = {{-1, {1 << width}, true}}; '
                      'int i = -3; uint x = values[i]; uint y = values[-2];',
                      ('X1', 'Y1', 'VALUES63:2', 'I1'), [str(mask), '0', '1', '-3'])

    for width in (1, 4, 8, native):
        seed = min((1 << width) - 1, 13)
        for operator in ('=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '**='):
            for rhs_type, rhs_width, rhs in (('int', None, 2), ('int', None, -2),
                                             ('uint[8]', 8, 2), ('uint', native, 2)):
                if operator in ('<<=', '>>='):
                    rhs = min(width - 1, 1)
                if operator == '**=' and rhs < 0:
                    continue
                expected = str(result_for(operator, seed, rhs, width, rhs_width, native))
                for index in ('0', 'i'):
                    check_program(bra, f'array[uint[{width}], 2] values = {{{seed}, 7}}; '
                                  f'uint[{width}] scalar = {seed}; {rhs_type} rhs = {rhs}; int i = 0; '
                                  f'values[{index}] {operator} rhs; scalar {operator} rhs;',
                                  ('VALUES63:0', 'SCALAR63', 'I1'), [expected, expected, '0'])

    # Intermediate element widths, not the final destination, control wrapping.
    check_program(bra, '''array[uint[4], 2] values = {15, 1}; int i = 0;
        uint sum = values[i] + values[1]; uint complement = ~values[i];
        uint negated = -values[i]; uint shifted = values[i] << 1;
        uint powered = values[i] ** uint[4](2);
        array[uint[8], 2] copies = {values[0], values[i] + values[1]};''',
                  ('SUM7', 'COMPLEMENT1023', 'NEGATED127', 'SHIFTED127', 'POWERED127',
                   'COPIES63:0', 'COPIES63:1'), ['0', '0', '1', '14', '1', '15', '0'])

    check_program(bra, '''array[uint[4], 2] values = {1, 3}; int i = 0;
        values[values[i]] += values[i] + uint[4](1);
        values[i] **= values[values[i]];
        values[1] <<= values[i];''', ('VALUES63:0', 'VALUES63:1', 'I1'), ['1', '10', '0'])
    check_program(bra, '''array[uint[4], 2] values = {0, 1}; int bad = 2; int zero = 0;
        bool flag = false && values[bad] > 0;
        flag = true || values[1 / zero] > 0;
        if (false) { values[bad] += 1; }
        while (false) { values[0] /= zero; }
        if (true) {} else { values[0] <<= 4; }''', ('VALUES63:0', 'VALUES63:1', 'FLAG15'), ['0', '1', '1'])

    for statement in ('uint x = values[i];', 'values[i] = 1;', 'values[i] += 1;', 'values[i] **= 2;'):
        for index in (-3, 2):
            check_error(bra, f'array[uint[4], 2] values; int i = {index}; ' + statement,
                        'assertion failed in ASSERT')
    for operator in ('/=', '%='):
        check_error(bra, 'array[uint[4], 2] values = {3, 4}; int i = 0; int zero = 0; '
                    f'values[i] {operator} zero;', 'integer division by zero in LET')
    for operator in ('<<=', '>>='):
        for count in (-1, 4):
            check_error(bra, f'array[uint[4], 2] values = {{1, 2}}; int i = 0; int count = {count}; '
                        f'values[i] {operator} count;', 'assertion failed in ASSERT')
    for exponent in (-1, native):
        check_error(bra, f'array[uint[4], 2] values = {{15, 1}}; int i = 0; int exponent = {exponent}; '
                    'values[i] **= exponent;', 'assertion failed in ASSERT')
    check_error(bra, 'array[uint, 1] values = {uint(-1)}; int i = 0; int x = int(values[i]);',
                'UINT to INT conversion out of range')
    check_error(bra, 'array[uint[4], 1] values; uint i = uint(-1); uint x = values[i];',
                'UINT to INT conversion out of range')

    print('UINT array numerical checks passed')


if __name__ == '__main__':
    main()
