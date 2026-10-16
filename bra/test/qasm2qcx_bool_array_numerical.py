#!/usr/bin/env python3

"""Execute Boolean array initialization, access, and logical expressions.

Example:
  python3 bra/test/qasm2qcx_bool_array_numerical.py --bra bra/bin/bra
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

    for size in (1, 2, 3):
        for pattern in range(1 << size):
            values = [bool(pattern & (1 << i)) for i in range(size)]
            literal = ', '.join(str(value).lower() for value in values)
            for index in range(-size, size):
                expected = str(int(values[index]))
                check_program(bra, f'array[bool, {size}] flags = {{{literal}}}; int i = {index}; '
                              f'bool ready = flags[i]; bool other = flags[{index}];',
                              ('READY31', 'OTHER31', 'I1'), [expected, expected, str(index)])
                changed = values.copy()
                changed[index] = not values[index]
                check_program(bra, f'array[bool, {size}] flags = {{{literal}}}; int i = {index}; '
                              'flags[i] = !flags[i];', tuple(f'FLAGS31:{i}' for i in range(size)) + ('I1',),
                              [str(int(value)) for value in changed] + [str(index)])

    operations = {'&&': lambda a, b: a and b, '||': lambda a, b: a or b,
                  '==': lambda a, b: a == b, '!=': lambda a, b: a != b}
    for left in (False, True):
        for right in (False, True):
            for operator, operation in operations.items():
                expected = str(int(operation(left, right)))
                check_program(bra, f'array[bool, 2] flags = {{{str(left).lower()}, {str(right).lower()}}}; '
                              'int i = 0; int j = 1; '
                              f'bool ready = flags[i] {operator} flags[j]; int count = 0; '
                              f'if (flags[i] {operator} flags[j]) {{ count = 1; }}',
                              ('READY31', 'COUNT31'), [expected, expected])

    check_program(bra, '''array[bool, 2] flags = {false, true}; int i = 0; int j = 1;
        flags[i] = flags[j] && !flags[i];
        flags[j] = flags[i] || !flags[j];
        array[bool, 2] copies = {flags[i], !flags[j]};''',
                  ('FLAGS31:0', 'FLAGS31:1', 'COPIES63:0', 'COPIES63:1', 'I1', 'J1'),
                  ['1', '1', '1', '0', '0', '1'])

    check_program(bra, '''array[bool, 2] flags = {true, false}; int i = 0; int j = 1;
        bit b = flags[i]; bit[2] bits = "00"; bits[i] = flags[i]; bits[j] = flags[j];
        flags[j] = bits[i]; b = flags[j];''',
                  ('B1', 'BITS15:0', 'BITS15:1', 'FLAGS31:1'), ['1', '1', '0', '1'])
    check_program(bra, '''array[bool, 2] flags = {true, false}; int i = 0;
        int n = int(flags[i]); uint u = uint[4](flags[i]);
        float f = float(flags[i]); complex z = complex(flags[i]); float realpart = float(z);
        array[int, 2] values = {3, 7}; int selected = values[int(flags[i])];
        flags[int(flags[i])] = !flags[i];''',
                  ('N1', 'U1', 'F1', 'REALPART255', 'SELECTED255', 'FLAGS31:1'),
                  ['1', '1', '1', '1', '7', '0'])

    check_program(bra, '''array[bool, 2] flags = {true, false}; int bad = 2; int zero = 0;
        flags[0] = true || flags[bad]; flags[1] = false && flags[1 / zero];
        if (false) { flags[bad] = flags[bad]; }
        while (false) { flags[1 / zero] = false; }
        bool ready = flags[0] || flags[bad];
        ready = flags[1] && flags[1 / zero];''', ('FLAGS31:0', 'FLAGS31:1', 'READY31'), ['1', '0', '0'])
    check_program(bra, '''array[bool, 3] flags = {true, true, true}; int count = 0; int i = 0;
        while (i < 3 && flags[i]) { count += 1; i += 1; }
        int stop = 2; for int j in [0:stop] { if (flags[j]) { flags[j] = false; } }''',
                  ('COUNT31', 'FLAGS31:0', 'FLAGS31:1', 'FLAGS31:2'), ['3', '0', '0', '0'])

    check_program(bra, '''include "stdgates.inc"; qubit q; x q; bit b = measure q;
        array[bool, 2] flags = {b, !b}; int i = 0;
        if (flags[i]) { x q; } bit result = measure q; int n = int(result);''', ('N1',), ['0'])
    check_program(bra, '''include "stdgates.inc"; qubit q; array[bool, 1] flags = {true}; int i = 0;
        rx(float(flags[i]) * pi) q; bit result = measure q; int n = int(result);''', ('N1',), ['1'])

    for size in (1, 2, 3):
        for index in (-size - 1, size):
            for statement in ('bool ready = flags[i];', 'flags[i] = true;',
                              'flags[i] = false && flags[i];'):
                check_error(bra, f'array[bool, {size}] flags; int i = {index}; ' + statement,
                            'assertion failed in ASSERT')
    check_error(bra, 'array[bool, 1] flags = {true}; int zero = 0; bool b = flags[1 / zero];',
                'integer division by zero in LET')

    print('Boolean array numerical checks passed')


if __name__ == '__main__':
    main()
