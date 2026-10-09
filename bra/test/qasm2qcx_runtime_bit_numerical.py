#!/usr/bin/env python3

"""Execute runtime-indexed bit-register reads, writes, and measurements with bra."""

import argparse
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def check_regressions(bra: pathlib.Path) -> None:
    outputs = ('FLAGS31:0', 'FLAGS31:1', 'FLAGS31:2')
    # Overlapping source and destination reads must use the pre-store values.
    for pattern in range(8):
        bits = [(pattern >> index) & 1 for index in range(3)]
        for target in range(-3, 3):
            for source in range(-3, 3):
                expected = bits.copy()
                expected[target] = bits[source] and not bits[target]
                check_program(bra, f'''bit[3] flags = "{pattern:03b}";
                    int i = {target}; int j = {source};
                    flags[i] = flags[j] && !flags[i];''', (*outputs, 'I1', 'J1'),
                              [str(int(bit)) for bit in expected] + [str(target), str(source)])

    # Live reads, negative runtime bounds, and nearest-loop transfers.
    for pattern in range(8):
        for skip in (-2, -1, 1):
            for stop in (-1, 0, 2):
                expected = [(pattern >> index) & 1 for index in range(3)]
                total, visits = 0, 0
                for index in range(-3, 3):
                    visits += 1
                    if index == skip:
                        continue
                    expected[index] = 1 - expected[index]
                    total += expected[index]
                    if index == stop:
                        break
                check_program(bra, f'''bit[3] flags = "{pattern:03b}";
                    int first = -3; int last = 2; int total = 0; int visits = 0;
                    for int i in [first:last] {{
                        visits += 1; if (i == {skip}) {{ continue; }}
                        flags[i] = !flags[i]; total += int(flags[i]);
                        if (i == {stop}) {{ break; }}
                    }}''', (*outputs, 'TOTAL31', 'VISITS63'),
                              [*map(str, expected), str(total), str(visits)])

    check_program(bra, '''bit[3] flags = "101"; int i = -1;
        array[int, 3] a = {0, 1, 2};
        flags[a[int(flags[i])]] = flags[a[i]] && !flags[int(flags[i])];''', outputs, ['1', '1', '1'])
    check_program(bra, '''bit[3] flags = "101"; float f = -1.5; bool b = true;
        bit out = flags[int(f)]; flags[uint(b)] = out;
        int total = int(flags[int(b)]) + int(flags[int(f)]);''',
                  (*outputs, 'OUT7', 'TOTAL31'), ['1', '1', '1', '1', '2'])
    check_program(bra, '''bit[3] flags = "000"; int first = 0; int last = 2;
        for int i in [first:2:last] { flags[i] = 1; }
        for int i in {first, last} { flags[i] = !flags[i]; }
        int i = -3; while (i < 0) { flags[i] = !flags[i]; i += 1; }''', outputs, ['1', '1', '1'])
    check_program(bra, '''include "stdgates.inc"; qubit q; bit[3] flags = "000";
        array[int, 3] positions = {-3, -2, -1}; int total = 0;
        for int i in positions {
            reset q; if (i != -2) { x q; }
            flags[i] = measure q; total += int(flags[i]);
        }''', (*outputs, 'TOTAL31'), ['1', '0', '1', '2'])
    check_program(bra, '''include "stdgates.inc"; qubit q; x q;
        bit[2] flags = "10"; array[int, 2] a = {0, 1}; int i = 0;
        flags[a[int(flags[i])]] = measure q;''', ('FLAGS31:0', 'FLAGS31:1'), ['1', '1'])
    check_program(bra, '''bit[2] flags = "01"; int bad = 2; int zero = 0;
        flags[0] = false && flags[bad];
        flags[1] = true || flags[1 / zero];
        bool ready = false && flags[bad] || flags[1];
        if (ready || flags[bad]) { flags[0] = !flags[0]; }''',
                  ('FLAGS31:0', 'FLAGS31:1', 'READY31'), ['1', '1', '1'])

    for index in (qasm2qcx.QASM2QCXConverter.QCX_INT_MIN,
                  qasm2qcx.QASM2QCXConverter.QCX_INT_MAX):
        for statement in ('flags[i] = 1;', 'flags[i] = measure q;', 'measure q -> flags[i];'):
            check_error(bra, f'qubit q; bit[2] flags; int i = {index}; {statement}',
                        'assertion failed in ASSERT')
    # An invalid nested index fails before attempting the outer access.
    for statement in ('bit out = flags[int(flags[bad])];',
                      'flags[int(flags[bad])] = 1;',
                      'flags[int(flags[bad])] = measure q;'):
        check_error(bra, 'qubit q; bit[2] flags; int bad = 2; ' + statement,
                    'assertion failed in ASSERT')
    for statement in ('flags[1 / zero] = true;', 'flags[1 / zero] = measure q;'):
        check_error(bra, 'qubit q; bit[2] flags; int zero = 0; ' + statement,
                    'integer division by zero in LET')
    check_error(bra, 'bit[2] flags; int i = 0; int bad = 2; flags[i] = flags[bad];',
                'assertion failed in ASSERT')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    for size in (1, 2, 3):
        for pattern in range(1 << size):
            for index in range(-size, size):
                value = (pattern >> (index % size)) & 1
                check_program(bra, f'''bit[{size}] flags = "{pattern:0{size}b}";
                    int i = {index}; bit out = flags[i]; bool ready = flags[i];
                    int total = int(flags[i]);''', ('OUT7', 'READY31', 'TOTAL31', 'I1'),
                              [str(value), str(value), str(value), str(index)])
        for index in (-size - 1, size, qasm2qcx.QASM2QCXConverter.QCX_INT_MIN,
                      qasm2qcx.QASM2QCXConverter.QCX_INT_MAX):
            check_error(bra, f'bit[{size}] flags; int i = {index}; bit out = flags[i];',
                        'assertion failed in ASSERT')

    check_program(bra, '''bit[3] flags = "101"; int i = -1;
        array[int, 2] a = {0, 1}; bit out = flags[int(flags[i])];
        int total = int(flags[a[0]]) + int(flags[a[1]]);
        out = flags[a[1]];''', ('OUT7', 'TOTAL31', 'I1'), ['0', '1', '-1'])
    check_program(bra, '''bit[3] flags = "101"; array[int, 3] a = {0, 1, 2}; int total = 0;
        for int i in a { if (flags[i]) { total += 1; } }
        for bit value in flags { total += int(flags[int(value)]); }
        int i = 0; while (i < 3) { total += int(flags[i]); i += 1; }''',
                  ('TOTAL31',), ['5'])
    check_program(bra, '''bit[2] flags = "01"; int bad = 2; int zero = 0;
        bool ready = false && flags[bad]; ready = true || flags[1 / zero];
        if (false) { ready = flags[bad]; }
        while (false) { ready = flags[bad]; }
        int total = int(ready);''', ('TOTAL31',), ['1'])
    check_error(bra, 'bit[2] flags; int zero = 0; bit out = flags[1 / zero];',
                'integer division by zero in LET')

    for size in (1, 2, 3):
        outputs = (('FLAGS31',) if size == 1
                   else tuple(f'FLAGS31:{index}' for index in range(size)))
        for index in range(-size, size):
            for value in (0, 1):
                expected = [0] * size
                expected[index] = value
                for rhs in (str(value), 'true' if value else 'false'):
                    check_program(bra, f'bit[{size}] flags; int i = {index}; flags[i] = {rhs};',
                                  (*outputs, 'I1'), [*map(str, expected), str(index)])
                for measurement in ('flags[i] = measure q;', 'measure q -> flags[i];'):
                    check_program(bra, f'''include "stdgates.inc"; qubit q;
                        bit[{size}] flags; int i = {index}; {'x q;' if value else ''}
                        {measurement}''', (*outputs, 'I1'), [*map(str, expected), str(index)])
        for index in (-size - 1, size):
            for statement in ('flags[i] = true;', 'flags[i] = measure q;',
                              'measure q -> flags[i];'):
                check_error(bra, f'qubit q; bit[{size}] flags; int i = {index}; {statement}',
                            'assertion failed in ASSERT')
        # Destination checks precede RHS errors, not just the final store.
        check_error(bra, f'''bit[{size}] flags; int i = {size}; int zero = 0;
            flags[i] = bool(1 / zero);''', 'assertion failed in ASSERT')

    check_program(bra, '''bit[3] flags = "101"; int i = 0;
        flags[i + 1] = flags[i];''', ('FLAGS31:0', 'FLAGS31:1', 'FLAGS31:2'), ['1', '1', '1'])
    check_program(bra, '''bit[3] flags = "001"; int i = 0;
        flags[int(flags[i])] = flags[i];''', ('FLAGS31:0', 'FLAGS31:1', 'FLAGS31:2'), ['1', '1', '0'])
    check_program(bra, '''bit[2] flags = "01"; int i = 0; int zero = 0;
        flags[i] = true || flags[1 / zero]; flags[i + 1] = !flags[i];''',
                  ('FLAGS31:0', 'FLAGS31:1'), ['1', '0'])
    check_program(bra, '''bit[2] flags = "01"; array[int, 2] a = {0, 1};
        for int i in a { flags[i] = !flags[i]; }
        for bit value in flags { flags[int(value)] = value; }''',
                  ('FLAGS31:0', 'FLAGS31:1'), ['0', '1'])
    check_program(bra, '''include "stdgates.inc"; qubit q; x q;
        bit[2] flags = "01"; int i = 0;
        flags[int(flags[i])] = measure q;''', ('FLAGS31:0', 'FLAGS31:1'), ['1', '1'])
    check_program(bra, '''qubit q; bit[2] flags = "01"; int bad = 2;
        if (false) { flags[bad] = measure q; }
        while (false) { flags[bad] = true; }''', ('FLAGS31:0', 'FLAGS31:1'), ['1', '0'])
    check_regressions(bra)
    print('Runtime bit-register indexing numerical tests passed')


if __name__ == '__main__':
    main()
