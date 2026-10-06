#!/usr/bin/env python3

"""Execute converted constant-range for loops with bra.

Example:
  python3 bra/test/qasm2qcx_for_loop_numerical.py --bra bra/bin/bra
"""

import argparse
import importlib.util
import pathlib
import subprocess


CONVERTER_PATH = pathlib.Path(__file__).parents[1] / 'qcx' / 'qasm2qcx.py'
SPEC = importlib.util.spec_from_file_location('qasm2qcx', CONVERTER_PATH)
qasm2qcx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qasm2qcx)


def check_program(bra: pathlib.Path, source: str, outputs: tuple[str, ...],
                  expected: list[str]) -> None:
    lines = qasm2qcx.convert(source)
    lines.extend(f'PRINTLN {output}' for output in outputs)
    result = subprocess.run(
        [str(bra)], input='\n'.join(lines) + '\n', check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
    )
    if [line.strip() for line in result.stdout.splitlines()] != expected:
        raise RuntimeError(f'For-loop numerical test failed\nsource:\n{source}\n'
                           f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}')


def check_runtime_transfers(bra: pathlib.Path) -> None:
    # An independent Python oracle covers first/middle/final transfers, missing
    # targets, both range directions, striding, singleton and empty ranges.
    for bounds, values in (
            ('[0:4]', list(range(5))), ('[4:-1:0]', list(range(4, -1, -1))),
            ('[5:-2:-1]', [5, 3, 1, -1]), ('[2:2]', [2]), ('[2:1]', [])):
        targets = sorted({99, *(values[index] for index in (0, len(values) // 2, -1)
                                 if values)})
        for skip in targets:
            for stop in targets:
                total = visits = 0
                for value in values:
                    visits += 1
                    if value == skip:
                        continue
                    if value == stop:
                        break
                    total += value
                source = f'''OPENQASM 3.0; int skip = {skip}; int stop = {stop};
                    int total = 0; int visits = 0;
                    for int i in {bounds} {{
                        visits += 1;
                        if (i == skip) {{ continue; }}
                        if (i == stop) {{ break; }}
                        total += i;
                    }}
                    total += 100;'''
                check_program(bra, source, ('TOTAL31', 'VISITS63'),
                              [str(total + 100), str(visits)])


def check_measurement_transfers(bra: pathlib.Path) -> None:
    for transfer in ('break', 'continue'):
        for trigger in (0, 1, 3):
            source = f'''OPENQASM 3.0; include "stdgates.inc";
                qubit q; qubit r; bit outcome; bit tail;
                int sum = 0; int visits = 0;
                for int i in [0:3] {{
                    reset q;
                    if (i == {trigger}) {{ x q; }}
                    outcome = measure q;
                    visits += 1;
                    if (outcome) {{ {transfer}; }}
                    x r;
                    sum += i + 1;
                }}
                tail = measure r;'''
            if transfer == 'break':
                expected = [str(trigger * (trigger + 1) // 2),
                            str(trigger + 1), '1', str(trigger % 2)]
            else:
                expected = [str(10 - trigger - 1), '4', str(int(trigger == 3)), '1']
            check_program(bra, source, ('SUM7', 'VISITS63', 'OUTCOME127', 'TAIL15'), expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    source = '''OPENQASM 3.0; int sum = 0; int i = 99;
        for int i in [0:3] { sum += i; }
        for int i in [3:-2:0] { sum += i; }
        for int i in [3:0] { sum += i; }
        sum += i;'''
    check_program(arguments.bra, source, ('SUM7', 'I1'), ['109', '99'])

    source = '''OPENQASM 3.0; include "stdgates.inc";
        qubit[4] q; bit[4] flags; int sum = 0;
        for int i in [0:3] {
            if (i % 2 == 0) { x q[i]; }
            flags[i] = measure q[i];
            if (flags[i]) { sum += i + 1; }
            reset q[i];
        }
        for int i in [0:3] { flags[i] = measure q[i]; }
        if (flags[0] || flags[1] || flags[2] || flags[3]) { sum += 100; }'''
    check_program(arguments.bra, source, ('SUM7',), ['4'])

    source = '''OPENQASM 3.0; int sum = 0; int divisor = 0;
        for int i in [0:3] {
            if (i == 0 || false && 1 / divisor > 0) { sum += 1; }
            else { sum += i; }
        }
        sum += 10;'''
    check_program(arguments.bra, source, ('SUM7',), ['17'])

    source = '''OPENQASM 3.0; const int i = 8; int sum = 0; int value = 7;
        for int i in [0:2] { sum += value % (i + 1); value -= 1; }
        sum += i;'''
    check_program(arguments.bra, source, ('SUM7', 'VALUE31'), ['10', '4'])

    source = '''OPENQASM 3.0; int sum = 0;
        for int i in [0:2] {
            for int j in [0:i] { sum += i * 10 + j; }
        }'''
    check_program(arguments.bra, source, ('SUM7',), ['84'])

    source = '''OPENQASM 3.0; const int i = 9; int sum = 0;
        for int i in [1:2] {
            sum += i;
            for int i in [0:i] { sum += i; }
            sum += i;
        }
        sum += i;'''
    check_program(arguments.bra, source, ('SUM7',), ['19'])

    source = '''OPENQASM 3.0; include "stdgates.inc";
        qubit[4] q; bit[4] flags; int sum = 0;
        for int i in [0:1] {
            for int j in [0:1] {
                x q[2 * i + j];
                flags[2 * i + j] = measure q[2 * i + j];
                if (flags[2 * i + j]) { sum += 2 * i + j + 1; }
            }
        }'''
    check_program(arguments.bra, source,
                  ('FLAGS31:0', 'FLAGS31:1', 'FLAGS31:2', 'FLAGS31:3', 'SUM7'),
                  ['1', '1', '1', '1', '10'])

    source = '''OPENQASM 3.0; bool ready = false; int value = 7; int sum = 0;
        if (ready) {
            for int i in [0:1] { for int j in [0:1] { sum += value / j; } }
        }
        for int i in [0:1] { for int j in [1:2] { sum += value % j; } }
        sum += value;'''
    check_program(arguments.bra, source, ('SUM7',), ['9'])

    source = '''OPENQASM 3.0; int sum = 0;
        for int i in [2:-1:0] {
            for int j in [i:-1:1] { sum += i * 10 + j; }
        }'''
    check_program(arguments.bra, source, ('SUM7',), ['54'])

    # Complete classical example from bra/qcx/qasm2qcx.md.
    source = '''OPENQASM 3.0; int total = 0;
        for int i in [1:3] {
            for int j in [0:i] { total += i + j; }
        }'''
    check_program(arguments.bra, source, ('TOTAL31',), ['30'])

    source = '''OPENQASM 3.0; int total = 0;
        for int i in [0:5] {
            if (i == 2) { continue; }
            if (i == 4) { break; }
            total += i;
        }'''
    check_program(arguments.bra, source, ('TOTAL31',), ['4'])

    # The first declarations and the zero-divisor computations are skipped,
    # but later arithmetic must still be able to reuse the temporary storage.
    for transfer in ('break;', 'continue;'):
        source = '''OPENQASM 3.0; int value = 7; int sum = 0;
            for int i in [0:1] { ''' + transfer + ''' sum += value % 0; }
            sum += value + 1;'''
        check_program(arguments.bra, source, ('SUM7', 'VALUE31'), ['8', '7'])

    source = '''OPENQASM 3.0; int sum = 0;
        for int i in [0:1] {
            for int j in [0:2] {
                if (j == 0) { continue; }
                sum += 10 * i + j;
                break;
            }
            sum += i;
        }'''
    check_program(arguments.bra, source, ('SUM7',), ['13'])

    check_runtime_transfers(arguments.bra)
    check_measurement_transfers(arguments.bra)

    # Inner break, outer continue, and outer break must remain independent.
    source = '''OPENQASM 3.0; int total = 0;
        for int i in [0:3] {
            for int j in [0:2] {
                if (j == 1) { break; }
                total += i + j;
            }
            if (i == 1) { continue; }
            if (i == 2) { break; }
            total += 100;
        }'''
    check_program(arguments.bra, source, ('TOTAL31',), ['103'])

    source = '''OPENQASM 3.0; int i = 9; int total = 0;
        for int i in [1:2] {
            for int i in [0:i] {
                if (i == 0) { continue; }
                if (i == 1) { break; }
                total += 100;
            }
            total += i;
        }
        total += i;'''
    check_program(arguments.bra, source, ('TOTAL31', 'I1'), ['12', '9'])

    # The first iteration skips arithmetic and Boolean value temporaries of
    # different storage types; a later iteration and code after break reuse them.
    source = '''OPENQASM 3.0; int value = 7; float f = 1.5;
        complex c = 1.0im; bool ready = false;
        for int i in [0:3] {
            if (i == 0) { continue; }
            f = f + float(i);
            c = c + complex(f);
            ready = value % i == 0;
            if (ready) { break; }
        }
        f += float(value + 1);
        ready = !ready;'''
    check_program(arguments.bra, source, ('F1', ':REAL:C1', ':IMAG:C1', 'READY31'),
                  ['10.5', '2.5', '1', '0'])

    # A whole loop skipped by an enclosing runtime branch still declares the
    # storage required by later unrolled loops and arithmetic.
    source = '''OPENQASM 3.0; bool active = false; int value = 7; int total = 0;
        if (active) {
            for int i in [0:2] {
                if (i == 0) { continue; }
                total += value % i;
                break;
            }
        }
        for int i in [0:2] {
            if (i == 0) { continue; }
            total += value % i;
        }
        total += value + 1;'''
    check_program(arguments.bra, source, ('TOTAL31',), ['9'])

    source = '''OPENQASM 3.0; int total = 0; int visits = 0;
        for int i in {5, -1, 5, 0} { total += i; visits += 1; }'''
    check_program(arguments.bra, source, ('TOTAL31', 'VISITS63'), ['9', '4'])

    source = '''OPENQASM 3.0; const int n = 3; const uint u = 2; int total = 0;
        for int i in {n, u - 1, int(3.5)} { total = total * 10 + i; }'''
    check_program(arguments.bra, source, ('TOTAL31',), ['313'])

    source = '''OPENQASM 3.0; include "stdgates.inc"; qubit[6] q; bit[6] flags;
        for int i in {0, 2, 2, 5} { x q[i]; }
        flags = measure q;'''
    check_program(arguments.bra, source, ('FLAGS31:0', 'FLAGS31:2', 'FLAGS31:5'), ['1', '0', '1'])

    source = '''OPENQASM 3.0; int i = 9; int total = 0;
        for int i in {2, 3} {
            for int i in {i, i + 1} { total += i; }
            total += i;
        }
        total += i;'''
    check_program(arguments.bra, source, ('TOTAL31', 'I1'), ['26', '9'])

    source = '''OPENQASM 3.0; int skip = 2; int stop = 4; int total = 0;
        for int i in {1, 2, 3, 4, 5} {
            if (i == skip) { continue; }
            if (i == stop) { break; }
            total += i;
        }'''
    check_program(arguments.bra, source, ('TOTAL31',), ['4'])

    source = '''OPENQASM 3.0; int n = 0; int total = 0;
        while (n < 2) {
            n += 1;
            for int i in {0, 2, 2} {
                if (i == 0) { continue; }
                total += n + i;
            }
        }'''
    check_program(arguments.bra, source, ('N1', 'TOTAL31'), ['2', '14'])

    for transfer in ('break;', 'continue;'):
        source = '''OPENQASM 3.0; int value = 7; int total = 0;
            for int i in {2, 2} { ''' + transfer + ''' total += value % 0; }
            total += value + 1;'''
        check_program(arguments.bra, source, ('TOTAL31',), ['8'])

    for transfer in ('break', 'continue'):
        source = '''OPENQASM 3.0; include "stdgates.inc";
            qubit q; bit outcome = 0; int visits = 0; int total = 0;
            for int i in {0, 2, 5} {
                visits += 1; reset q;
                if (i == 2) { x q; }
                outcome = measure q;
                if (outcome) { ''' + transfer + '''; }
                total += i;
            }'''
        expected = ['2', '0', '1'] if transfer == 'break' else ['3', '5', '0']
        check_program(arguments.bra, source, ('VISITS63', 'TOTAL31', 'OUTCOME127'), expected)


if __name__ == '__main__':
    main()
