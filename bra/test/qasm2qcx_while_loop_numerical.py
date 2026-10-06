#!/usr/bin/env python3

"""Execute converted runtime while loops with bra.

Example:
  python3 bra/test/qasm2qcx_while_loop_numerical.py --bra bra/bin/bra
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
    lines = qasm2qcx.convert('OPENQASM 3.0; ' + source)
    lines.extend(f'PRINTLN {output}' for output in outputs)
    result = subprocess.run(
        [str(bra)], input='\n'.join(lines) + '\n', check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
    )
    if [line.strip() for line in result.stdout.splitlines()] != expected:
        raise RuntimeError(f'While-loop numerical test failed\nsource:\n{source}\n'
                           f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}')


def check_runtime_transfers(bra: pathlib.Path) -> None:
    # Independent Python loops check entry/exit values as well as accumulation.
    for start, stop, step in ((-2, 3, 1), (4, -1, -2), (0, 1, 1),
                              (1, 1, 1), (3, 1, 1)):
        values = list(range(start, stop, step))
        targets = sorted({99, *(values[index] for index in (0, len(values) // 2, -1)
                                 if values)})
        comparison = '<' if step > 0 else '>'
        for skip in targets:
            for finish in targets:
                n, visits, total = start, 0, 0
                while (n < stop if step > 0 else n > stop):
                    entry = n
                    n += step
                    visits += 1
                    if entry == skip:
                        continue
                    if entry == finish:
                        break
                    total += entry
                source = f'''int n = {start}; int entry = 0; int visits = 0;
                    int total = 0; int skip = {skip}; int finish = {finish};
                    while (n {comparison} {stop}) {{
                        entry = n; n += {step}; visits += 1;
                        if (entry == skip) {{ continue; }}
                        if (entry == finish) {{ break; }}
                        total += entry;
                    }}
                    total += 100;'''
                check_program(bra, source, ('N1', 'VISITS63', 'TOTAL31'),
                              [str(n), str(visits), str(total + 100)])


def check_condition_types(bra: pathlib.Path) -> None:
    cases = (
        ('bool ready = true;', 'ready', 'ready = false;', ('READY31',), ['0']),
        ('bit flag = 1;', 'flag', 'flag = 0;', ('FLAG15',), ['0']),
        ('bit[2] flags = "01";', 'flags[0]', 'flags[0] = 0;', ('FLAGS31:0',), ['0']),
        ('int n = 2;', 'bool(n)', 'n -= 1;', ('N1',), ['0']),
        ('float f = 1.0;', 'bool(f)', 'f -= 0.5;', ('F1',), ['0']),
        ('float f = 1.0;', '0 < f', 'f -= 0.5;', ('F1',), ['0']),
        ('uint n = 2;', 'n > 0', 'n -= 1;', ('N1',), ['0']),
        ('int n = 2; float f = 0.0;', 'n > f', 'n -= 1;', ('N1',), ['0']),
        ('int n = 0;', 'n < 3 || 1 / (n - 2) > 1', 'n += 1;', ('N1',), ['3']),
        ('int n = 2;', '!(n == 0) && bool(n + 1)', 'n -= 1;', ('N1',), ['0']),
    )
    for declarations, condition, body, outputs, expected in cases:
        check_program(bra, f'{declarations} while ({condition}) {{ {body} }}', outputs, expected)


def check_measurement_transfers(bra: pathlib.Path) -> None:
    for transfer in ('break', 'continue'):
        for trigger in (1, 2, 4):
            source = f'''include "stdgates.inc";
                qubit q; qubit r; bit outcome = 0; bit tail;
                int visits = 0; int total = 0;
                while (!outcome) {{
                    visits += 1;
                    reset q;
                    if (visits == {trigger}) {{ x q; }}
                    barrier q;
                    outcome = measure q;
                    if (outcome) {{ {transfer}; }}
                    x r;
                    total += visits;
                }}
                tail = measure r;'''
            check_program(bra, source, ('VISITS63', 'TOTAL31', 'OUTCOME127', 'TAIL15'),
                          [str(trigger), str(trigger * (trigger - 1) // 2),
                           '1', str((trigger - 1) % 2)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    check_program(arguments.bra, '''int n = 0; int total = 0;
        while (n < 5) { n += 1; total += n; }''', ('N1', 'TOTAL31'), ['5', '15'])

    # Neither the body nor its zero-divisor computation executes; the following
    # expression must still be able to reuse the temporary storage.
    check_program(arguments.bra, '''int value = 7; int total = 0;
        while (false) { total += value % 0; }
        total += value + 1;''', ('TOTAL31',), ['8'])

    check_program(arguments.bra, '''int n = 0; int total = 0;
        while (n < 6) {
            n += 1;
            if (n == 2) { continue; }
            if (n == 4) { break; }
            total += n;
        }
        total += 100;''', ('N1', 'TOTAL31'), ['4', '104'])

    # Arithmetic in the condition must be recomputed on every back edge.
    check_program(arguments.bra, '''int n = 5;
        while (n % 3 != 0) { n -= 1; }''', ('N1',), ['3'])

    # Short-circuiting keeps runtime zero divisors out of executed paths.
    check_program(arguments.bra, '''int n = 2; int total = 0;
        while (n > 0 && 4 / n > 0) { total += n; n -= 1; }''',
                  ('N1', 'TOTAL31'), ['0', '3'])

    check_program(arguments.bra, '''include "stdgates.inc";
        qubit q; bit outcome = 0; int visits = 0;
        while (!outcome) {
            visits += 1;
            if (visits == 2) { x q; }
            outcome = measure q;
        }''', ('VISITS63', 'OUTCOME127'), ['2', '1'])

    check_program(arguments.bra, '''int n = 0; int total = 0;
        while (n < 2) {
            n += 1;
            for int i in [0:2] {
                if (i == 0) { continue; }
                total += n + i;
                break;
            }
        }''', ('N1', 'TOTAL31'), ['2', '5'])

    check_program(arguments.bra, '''int n = 0; int total = 0;
        for int i in [1:2] {
            n = 0;
            while (n < 4) {
                n += 1;
                if (n == 1) { continue; }
                total += i;
                break;
            }
            total += 10;
        }''', ('N1', 'TOTAL31'), ['2', '23'])

    check_program(arguments.bra, '''int outer = 0; int inner = 0; int total = 0;
        while (outer < 2) {
            outer += 1;
            inner = 0;
            while (inner < 3) {
                inner += 1;
                if (inner == 1) { continue; }
                total += outer;
                break;
            }
            total += 10;
        }''', ('TOTAL31',), ['23'])

    # First use of body temporaries is skipped by continue. Later iterations
    # and code after break reuse the same storage safely.
    check_program(arguments.bra, '''int value = 7; int n = 0; int total = 0;
        while (n < 4) {
            n += 1;
            if (n == 1) { continue; }
            total += value % n;
            if (n == 3) { break; }
        }
        total += value + 1;''', ('N1', 'TOTAL31'), ['3', '10'])

    check_program(arguments.bra, '''bool ready = true; int n = 2;
        while (ready && bool(n)) { n -= 1; ready = n > 0; }''',
                  ('N1', 'READY31'), ['0', '0'])

    check_runtime_transfers(arguments.bra)
    check_condition_types(arguments.bra)
    check_measurement_transfers(arguments.bra)

    # Arithmetic and Boolean temporaries of different types are repeatedly
    # reused, including after a continue skips their first body use.
    check_program(arguments.bra, '''int value = 7; int n = 0; float f = 1.5;
        complex c = 1.0im; bool ready = false;
        while (n < 4 && !ready) {
            n += 1;
            if (n == 1) { continue; }
            f = f + float(n);
            c = c + complex(f);
            ready = value % n == 0;
            if (n == 3) { break; }
        }
        f += float(value + 1); ready = !ready;''',
                  ('N1', 'F1', ':REAL:C1', ':IMAG:C1', 'READY31'),
                  ['3', '14.5', '10', '1', '1'])

    # A skipped enclosing branch must not hide temporary declarations needed
    # by a later loop with arithmetic both in its condition and body.
    check_program(arguments.bra, '''bool active = false; int n = 5; int total = 0;
        if (active) {
            while (n > 0) { total += n % 0; break; }
        }
        while (n % 3 != 0) { total += n + 1; n -= 1; }
        total += n + 1;''', ('N1', 'TOTAL31'), ['3', '15'])

    # The inner for shadows a global used in the outer while condition; the
    # original runtime binding must be restored for the next condition check.
    check_program(arguments.bra, '''int i = 3; int total = 0;
        while (i > 0) {
            for int i in [0:2] {
                if (i == 0) { continue; }
                total += i;
                break;
            }
            i -= 1;
        }''', ('I1', 'TOTAL31'), ['0', '3'])

    # Both inner transfers and outer transfers occur in the same execution.
    # The inner break must leave the outer continue/break reachable.
    for inner in (
            '''for int i in [0:2] {
                if (i == 0) { continue; }
                total += n + i; break;
            }''',
            '''inner = 0; while (inner < 3) {
                inner += 1;
                if (inner == 1) { continue; }
                total += n + inner - 1; break;
            }'''):
        check_program(arguments.bra, '''int n = 0; int total = 0; int inner = 0;
            while (n < 4) {
                n += 1;
                ''' + inner + '''
                if (n == 2) { continue; }
                if (n == 3) { break; }
                total += 10;
            }''', ('N1', 'TOTAL31'), ['3', '19'])

    check_program(arguments.bra, '''include "stdgates.inc";
        qubit[2] q; bit[2] flags; int n = 0;
        for int i in [0:1] {
            n = 0;
            while (n < 2) {
                n += 1;
                if (n == 1) { continue; }
                x q[i];
            }
            flags[i] = measure q[i];
        }''', ('FLAGS31:0', 'FLAGS31:1'), ['1', '1'])


if __name__ == '__main__':
    main()
