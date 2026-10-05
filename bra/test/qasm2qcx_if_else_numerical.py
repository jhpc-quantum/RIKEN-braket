#!/usr/bin/env python3

"""Exercise converted OpenQASM branches, Boolean values, and conversions.

Example:
  python3 bra/test/qasm2qcx_if_else_numerical.py --bra bra/bin/bra
"""

import argparse
import importlib.util
import pathlib
import subprocess


CONVERTER_PATH = pathlib.Path(__file__).parents[1] / "qcx" / "qasm2qcx.py"
SPEC = importlib.util.spec_from_file_location("qasm2qcx", CONVERTER_PATH)
qasm2qcx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qasm2qcx)


SOURCE = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] outcomes;
int result = 0;

x q[0];
outcomes[0] = measure q[0];
outcomes[1] = measure q[1];

if (outcomes[0] && !outcomes[1]) {
    result += 1;
    if (outcomes[1] == 0) {
        result += 2;
    } else {
        result += 100;
    }
} else {
    result += 1000;
}

if (outcomes[1]) {
    result += 100;
} else {
    result += 4;
}

if (!outcomes[1]) {
    result += 8;
} else {
    result += 100;
}

if ((outcomes[0] || outcomes[1]) && result == 15) {
    x q[1];
}
outcomes[1] = measure q[1];
"""


def check_program(
        bra: pathlib.Path, source: str, outputs: tuple[str, ...],
        expected: list[str]) -> None:
    qcx_lines = qasm2qcx.convert(source)
    qcx_lines.extend(f'PRINTLN {output}' for output in outputs)
    result = subprocess.run(
        [str(bra)],
        input="\n".join(qcx_lines) + "\n",
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    output_lines = [line.strip() for line in result.stdout.splitlines()]
    if output_lines != expected:
        raise RuntimeError(
            "qasm2qcx if/else numerical test failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bra", required=True, type=pathlib.Path)
    arguments = parser.parse_args()
    check_program(arguments.bra, SOURCE,
                  ("RESULT63", "OUTCOMES255:1"), ["15", "1"])

    for a in (0, 1):
        for b in (0, 1):
            for condition, truth in (
                    ("a && b", a and b), ("a || b", a or b),
                    ("!(a && b)", not (a and b)),
                    ("(a || b) && !a", (a or b) and not a),
                    ("a || b && !a", a or (b and not a))):
                source = f'''OPENQASM 3.0; bit a = {a}; bit b = {b};
                    int result = 0;
                    if ({condition}) {{ result = 1; }}
                    else {{ result = 2; }}'''
                check_program(arguments.bra, source, ("RESULT63",),
                              ["1" if truth else "2"])

    # Division by a runtime zero must never execute in a skipped RHS.
    for condition, expected in (
            ("a && 1 / divisor > 0", "2"),
            ("!a || 1 / divisor > 0", "1")):
        source = f'''OPENQASM 3.0; bit a = 0; int divisor = 0;
            int result = 0;
            if ({condition}) {{ result = 1; }} else {{ result = 2; }}'''
        check_program(arguments.bra, source, ("RESULT63",), [expected])

    source = '''OPENQASM 3.0; bit a = 0; int divisor = 0;
        int result = 0;
        if (a && 1 / divisor > 0) { result = 100; }
        result = divisor + 1;'''
    check_program(arguments.bra, source, ("RESULT63",), ["1"])

    source = '''OPENQASM 3.0; bit a = 0; int divisor = 0;
        int result = 0;
        if (a) {
            if (a || 1 / divisor > 0) { result = 100; }
        }
        result = divisor + 1;'''
    check_program(arguments.bra, source, ("RESULT63",), ["1"])

    # Boolean variables use integer storage, but retain Boolean semantics.
    for a in (False, True):
        for b in (False, True):
            for condition, truth in (
                    ("a", a), ("!a", not a),
                    ("a && b", a and b), ("a || b", a or b),
                    ("a == b", a == b), ("true != a", not a),
                    ("a && !bit_value", a and not b)):
                source = f'''OPENQASM 3.0;
                    bool a = {str(a).lower()}; bool b = {str(b).lower()};
                    bit bit_value = {int(b)}; int result = 0;
                    if ({condition}) {{ result = 1; }}
                    else {{ result = 2; }}'''
                check_program(arguments.bra, source, ("RESULT63",),
                              ["1" if truth else "2"])

    source = '''OPENQASM 3.0;
        const bool yes = true; const bool no = false;
        const bool alias = yes; bool a = alias; bool b;
        int result = 0;
        b = a; a = no;
        if (b && !a) { result = 1; a = yes; }
        if (a) { result += 2; }
        if (!no) { result += 4; }
        if (false) { result = 100; }
        if (true) { result += 8; }'''
    check_program(arguments.bra, source, ("A1", "B1", "RESULT63"),
                  ["1", "1", "15"])

    for condition, expected in (
            ("false && 1 / divisor > 0", "2"),
            ("true || 1 / divisor > 0", "1"),
            ("a && 1 / divisor > 0", "2"),
            ("!a || 1 / divisor > 0", "1")):
        source = f'''OPENQASM 3.0; bool a = false; int divisor = 0;
            int result = 0;
            if ({condition}) {{ result = 1; }} else {{ result = 2; }}
            divisor = divisor + 1;'''
        check_program(arguments.bra, source, ("RESULT63", "DIVISOR127"),
                      [expected, "1"])

    # The same truth tables must hold when a condition produces a value.
    for a in (False, True):
        for b in (False, True):
            for expression, truth in (
                    ('!a', not a), ('!!a', a),
                    ('a && b', a and b), ('a || b', a or b),
                    ('!(a && b)', not (a and b)),
                    ('a || b && !a', a or (b and not a)),
                    ('(a || b) == !a', (a or b) == (not a)),
                    ('(a == b) != (a && b)', (a == b) != (a and b)),
                    ('flags[0] && !flags[1]', a and not b)):
                source = f'''OPENQASM 3.0;
                    bool a = {str(a).lower()}; bool b = {str(b).lower()};
                    bit[2] flags = "{int(b)}{int(a)}";
                    bool result = {expression}; bool copy = false;
                    copy = result; result = !result;'''
                check_program(arguments.bra, source, ('COPY15', 'RESULT63'),
                              [str(int(bool(truth))), str(int(not truth))])

    for n in (-1, 0, 1):
        for expression, truth in (
                ('n == 0', n == 0), ('n != 0', n != 0),
                ('n < 0', n < 0), ('n <= 0', n <= 0),
                ('n > 0', n > 0), ('n >= 0', n >= 0),
                ('n + 1 >= 0.5', n + 1 >= 0.5),
                ('0 < n + 1', 0 < n + 1),
                ('(n > 0) == (n < 0)', (n > 0) == (n < 0))):
            source = f'''OPENQASM 3.0; int n = {n};
                bool result = {expression};'''
            check_program(arguments.bra, source, ('RESULT63',),
                          [str(int(truth))])

    source = '''OPENQASM 3.0; bool a = true; bool b = false;
        a = !a; b = !b; a = a || b; b = a && !b;'''
    check_program(arguments.bra, source, ('A1', 'B1'), ['1', '0'])

    for expression, expected in (
            ('a && 1 / divisor > 0', '0'),
            ('!a || 1 / divisor > 0', '1'),
            ('(a && 1 / divisor > 0) == false', '1'),
            ('!(a && 1 / divisor > 0)', '1')):
        source = f'''OPENQASM 3.0; bool a = false; int divisor = 0;
            bool result = {expression}; divisor = divisor + 1;'''
        check_program(arguments.bra, source, ('RESULT63', 'DIVISOR127'),
                      [expected, '1'])

    # The first Boolean-value temporary can be introduced in a skipped body;
    # a later expression must still be able to reuse its storage safely.
    source = '''OPENQASM 3.0; bool a = false; bool result = true;
        int divisor = 0;
        if (a) { result = a && 1 / divisor > 0; }
        result = !a; divisor = divisor + 1;'''
    check_program(arguments.bra, source, ('RESULT63', 'DIVISOR127'), ['1', '1'])

    for expression, expected in (
            ('false && 1 / 0 > 0', '0'), ('true || bool(1 / 0)', '1'),
            ('a && 1 / 0 > 0', '0'), ('!a || bool(1.0 / 0.0)', '1')):
        source = f'''OPENQASM 3.0; bool a = false;
            bool result = {expression}; int n = 1;
            n = n + 1;'''
        check_program(arguments.bra, source, ('RESULT63', 'N1'), [expected, '2'])

    source = '''OPENQASM 3.0; include "stdgates.inc";
        qubit[2] q; bit[2] outcomes;
        x q[0]; outcomes = measure q;
        bool ready = outcomes[0] && !outcomes[1];
        int count = 2; bool enabled = bool(count);
        bool proceed = ready && enabled;
        if (proceed) { x q[1]; } else { reset q[1]; }
        outcomes[1] = measure q[1];
        ready = !ready; outcomes[0] = ready;'''
    check_program(arguments.bra, source,
                  ('READY31', 'PROCEED127', 'OUTCOMES255:0', 'OUTCOMES255:1'),
                  ['0', '1', '0', '1'])

    source = '''OPENQASM 3.0; const bool yes = 2 > 1 && !false;
        const bool no = false && 1 / 0 > 0;
        bool result = yes && !no;
        if (result == (1 < 2)) { result = !result; }'''
    check_program(arguments.bra, source, ('RESULT63',), ['0'])

    for declaration, values in (
            ('int', (-7, 0, 7)), ('uint', (0, 7)),
            ('float', (-0.25, -0.0, 0.0, 0.25))):
        for value in values:
            source = f'''OPENQASM 3.0; {declaration} value = {value};
                bool result = bool(value); int branch = 0;
                if (bool(value)) {{ branch = 1; }}
                result = !bool(value);'''
            check_program(arguments.bra, source, ('RESULT63', 'BRANCH63'),
                          [str(int(value == 0)), str(int(value != 0))])

    for value in (False, True):
        source = f'''OPENQASM 3.0; bool a = {str(value).lower()};
            bit b = a; bool c = b; bit[2] flags = "00";
            flags[0] = c; flags[1] = !c; c = flags[1];
            b = bool(flags[0]); a = bit(c);'''
        check_program(arguments.bra, source,
                      ('A1', 'B1', 'C1', 'FLAGS31:0', 'FLAGS31:1'),
                      [str(int(not value)), str(int(value)), str(int(not value)),
                       str(int(value)), str(int(not value))])

        source = f'''OPENQASM 3.0; bool a = {str(value).lower()};
            int n = int(a); uint u = uint(a); float f = float(a);
            complex z = complex(a); int real_value = int(f);
            int complex_value = int(z); int implicit = a;
            float promoted = a; complex widened = a;
            bool result = bool(int(a));'''
        check_program(arguments.bra, source,
                      ('N1', 'U1', 'REAL_VALUE991', 'COMPLEX_VALUE8159',
                       'IMPLICIT255', 'RESULT63'), [str(int(value))] * 6)

    source = '''OPENQASM 3.0; bool result = bool(pi);
        int n = 0; int divisor = 0;
        result = result && !bool(n + 0.0);
        if (bool(tau)) { result = result || bool(1 / divisor); }
        n = n + 1;'''
    check_program(arguments.bra, source, ('RESULT63', 'N1'), ['1', '1'])

    source = '''OPENQASM 3.0; bool a = false; int divisor = 0;
        bool result = bool(divisor) && bool(1 / divisor);
        if (a) { result = bool(1 / divisor); }
        result = !bool(divisor + 0.0); divisor = divisor + 1;'''
    check_program(arguments.bra, source, ('RESULT63', 'DIVISOR127'), ['1', '1'])


if __name__ == "__main__":
    main()
