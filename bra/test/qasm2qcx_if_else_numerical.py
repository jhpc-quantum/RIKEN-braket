#!/usr/bin/env python3

"""Exercise converted OpenQASM measurement-controlled branches.

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


if __name__ == "__main__":
    main()
