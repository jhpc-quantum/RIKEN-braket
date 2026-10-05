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

if (outcomes[0] == 1) {
    result += 1;
    if (outcomes[1] == 0) {
        result += 2;
    } else {
        result += 100;
    }
} else {
    result += 1000;
}

if (outcomes[1] == 1) {
    result += 100;
} else {
    result += 4;
}

if (result == 7) {
    x q[1];
}
outcomes[1] = measure q[1];
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bra", required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    qcx_lines = qasm2qcx.convert(SOURCE)
    qcx_lines.extend(("PRINTLN RESULT63", "PRINTLN OUTCOMES255:1"))
    result = subprocess.run(
        [str(arguments.bra)],
        input="\n".join(qcx_lines) + "\n",
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output_lines = [line.strip() for line in result.stdout.splitlines()]
    if output_lines != ["7", "1"]:
        raise RuntimeError(
            "qasm2qcx if/else numerical test failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


if __name__ == "__main__":
    main()
