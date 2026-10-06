#!/usr/bin/env python3

"""Exercise QCX ASSERT parsing, execution, diagnostics, and skipped checks.

Example:
  python3 bra/test/assert_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import resource
import subprocess


SETUP = """QUBITS 1
VAR VALUES INT 2
VAR INDEX INT
VAR POINTERS INT 2
VAR REALS REAL 2
LET INDEX := 1
LET POINTERS:1 := 0
LET VALUES:0 := 3
LET VALUES:1 := 7
LET REALS:0 := 0.5
LET REALS:1 := -0.5
"""


def run(bra: pathlib.Path, program: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(bra)], input=program, text=True, capture_output=True, timeout=10
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bra", required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    # Failure cases throw uncaught exceptions in the CLI; do not leave core dumps.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    success = SETUP + r"""
assert values:index == 7
ASSERT VALUES:POINTERS:INDEX == 3
ASSERT VALUES:INDEX \= VALUES:0
ASSERT VALUES:1 > VALUES:0
ASSERT VALUES:0 < VALUES:1
ASSERT VALUES:1 >= 7
ASSERT VALUES:1 <= :INT:7
ASSERT REALS:0 == :REAL:0.5
ASSERT REALS:0 \= REALS:1
ASSERT REALS:0 > REALS:INDEX
ASSERT REALS:1 < :PI
ASSERT REALS:0 >= 0.5
ASSERT REALS:1 <= -0.5
JUMP SKIPPED
ASSERT VALUES:1 == 0
ASSERT MISSING == 0
@SKIPPED
JUMPIF DONE INDEX == 1
ASSERT VALUES:1 == 0
@DONE
PRINTLN VALUES:0 VALUES:1 REALS:0 REALS:1
"""
    result = run(bra, success)
    if result.returncode != 0 or result.stdout.strip() != "3 7 0.5 -0.5":
        raise RuntimeError(f"successful/skipped assertions failed: {result}")

    failures = [
        ("VALUES:INDEX == 3", "7 == 3"),
        (r"VALUES:INDEX \= 7", r"7 \= 7"),
        ("VALUES:INDEX > 7", "7 > 7"),
        ("VALUES:INDEX < 7", "7 < 7"),
        ("VALUES:INDEX >= 8", "7 >= 8"),
        ("VALUES:INDEX <= 6", "7 <= 6"),
        ("REALS:0 == 1", "0.5 == 1"),
        (r"REALS:0 \= 0.5", r"0.5 \= 0.5"),
        ("REALS:0 > 0.5", "0.5 > 0.5"),
        ("REALS:0 < 0.5", "0.5 < 0.5"),
        ("REALS:0 >= 1", "0.5 >= 1"),
        ("REALS:0 <= 0", "0.5 <= 0"),
        (r"INDEX \= 0", r"0 \= 0"),
    ]
    for comparison, values in failures:
        setup = SETUP + ("LET INDEX := 0\n" if comparison.startswith("INDEX ") else "")
        result = run(bra, setup + f"ASSERT {comparison}\nPRINTLN VALUES:0\n")
        diagnostic = f"assertion failed in ASSERT {comparison} (evaluated: {values})"
        if result.returncode == 0 or diagnostic not in result.stderr or result.stdout.strip():
            raise RuntimeError(f"failed assertion {comparison!r} did not stop with its diagnostic: {result}")

    for instruction in (
        "ASSERT", "ASSERT INDEX", "ASSERT INDEX ==", "ASSERT INDEX == 1 EXTRA",
        "ASSERT INDEX != 0", "ASSERT INDEX = 1", "ASSERT INDEX ? 1",
    ):
        result = run(bra, SETUP + instruction + "\n")
        if result.returncode == 0 or "assertion failed" in result.stderr:
            raise RuntimeError(f"malformed instruction was not rejected during parsing: {result}")

    for lhs in ("0", ":INT:1", "MISSING"):
        result = run(bra, SETUP + f"ASSERT {lhs} == 0\n")
        if result.returncode == 0 or "wrong argument" not in result.stderr:
            raise RuntimeError(f"invalid assertion lhs was not rejected: {result}")


if __name__ == "__main__":
    main()
