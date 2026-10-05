#!/usr/bin/env python3

"""Exercise scalar and indexed QCX JUMPIF comparisons.

Example:
  python3 bra/test/jumpif_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import subprocess


PROGRAM = r"""QUBITS 1
VAR COUNT INT
LET COUNT := 0
VAR VALUES INT 2
LET VALUES:0 := 5
LET VALUES:1 := 2
VAR REALS REAL 2
LET REALS:0 := 1.5
LET REALS:1 := -0.5

JUMPIF EQ_TRUE VALUES:0 == 5
JUMP EQ_TRUE_END
@EQ_TRUE
LET COUNT += 1
@EQ_TRUE_END
JUMPIF EQ_FALSE VALUES:0 == 4
JUMP EQ_FALSE_END
@EQ_FALSE
LET COUNT += 100
@EQ_FALSE_END

JUMPIF NE_TRUE VALUES:1 \= 5
JUMP NE_TRUE_END
@NE_TRUE
LET COUNT += 1
@NE_TRUE_END
JUMPIF NE_FALSE VALUES:1 \= 2
JUMP NE_FALSE_END
@NE_FALSE
LET COUNT += 100
@NE_FALSE_END

JUMPIF GT_TRUE VALUES:0 > 4
JUMP GT_TRUE_END
@GT_TRUE
LET COUNT += 1
@GT_TRUE_END
JUMPIF GT_FALSE VALUES:0 > 5
JUMP GT_FALSE_END
@GT_FALSE
LET COUNT += 100
@GT_FALSE_END

JUMPIF LT_TRUE REALS:0 < 2.0
JUMP LT_TRUE_END
@LT_TRUE
LET COUNT += 1
@LT_TRUE_END
JUMPIF LT_FALSE REALS:0 < 1.5
JUMP LT_FALSE_END
@LT_FALSE
LET COUNT += 100
@LT_FALSE_END

JUMPIF GE_TRUE VALUES:1 >= 2
JUMP GE_TRUE_END
@GE_TRUE
LET COUNT += 1
@GE_TRUE_END
JUMPIF GE_FALSE VALUES:1 >= 3
JUMP GE_FALSE_END
@GE_FALSE
LET COUNT += 100
@GE_FALSE_END

JUMPIF LE_TRUE REALS:1 <= -0.5
JUMP LE_TRUE_END
@LE_TRUE
LET COUNT += 1
@LE_TRUE_END
JUMPIF LE_FALSE REALS:1 <= -1.0
JUMP LE_FALSE_END
@LE_FALSE
LET COUNT += 100
@LE_FALSE_END

PRINTLN COUNT
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bra", required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    result = subprocess.run(
        [str(arguments.bra)],
        input=PROGRAM,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output_lines = [line.strip() for line in result.stdout.splitlines()]
    if output_lines != ["6"]:
        raise RuntimeError(
            "JUMPIF regression test failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


if __name__ == "__main__":
    main()
