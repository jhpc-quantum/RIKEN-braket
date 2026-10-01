# OpenQASM 3 to QCX converter

`qasm2qcx.py` is an experimental command-line converter from an OpenQASM 3
program to the QCX input format understood by `bra`.

```console
python3 bra/qcx/qasm2qcx.py circuit.qasm > circuit.qcx
```

The converter requires Python 3.10 or newer and the `openqasm3` package with
its parser support.

```console
python3 -m pip install -r bra/qcx/requirements.txt
```

Python callers can use `qasm2qcx.convert(source)` to obtain the generated QCX
lines as a list of strings.

## Current integration scope

The converter currently covers:

- OpenQASM 3 qubit declarations;
- `stdgates.inc` gates that have direct QCX equivalents;
- a single literal index or a complete qubit register as a gate operand;
- equal-sized whole-register gate broadcasting;
- scalar `int`, `uint`, `float`, and `complex` arithmetic;
- scalar constants, variables, assignments, and numeric casts; and
- scalar expressions used as gate parameters.

QCX has no unsigned integer type, so OpenQASM `uint` values are represented by
QCX `INT`, just like OpenQASM `int`. Consequently, unsigned ranges and
wraparound behavior are not preserved. Declared integer and floating-point
widths are also accepted but are not enforced by QCX.

The built-in constants `pi`, `tau`, and `euler` are available in scalar
expressions. Runtime expressions preserve `pi` and `tau` as the native QCX
symbols `:PI` and `:TWO_PI`; constant declarations are evaluated during
conversion. Unsupported gates and statements are reported as conversion
errors rather than silently omitted.

OpenQASM rotation angles are converted to the half-turn convention used by
the QCX `EX`, `EY`, `EZ`, `CEX`, `CEY`, and `CEZ` operations.

## Not yet in scope

The current prototype does not reliably support:

- measurement, reset, barriers, delays, or classical control flow;
- user-defined gates or gate modifiers;
- `gphase`, `cu`, or `id`;
- index ranges, discrete index sets, or dynamically computed qubit indices;
- bit strings, booleans, or multidimensional arrays;
- comparisons, logical operations, or classical functions; or
- arithmetic operators other than `+`, `-`, `*`, and `/`.

The characterization tests in `bra/test/test_qasm2qcx.py` define the working
baseline. Measurement remains marked as an expected failure for a later
integration stage.
