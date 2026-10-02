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
- scalar constants, variables, assignments, and numeric casts;
- scalar and register `bit` values, including bit-string initialization;
- projective measurement of individual qubits and complete registers;
- OpenQASM barriers as ordering-only operations;
- scalar expressions used as gate parameters; and
- final-state amplitude output through a namespaced pragma.

QCX has no unsigned integer type, so OpenQASM `uint` values are represented by
QCX `INT`, just like OpenQASM `int`. Consequently, unsigned ranges and
wraparound behavior are not preserved. Declared integer and floating-point
widths are also accepted but are not enforced by QCX.
OpenQASM `bit` values are represented by QCX `INT` variables whose elements
are restricted to zero or one by the converter. A measurement is emitted as a
QCX `M` operation followed immediately by assignment from `:OUTCOME`.

The built-in constants `pi`, `tau`, and `euler` are available in scalar
expressions. Runtime expressions preserve `pi` and `tau` as the native QCX
symbols `:PI` and `:TWO_PI`; constant declarations are evaluated during
conversion. Unsupported gates and statements are reported as conversion
errors rather than silently omitted.

OpenQASM rotation angles are converted to the half-turn convention used by
the QCX `EX`, `EY`, `EZ`, `CEX`, `CEY`, and `CEZ` operations.
OpenQASM `gphase(angle)` is emitted as the QCX global-phase instruction
`PHASE angle`.

OpenQASM barriers are accepted for scalar qubits, complete qubit registers,
and qubits selected by a single literal index. A barrier without operands is
also accepted. Because the converter preserves source order and does not
optimize or reorder operations, barriers emit no QCX instruction; their
explicit operands are nevertheless validated.

## Amplitude output

The RIKEN-braket-specific `riken_braket.amplitudes` pragma requests that the
generated QCX program print the final state-vector amplitudes:

```qasm
pragma riken_braket.amplitudes
```

Optional nonnegative decimal basis-state indices restrict the output to
selected amplitudes:

```qasm
pragma riken_braket.amplitudes 0 3 7
```

The pragma may appear anywhere in the OpenQASM program, but it may appear only
once. The converter always emits the corresponding `DO AMPLITUDES` instruction
after all circuit operations. Indices must be unique and within the state
vector defined by the program's qubit declarations.

## Not yet in scope

The current prototype does not reliably support:

- reset, delays, or classical control flow;
- user-defined gates or gate modifiers;
- index ranges, discrete index sets, or dynamically computed qubit indices;
- booleans or multidimensional arrays;
- comparisons, logical operations, or classical functions; or
- arithmetic operators other than `+`, `-`, `*`, and `/`.

The characterization tests in `bra/test/test_qasm2qcx.py` define the working
baseline.
