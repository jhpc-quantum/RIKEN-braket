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
- literal indices, static index ranges, discrete index sets, and complete
  qubit registers as gate operands;
- equal-sized register-selection gate broadcasting;
- scalar `int`, `uint`, `float`, and `complex` arithmetic;
- scalar constants, variables, assignments, and numeric casts;
- scalar and register `bit` values, including bit-string initialization;
- projective measurement of individual qubits and register selections;
- reset of individual qubits and register selections;
- OpenQASM barriers as ordering-only operations;
- comparison-based `if`, `else if`, and `else` control flow;
- scalar expressions used as gate parameters; and
- final-state amplitude output through a namespaced pragma.

QCX has no unsigned integer type, so OpenQASM `uint` values are represented by
QCX `INT`, just like OpenQASM `int`. Consequently, unsigned ranges and
wraparound behavior are not preserved. Declared integer and floating-point
widths are also accepted but are not enforced by QCX.
OpenQASM `bit` values are represented by QCX `INT` variables whose elements
are restricted to zero or one by the converter. A measurement is emitted as a
QCX `M` operation followed immediately by assignment from `:OUTCOME`.

OpenQASM `reset` accepts scalar qubits and qubit-register selections. A
register reset is expanded to one QCX `RESET` instruction per selected qubit.
Reset is nonunitary and therefore cannot be used inside a QCX gate-fusion
block.

The built-in constants `pi`, `tau`, and `euler` are available in scalar
expressions. Runtime expressions preserve `pi` and `tau` as the native QCX
symbols `:PI` and `:TWO_PI`; constant declarations are evaluated during
conversion. Unsupported gates and statements are reported as conversion
errors rather than silently omitted.

OpenQASM rotation angles are converted to the half-turn convention used by
the QCX `EX`, `EY`, `EZ`, `CEX`, `CEY`, and `CEZ` operations.
OpenQASM `gphase(angle)` is emitted as the QCX global-phase instruction
`PHASE angle`.

OpenQASM barriers accept scalar qubits and qubit-register selections. A
barrier without operands is also accepted. Because the converter preserves
source order and does not optimize or reorder operations, barriers emit no QCX
instruction; their explicit operands are nevertheless validated.

## Static indexing

Qubit and bit registers can be selected with literal indices, inclusive
ranges, and discrete index sets:

```qasm
x q[2];
x q[1:3];
x q[0:2:6];
x q[6:-2:0];
x q[{0, 3, 5}];
```

The three-part range syntax is `start:step:end`; the step defaults to one when
omitted. Endpoints are inclusive, and negative indices count from the end of
the register. Selection order and repeated discrete indices are preserved.

Ranges and discrete sets are register operands even when they select only one
element. Consequently, all register operands in a broadcast gate must select
the same number of qubits. Scalar indexed operands can broadcast across those
register selections. Measurement requires both operands to be scalars or both
to be equal-sized register selections.

Single indices, range bounds, range steps, and discrete indices must be signed
integer literals. Both range bounds must be present. A zero step, an empty
range, or an out-of-bounds index is rejected during conversion.

## Classical control flow

The converter supports `if`/`else` statements with an explicit comparison as
their condition. The supported comparison operators are `==`, `!=`, `>`, `<`,
`>=`, and `<=`. OpenQASM `!=` is emitted as the QCX not-equal operator `\=`.

Conditions can compare scalar `int`, `uint`, `float`, and `bit` expressions.
A statically indexed element of a bit register is also accepted. Compatible
integer and floating-point operands are promoted when necessary. Complex
values and complete multi-element bit registers cannot be compared.

For example, a measurement result can control later operations:

```qasm
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit outcome = measure q[0];

if (outcome == 0) {
    x q[1];
} else {
    reset q[1];
}
```

The converter lowers structured branches to generated QCX labels, `JUMP`, and
`JUMPIF` instructions. Nested `if` statements and `else if` chains receive
distinct generated labels. Branch bodies may contain supported gates,
measurements, resets, assignments, barriers, and nested branches.

The condition must currently be a single explicit comparison. Conditions such
as `if (flag)`, logical combinations using `&&` or `||`, logical negation, and
comparisons of complex values or complete multi-element bit registers are not
yet supported. Variables used by a branch must be declared outside it;
block-local declarations and lexical scopes are also not yet supported.

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

- delays, loops, or `switch` statements;
- user-defined gates or gate modifiers;
- dynamically computed indices, ranges with omitted bounds, or
  multidimensional indexing;
- general classical arrays, booleans, or block-local declarations;
- non-comparison branching conditions, logical operations, or classical
  functions; or
- arithmetic operators other than `+`, `-`, `*`, and `/`.

The characterization tests in `bra/test/test_qasm2qcx.py` define the working
baseline. `bra/test/qasm2qcx_if_else_numerical.py` additionally converts and
executes a deterministic measurement-controlled program with `bra`.
