# Design study: certified powers of diagonal user gates

This is a proposed initial subset for #640, not implemented converter
behavior. The accuracy interface, resource limits, and backend guarantees are
to be decided in later stages. Existing supported converter paths must remain
available when the new path is introduced.

## Eligible definitions

The proposed new path starts with a user-defined gate having exactly one formal
qubit and no formal angle parameters. Its body may be empty or contain a
finite sequence of unmodified calls to the following fixed one-qubit gates on
that formal qubit:

`id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, and `sx`.

The body may also call previously defined, unmodified, parameterless one-qubit
user gates satisfying these same rules. Nested calls are expanded in execution
order under bounded work and nesting limits. Existing definition rules,
including availability of `stdgates.inc`, apply. Forward references, recursion,
body modifiers, parameterized gates, `gphase`, and multi-qubit operations are
outside this new path's initial scope.

Compose the entire expanded body as an exact 2x2 matrix using the fixed-gate
representation from #632. It is eligible only if both off-diagonal entries are
exactly zero. No tolerance or floating-point conversion is used for this test.
The empty sequence produces identity; a diagonal matrix with equal entries is
a scalar gate, whose full phase must be preserved.

For example, all three bodies below are eligible, including the one whose
individual operations are not diagonal:

```qasm
gate z_from_t q { t q; t q; t q; t q; }
gate identity_from_h q { h q; h q; }
gate scalar_i q { z q; y q; x q; }
```

A body containing only `h q;` is not diagonal. It can still be accepted by the
already-supported single-operation wrapper path; eligibility for this new
path must not override or weaken that existing behavior.

## Eligible calls and exponent semantics

Initially, the new path accepts exactly one `pow(r)` whose expression evaluates
to a finite compile-time `float`. Optional `ctrl` and `negctrl` modifiers with
constant positive counts may appear *outside* that power, before it in the
modifier chain. Scalar qubits, supported static selections and broadcasting
follow the converter's existing rules, including distinct operands within each
expanded tuple.

```qasm
OPENQASM 3;
include "stdgates.inc";
gate z_from_t q { t q; t q; t q; t q; }
qubit[2] q;
ctrl @ pow(0.5) @ z_from_t q[0], q[1];
```

The proposed ideal exponent is the converter's existing evaluated Python
`float` value, converted exactly to a rational with `as_integer_ratio()`. This
is not a promise of exact decimal-source semantics or a new float width rule.
`pow(2.0)` follows the floating-power convention; integer-only powers retain
their existing lowering. Positive, zero, and negative finite floating powers
are included. Rejected exponent expressions and unsupported modifier chains
must be diagnosed even when an enclosing classical loop has no iterations.

The first floating power acts on the complete exact matrix. Its eigenphases
use the converter's existing `(-pi, pi]` convention, including `+pi` for
`-1`. Both diagonal phases are retained, so added controls preserve the
relative phase of the active operation. This is a converter policy: the
[OpenQASM power modifier](https://openqasm.com/versions/3.0/language/gates.html#power-modifier)
does not explicitly select that branch.

Initially, `inv`, an additional `pow`, or a control *inside* the floating
power are outside the new path. Modifier order must never be simplified by
blindly multiplying exponents. Runtime exponents, dynamic qubit indexing, and
the other limitations of the existing user-gate implementation remain outside
scope. Existing single-operation wrapper and integer-power support must not be
regressed by these restrictions.

## Validation and boundaries

Before emitting any gate operation, validate the source call, exponent,
control counts, operand arity, and exact eligibility, including zero powers.
Unsupported cases on the new path must produce a clear conversion error; they
must not silently fall back to uncertified floating-point composition. An
unexecuted classical path does not evaluate runtime expressions, but static
types, operand forms, and modifier validity still need checking as elsewhere.

The existing prototype bounds serialized decimal angles as exact real values.
It does not certify how a specific `bra` binary parses or executes them. The
target format or accuracy interface, parsing assumptions, achievable error
contract, and handling of exhausted certification resources remain open for
Stage 2. This document does not claim an end-to-end backend error guarantee.
