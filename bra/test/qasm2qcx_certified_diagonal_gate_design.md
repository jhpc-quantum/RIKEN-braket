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

## Proposed accuracy interface

For the initial path, require an explicit **absolute operator-error goal**;
do not require a target format to make the converter-only guarantee. The
mathematical operation represented by the emitted decimal QCX parameters is
independent of a particular `bra` build. A target descriptor cannot by itself
certify that a particular executable parses or computes those parameters with
the described rounding behavior.

One possible CLI spelling is:

```text
qasm2qcx.py --diagonal-power-error 1e-12 program.qasm
```

The corresponding Python API could add the keyword-only argument
`diagonal_power_error`, supplied as an exact decimal string. These names are a
proposal for review, not existing options. Parse the value directly as an exact
rational; require it to be finite and strictly between zero and one. Do not
round the requested bound through Python `float`.

The option is required only when the new certified diagonal path is
selected; existing converter calls and previously supported gate paths must not
require it or silently change semantics. There is no default certificate goal
for the new path. If a selected new-path call lacks a goal, report a conversion
error with an actionable diagnostic.

Optional target-format metadata can be added later for a separately justified
parsing or backend analysis. It must identify the actual target environment
and must not be presented as certified merely because the user selected a
format. In the current diagonal prototype, `AccuracyRequest.target` is not
used in the converter-only bound: the same exact-real angles and bound result
for binary32 and binary64. Integration should make this separation explicit
rather than implying otherwise.

## Converter-only contract

For each selected call, let `G` be its exact composed diagonal matrix and let
`r` be the exact rational representation of its already-evaluated finite Python
float exponent. Using the selected principal phases, form the ideal `G^r`.
Treat each serialized decimal angle as an exact real number and the QCX gate
operations as exact mathematical unitaries. The certificate must bound the
operator 2-norm of the difference between that serialized operation and
`G^r`, **including global phase**, by the requested goal. The same bound holds
for added controls when the inactive block is exactly identity.

Use certified rational phase enclosures paired with computational basis values
zero and one. Bound multiplication by `r` and exact decimal serialization,
and accept only when the resulting conservative bound is within the goal.
For an exact scalar matrix, both basis phases are still retained. For zero
powers, perform static eligibility and call validation before certifying the
identity. If a phase, label, resource bound, or output accuracy cannot be
certified, report an error; never guess a branch or silently drop the phase.

This guarantee does **not** include parsing the decimal strings into a
particular native type, `bra` transcendental functions, or simulation error.
In particular, it is not an end-to-end guarantee for `float`, `double`, or
`long double`. Those errors require a distinct backend contract and verified
build/runtime assumptions. QCX output must not be described as target-precision
certified merely because its decimal strings have many digits.

## Proposed resource limits and failure behavior

The following are initial **candidates for validation**, not established safe
production defaults. All counters should be shared across the complete
source call, including nested expansion and broadcast output:

| Resource | Candidate limit |
| --- | ---: |
| Gate-definition expansion steps / nesting | Existing 100,000 / 128 limits |
| Exact input or intermediate integer bit length | 4,096 bits |
| Rational numerator or denominator bit length | 8,192 bits |
| Interval refinement precision | 1,024 bits |
| Arctangent series terms per evaluation | 512 |
| Total arithmetic/refinement operations per call | 100,000 |
| Decimal places per emitted angle | 200 |
| Gate-definition / enclosing loop QCX lines | Existing 100,000 / 1,000,000 limits |

Check operand growth **before** allocating large shifts, products, powers,
decimal strings, and intermediate rationals. The limits must cover both input
and intermediate values; checking only the final result is insufficient. The
existing prototype does not yet enforce this complete budget. Stage 3 should
measure representative cases, test each failure path, and determine whether
the candidates are practical or need adjustment.

Conversion is atomic at the public CLI boundary: no QCX lines should be
printed if validation or certification fails. Diagnostics should distinguish
an invalid option, unsupported gate or modifier, exhausted work/precision/size
budget, and a bound that remains above the requested goal. Internal temporary
and binding state must be restored on failures. Static checks must also apply
inside unexecuted classical branches and loops without evaluating runtime
expressions from those paths.
