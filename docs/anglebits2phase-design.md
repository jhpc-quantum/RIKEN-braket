# ANGLEBITS2PHASE design contract

The evaluator converts fixed-point angle bits into a REAL phase after
an ordered INV/POW chain. Exact turn arithmetic preserves branch decisions
through nested powers. This is the design contract for
[issue #396](https://github.com/jhpc-quantum/RIKEN-braket/issues/396).

Status: Stage 2 implements the native bra function and its APPLY interface.
The qasm2qcx integration described below remains planned.

## Interface

```text
APPLY result <- ANGLEBITS2PHASE bits width INV POW 0.5
```

The native signature has one REAL destination, a UINT bit pattern, an INT
width, and an optional sequence of literal modifier descriptors. Ordinary
APPLY scalar operand rules apply to `bits` and `width`, including indexed
elements and explicitly typed operands. Result types do not select overloads.
No public ANGLE, rational, string, or modifier-chain storage type is added.

The descriptor grammar is:

```text
chain     := (INV | POW exponent)*
integer   := [+-]?[0-9]+
floating  := [+-]?([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?
           | [+-]?[0-9]+[eE][+-]?[0-9]+
exponent  := integer | floating
```

`INV` has no operand. `POW` consumes one literal, not a variable, cast,
built-in constant symbol, expression, or array reference. An empty chain
returns the initial principal phase. Keywords follow normal QCX case handling.
Quantum controls are not descriptors: qasm2qcx handles their operands and
polarities separately.

Descriptors are in **evaluation order, left to right**, not OpenQASM source
modifier order. For example, `pow(0.5) @ inv @ p(a)` becomes `INV POW 0.5`;
`inv @ pow(0.5) @ p(a)` becomes `POW 0.5 INV`. The complete chain is evaluated
in one invocation; separate calls returning REAL after each modifier are not
equivalent because they lose exact intermediate branch information.

## Exponent encoding and precision

Integer literals are exact signed integers, independently of native INT/UINT
range, subject to resource limits. Floating literals are decoded once as IEEE
binary64, round-to-nearest with ties to even, then interpreted as the exact
dyadic value of those bits. This decoding must be independent of backend REAL
precision, rounding direction, and DAZ/FTZ. Thus `9007199254740993` is an exact
integer, while `9007199254740993.0` denotes binary64 `9007199254740992`.
`0.1` means the stored binary64 approximation, not the exact decimal `1/10`.

This policy matches qasm2qcx's current Python constant-exponent precision.
The converter will serialize folded integers as decimal integer tokens and
folded floats using a round-tripping floating spelling, retaining `.0` or an
exponent marker when needed. It will not promote integer factors to REAL in
this angle-specific path. Native float/double/long-double configurations must
therefore agree on exact branch decisions for an identical descriptor chain.
Existing non-angle floating-power behavior remains unchanged.

Both signs of floating zero denote zero. A floating literal that rounds to
binary64 zero is a zero exponent; one that rounds to infinity is invalid.
Infinity and NaN are rejected. Integer-valued floating literals such as `2.0`
still select the floating-power converter path, though their exact numerical
factor in this evaluator equals the corresponding integer.

## Exact turn semantics

On a reached invocation, require `1 <= width <= native UINT bits` and
`0 <= bits < 2^width`. Begin with the exact turn fraction `bits / 2^width`.
Represent all intermediate values as exact dyadic rationals. Do not reconstruct
them from complex `EXP`/`ARG` results, compare rounded radians to PI, use an
epsilon window, or requantize into the source angle width.

Let `principal(t)` be the representative of `t` modulo one turn in
`(-1/2, 1/2]`. For each descriptor:

- `INV`: replace the phase by its exact negative.
- `POW r`: replace the phase by `principal(phase) * r`, exactly.

Canonicalizing modulo one turn after each operation is allowed and bounds
intermediate integer parts; it does not flatten or reorder operations. The
final result also uses the principal representative. In particular, both
representations of the negative-real endpoint select the positive half-turn.
Exact zero is canonical positive zero.

For a source representing exactly half a turn:

| Evaluation-order chain | Final turns |
| --- | --- |
| `POW 0.5` | `1/4` |
| `POW 0.5 INV` | `-1/4` |
| `INV POW 0.5` | `1/4` |
| `POW 2 POW 0.5` | `0` |

This convention extends the existing converter's principal spectral powers;
the positive-cut endpoint is a converter policy, not an extra explicit rule
in the specification's general description of gate powers. See
[OpenQASM angles](https://openqasm.com/language/types.html#angles) and
[power modifiers](https://openqasm.com/language/gates.html#power-modifier).

## Final REAL result and errors

Only after the complete chain, form the product of the exact final turn
fraction and the **stored backend REAL `:TWO_PI`**, then round to backend REAL
with round-to-nearest, ties to even. Do not first round the turn fraction to
REAL: premature underflow could erase a representable radian result. This is
not a guarantee of correctly rounding an irrational mathematical `2*pi`.

There is no intermediate floating-point overflow or underflow: finite factors
and phases are processed exactly and reduced modulo one turn. Consequently,
some finite chains that overflow the old radian-based path can be supported
here. The old path's overflow checks must not change. Final underflow is
allowed, including a signed zero for a nonzero negative result; existing native
gate arithmetic may still flush subnormal operands/results according to its
floating-point environment. Calls must not alter rounding/control modes or
require compiler-option changes.

Initial implementation limits are:

| Resource | Limit |
| --- | --- |
| Descriptors per chain | 1,024 |
| Characters per exponent literal, including sign | 4,096 |
| Bits in each reduced exact numerator or denominator | 65,536 |

Validate all descriptors and exponent values, even after a zero factor. Bound
literal sizes before integer/decimal parsing. Check normalized intermediate
representations after every operation; do not let a later zero factor hide
an earlier resource error. Arithmetic scratch may require up to twice the
exact-bit limit, but must remain bounded. These are explicit safety limits,
not silent precision reductions, and must be shared with the converter.

Malformed descriptors, invalid/nonfinite literals, and static resource limits
are diagnosed when the descriptor is constructed, including in skipped calls.
`bits`/`width` values and data-dependent exact-arithmetic limits are checked
only on reached invocations. Skipped calls do not read operands or evaluate
runtime conversions. Errors leave the REAL destination unchanged and identify
the complete APPLY instruction. No zero-exponent shortcut may suppress input
validation on a reached invocation.

## Integration boundaries and validation

The future converter integration covers `p`, `phase`, `u1`, `cp`, `cphase`,
and `gphase`, with constant or runtime angle inputs and literal descriptor
chains folded at conversion time. Integer-only paths remain unchanged.
Runtime floating exponents and mixed runtime-integer/floating-power chains
remain out of scope. Final REAL phases lower through existing U1/controlled
forms or PHASE, preserving full global and relative phases.

`bra/test/anglebits2phase_reference.py` is the independent exact-turn oracle;
it does not import converter reduction helpers or native decomposition code.
`bra/test/test_anglebits2phase_reference.py` checks encoding, endpoint neighbors,
modifier order, extended precision, all native input widths, and resource/error
policies. Stage 2 must independently compare the native kernel against this
oracle across REAL precisions, rounding modes, optimization modes, and UBSan.
