# OpenQASM 3 to QCX converter

`qasm2qcx.py` is an experimental command-line converter from an OpenQASM 3
program to the QCX input format understood by `bra`.

```console
python3 bra/qcx/qasm2qcx.py circuit.qasm > circuit.qcx
```

## Environment setup

The converter requires Python 3.10 or newer and the `openqasm3` package with
its parser support (`openqasm3[parser]>=1.0,<2`). The test suite has been run
with Python 3.10.12; newer Python versions have not yet been tested as a matrix.
Converting a program does not require building `bra`; executing the generated
QCX does.

The following commands assume a POSIX shell (for example, Bash on Linux) and
that the current directory is the repository root. Create a virtual environment
outside the repository to keep installed packages separate from its source:

```console
python3 --version
python3 -m venv ../qasm2qcx-venv
. ../qasm2qcx-venv/bin/activate
python -m pip install -r bra/qcx/requirements.txt
```

Check that the first command reports Python 3.10 or newer. If `python3` is an
older version, use an installed Python 3.10-or-newer interpreter to create the
environment instead. Choose another environment path if
`../qasm2qcx-venv` already belongs to another project, and use that path in the
activation commands as well. If `venv` or its bundled `pip` is unavailable,
install the corresponding support package for that interpreter using your
system's package manager, then retry. Dependency installation requires access
to a package index, or a configured local package mirror.

After activation, `python` and `python -m pip` refer to the same isolated
environment. Check the parser installation with a minimal OpenQASM program:

```console
python -c 'import openqasm3; openqasm3.parse("OPENQASM 3.0; qubit q;"); print("Parser OK")'
python -m unittest bra/test/test_qasm2qcx.py
```

The first command should print `Parser OK`; the second runs the converter's
unit tests without requiring a `bra` executable. To convert your input file:

```console
python bra/qcx/qasm2qcx.py circuit.qasm > circuit.qcx
```

To execute `circuit.qcx`, build `bra` as described in
[`docs/bra.md`](../../docs/bra.md). Leave the environment with `deactivate`;
reactivate it in a later shell using `. ../qasm2qcx-venv/bin/activate` from the
repository root. Alternatively, invoke `../qasm2qcx-venv/bin/python` directly
without activation. The converter is invoked from the repository; no separate
installation of `qasm2qcx.py` is needed.

Python callers can use `qasm2qcx.convert(source)` to obtain the generated QCX
lines as a list of strings.

## Current integration scope

The converter currently covers:

- OpenQASM 3 qubit declarations;
- `stdgates.inc` gates that have direct QCX equivalents;
- `inv` modifiers on supported gates and `gphase`, including repeated inversions;
- `ctrl` modifiers with constant counts on supported gates and `gphase`, including
  repeated controls and controls added to already-controlled gates;
- `negctrl` modifiers and combinations of `inv`, `ctrl`, and `negctrl`;
- constant and runtime integer `pow` modifiers on supported gates and `gphase`,
  including nested powers and combinations with inverse and control modifiers;
- constant floating-point `pow` modifiers on selected fixed gate families,
  `p`/`phase`/`u1`, `cp`/`cphase`, and `gphase` with scalar angle or radian inputs, and
  `rx`/`ry`/`rz`/`crx`/`cry`/`crz`, `u2`, `u3`, built-in `U`, and `cu`,
  with principal spectral powers and ordered modifier evaluation;
- literal indices, static index ranges, discrete index sets, and complete
  qubit registers as gate operands;
- equal-sized register-selection gate broadcasting;
- scalar `int`, `uint`, `float`, and `complex` arithmetic;
- native unsigned storage and modulo-width arithmetic for scalar `uint` and
  supported `uint[n]` widths;
- explicit same-width casts between UINT values and bit-register expressions;
- integer remainder expressions and compound assignments using `%` and `%=`;
- scalar integer exponentiation (`**`, `**=`), with nonnegative exponents,
  modular UINT results, and checked native INT results;
- checked native INT unary negation, including minimum-value literal syntax;
- scalar integer and bit bitwise expressions (`&`, `|`, `^`, `~`) and
  compound assignments (`&=`, `|=`, `^=`);
- whole-bit-register bitwise expressions and compound assignments with
  equal-width register operands and bit-string literals;
- whole-bit-register shifts with constant or runtime integer counts,
  including `<<=` and `>>=` assignments;
- circular rotations with `rotl` and `rotr` for bit registers and sized
  `uint[n]`, with constant or runtime signed distances;
- static bit-register range/set selections as bitwise and shift operands
  and overlap-safe assignment destinations, including ordinary `=` assignments
  from runtime-range expressions;
- runtime-dependent discrete-set bit-register expression operands with fixed
  result widths and snapshot reads;
- whole-register `=` assignments and top-level `bit[n]` initializers from
  runtime-range expressions, including nested bitwise/shift/rotation operations;
- `popcount` on runtime ranges, arbitrary complement/shift/rotation chains,
  and nested bitwise expression trees mixing runtime ranges and known-width
  register operands, with shifts, rotations, and complements at any level,
  bounded storage, and a native UINT result;
- runtime-dependent discrete-set bit-register assignment destinations, including
  ordinary `=` assignments from runtime-range expressions and existing bitwise
  and shift compound assignments;
- runtime-dependent bit-register range destinations for ordinary `=` assignments
  from nested runtime-range expressions, with both selected lengths checked at runtime;
- equal-width bit-register `==` and `!=` comparisons, including static/runtime
  selections, bit-string literals, and nested bitwise/shift/rotation expressions;
- scalar integer shift expressions (`<<`, `>>`) and compound assignments
  (`<<=`, `>>=`), with checked counts and signed left-shift overflow;
- scalar constants, variables, assignments, and numeric casts;
- scalar `const bit` declarations and supported constant-bit expressions;
- scalar `angle`/`angle[n]` variables and constants, float-to-angle conversion,
  angle-width conversion, and explicit Boolean conversion;
- scalar angle addition, subtraction, unary negation, UINT scaling, angle ratios,
  and corresponding supported compound assignments;
- scalar angle comparisons with width promotion and unsigned angle ordering;
- scalar angle bitwise expressions (`~`, `&`, `|`, `^`) and `&=`/`|=`/`^=`
  assignments, with width promotion and angle result types;
- scalar angle shifts (`<<`, `>>`, `<<=`, `>>=`), rejecting negative counts
  and producing zero for counts at least the angle width;
- scalar angle `rotl`, `rotr`, and `popcount`, preserving rotation widths and
  returning native UINT population counts;
- scalar angle `sin`, `cos`, and `tan`, returning floating-point values;
- explicit same-width casts between scalar angles and bit-register expressions,
  preserving their exact stored bit patterns;
- static bit indices, ranges, and sets on sized scalar angles and angle-array
  elements, with register-valued reads and overlap-safe plain, bitwise, and shift
  assignments;
- runtime single-bit indices on sized angles and angle-array elements, with
  checked INT/UINT indices, `bit[1]` reads, and plain/compound assignments;
- runtime-valued discrete-set angle bit selections with fixed-width register
  reads, duplicate-checked plain/compound assignments, and scalar-bit iteration;
- scalar angle parameters for `p`, `phase`, `u1`, `cp`, `cphase`, and `gphase`, including
  inverse/control modifiers and constant/runtime integer powers;
- scalar built-in `exp`, `sqrt`, `log`, `sin`, `cos`, `tan`, `arcsin`, `arccos`,
  `arctan`, `ceiling`, `floor`, `real`, and `imag` calls in constant and runtime
  expressions, including nested calls;
- fixed-size, one-dimensional `int`, `uint`, `float`, `bool`, `complex`, and `angle` arrays
  with static and runtime element reads and writes;
- scalar and register `bit` values, including bit-string initialization;
- compile-time `const bit[n]` registers, expressions, and static selections;
- runtime-indexed bit-register element reads, assignments, and scalar
  measurement destinations;
- runtime index-set and range measurement destinations with static qubit-register sources;
- copies from runtime ranges on mutable or constant bit registers into whole
  mutable registers, static selections, or runtime index-set/range destinations;
- bitwise compound assignments into whole registers or static/runtime selections
  with runtime bit-register range RHS values, including nested bitwise,
  complement, shift, and rotation expressions;
- plain and bitwise compound assignments into runtime bit-register ranges with
  known-width register RHS values;
- shift compound assignments into runtime bit-register ranges using their
  actual runtime selection lengths;
- scalar `bool` variables and constants, Boolean literals, and expression
  assignments;
- projective measurement of individual qubits and register selections;
- reset of individual qubits and register selections;
- OpenQASM barriers as ordering-only operations;
- `if`, `else if`, and `else` control flow with comparisons and logical
  conditions;
- compile-time unrolling of constant integer-range and integer-set `for` loops,
  including `break` and `continue`;
- integer-range `for` loops with runtime-valued bounds or steps,
  including nested loops, `break`, and `continue`;
- integer-set `for` loops with runtime-valued elements captured at loop entry;
- runtime iteration over one-dimensional `int`, `uint`, `float`, `bool`, and
  `complex` arrays;
- runtime iteration over mutable bit registers with scalar `bit` iterators;
- unrolled iteration over named constant bit registers with scalar `bit` iterators;
- scalar-bit iteration over static selections and runtime sets/ranges on named
  mutable and constant bit registers;
- iteration over supported sized-angle bit selections with scalar `bit` iterators;
- runtime `while` loops, including nested loops, `break`, and `continue`;
- scalar expressions used as gate parameters; and
- final-state amplitude output through a namespaced pragma.

OpenQASM `int` uses QCX `INT`; scalar `uint` uses native QCX `UINT`.
Narrow `uint[n]` values use the same native storage with explicit masks, as
described below. Declared signed-integer and floating-point widths generally
remain unenforced; shifted signed operands and mixed signed/unsigned operations
have the explicit width restrictions documented below.
Mutable OpenQASM `bit` values are represented by QCX `INT` variables whose elements
are restricted to zero or one by the converter. A measurement is emitted as a
QCX `M` operation followed immediately by assignment from `:OUTCOME`.

### Scalar angles

Scalar `angle[n]` declarations use an unsigned fixed-point fraction of a full
turn. A stored integer `m` represents $2\pi m / 2^n$, with
$0 \le m < 2^n$. Widths must be constant integers in `[1, W]`, where `W` is
the native QCX UINT width (normally 32 bits). Unsized `angle` uses `W` bits.
Declaration widths are resolved in declaration scope; later iterator shadowing
does not change them. A cast width cannot depend on a loop iterator or runtime
variable.

Both `angle[n]` and `const angle[n]` accept a `float` expression or another
angle as an initializer. Mutable scalar angles support `=` assignment and
`+=`, `-=`, `&=`, `|=`, and `^=` with an angle RHS. Scaling assignments `*=` and
`/=` require a UINT RHS with matching effective width.
Shift assignments `<<=` and `>>=` accept INT or UINT counts without promoting
the destination angle's width.
Explicit casts use `angle[n](value)` or `angle(value)`. Matching-width bit-register
casts are also supported as described below. Integer, Boolean, scalar-bit,
and complex inputs are not accepted directly; use `float(...)` first when an
existing numeric conversion is appropriate. Constants require constant inputs,
are read-only, and allocate no runtime storage. Mutable angles use QCX `UINT`
storage, but retain distinct angle typing in the converter: their bits are not
implicitly usable as integers.

| Conversion | Behavior |
| --- | --- |
| `float` to `angle[n]` | Quantize radians to the nearest grid point, ties to even, and wrap modulo one full turn |
| `angle[n]` to `angle[m]`, `m > n` | Left-shift the stored bits by `m - n`, preserving the represented angle |
| `angle[n]` to `angle[m]`, `m < n` | Right-shift by `n - m`, truncating the discarded low bits rather than rounding |
| `bool(value)` for an angle | False exactly when all stored angle bits are zero |
| Explicit `bit[n]` to `angle[n]`, or `angle[n]` to `bit[n]` | Copy the stored pattern exactly, without quantization or rescaling |

Float conversion uses the exact ratio of the stored input and stored full-turn
value, scales it by `2^n`, and rounds once. Negative inputs wrap naturally;
rounding to `2^n` wraps to zero. Both signs of zero become zero. For example,
`angle[4](pi / 2)` stores `4`, `angle[4](-pi / 2)` stores `12`, and
`angle[4](tau)` stores `0`.

Constant evaluation uses Python's binary64 values and `math.tau`. Executable
float-to-angle conversions, including literal inputs, use
`APPLY temporary <- ANGLEBITS :REAL:operand n` at bra's configured REAL precision,
with its stored `:TWO_PI` as the full turn. Consequently, constants and runtime
conversions can differ when their input values or backend precision differ.
The native conversion uses integer significands and remainders, avoiding a
prematurely rounded floating-point quotient even for huge finite inputs.
Nonfinite inputs are errors: evaluated constants fail conversion, while reached
runtime calls fail without updating the destination. Neither build options nor
floating-point control modes are changed.

<!-- scalar-angle-example -->
```qasm
OPENQASM 3.0;
const angle[4] QUARTER = pi / 2;
const angle[4] WRAPPED = -pi / 2;
float radians = pi / 2;
angle[8] value = radians;       // Stored bits: 64.
angle[4] coarse = value;        // Stored bits: 4.
angle copy = coarse;           // Widen without changing the angle.
bool nonzero = bool(coarse);
if (nonzero) {
  value = angle[8](WRAPPED);    // Stored bits: 192.
}
angle[4] same = QUARTER;
```

Angles can be copied and converted inside supported branches and loops; float
array elements and float iterators can supply conversion inputs. Loop iterators
may shadow angle names, and the outer angle type and width are restored afterward.
Skipped branches, short-circuited operands, and empty loops do not execute
conversion domain checks or skipped arithmetic. Empty loops still validate
names, widths, types, and read-only targets.

This is not full angle support. Mixed angle/non-angle comparisons and bitwise
operations, runtime-dependent bit selections outside the supported subsets, compound assignments other than
`+=`/`-=`/`*=`/`/=`/`&=`/`|=`/`^=`/`<<=`/`>>=`, and numeric casts from
angles remain unsupported.
Use an angle comparison or `bool(a)` explicitly in a condition; a bare angle
is not a Boolean condition.
Angle-valued gate parameters are limited to the phase gates described below.

#### Casts between angles and bit registers

`angle[n](bits)` and `bit[n](a)` copy the stored pattern between a bit-register
expression and a scalar angle. The effective widths must be equal. Unsized
`angle(bits)` uses native UINT width, so its source register must have that width.
Cast widths must be positive compile-time integers; the angle width limit
`[1, W]` applies in both directions. A mismatched width is an error rather than
an implicit resize. To resize an angle before extracting its bits, write an
explicit angle conversion, for example `bit[8](angle[8](a))`.

These casts follow the OpenQASM rules for
[casting from angle](https://openqasm.com/language/types.html#casting-from-angle)
and [casting from bit](https://openqasm.com/language/types.html#casting-from-bit).
Bit index zero is the least significant bit. At width `n`, the high bit has
weight π and bit zero has weight `2*pi/2^n`. For example, `"1000"` becomes
an `angle[4]` representing π, while `"0110"` becomes `3*pi/4`. No floating-point
conversion occurs, and round trips preserve every bit, including at native width.

Both constant and executable expressions are supported. Register sources can
be bit strings, named registers, supported static and runtime selections, or
supported register bitwise, shift, and rotation expressions. Runtime ranges
follow the target-width rules below. Angle sources can use
supported angle expressions. Register results work in register initializers,
assignments, static selection destinations, bitwise operations, equality
comparisons, and register functions. Register results require a separate
[explicit UINT cast](#casts-between-uint-values-and-bit-registers) for integer
arithmetic; direct numeric casts from angles remain unsupported. Indexing a cast
result directly remains unsupported; assign it to a named register first. Scalar `bit` and `bit[1]`
remain distinct; these casts require the register form.

Sources are evaluated once and preserved. Runtime lowering packs or extracts
bits using existing QCX UINT operations and private temporaries. It computes
the full result before writing a destination, including overlapping selections.
Operand domain errors occur only when evaluated, so skipped branches,
short-circuit operands, and empty loops suppress them while still validating
names, types, widths, and selection bounds. Constants require constant sources;
there are no implicit conversions between angles and registers.

<!-- angle-bit-cast-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
const angle[4] HALF_TURN = pi;
const bit[4] HALF_BITS = bit[4](HALF_TURN);  // "1000".
bit[4] bits = "0110";
angle[4] a = angle[4](bits);               // Exactly 3*pi/4.
bit[4] copied = bit[4](a);                 // "0110"; bits is unchanged.
h q;
p(angle[4](copied)) q;
```

#### Angle casts from runtime bit-register ranges

`angle[n](flags[start:stop])` and `angle[n](flags[start:step:stop])`
accept inclusive ranges on named mutable or constant bit registers. The target
width `n` must be a positive compile-time integer within `[1, W]`, and each
range must select exactly `n` entries when evaluated. Unsized `angle(...)`
requires exactly `W` entries. The containing source register may be wider than
`W`; only the selected count must match the target width. This is an exact
copy of selected bits into angle storage, with the first selected bit becoming
bit zero. No numeric conversion or floating-point rounding occurs.

Bounds and steps may independently use supported INT/UINT expressions. Both
endpoints are required; an omitted step is `+1`. Start, step, and stop are
captured once in that order. The inclusive raw progression is generated before
normalizing negative indices against the source width. The stop need not be
selected: `1:2:6` selects `1, 3, 5`. Repeated source positions are preserved,
including negative aliases such as `flags[-1:3]` on a four-bit source, which
selects five entries.

The range can also appear inside nested register `~`, `&`, `|`, `^`, `<<`,
`>>`, `rotl`, and `rotr` expressions. Ranges may occur on either or both sides
of binary register operations, alongside supported fixed-width operands.
Every register operand and intermediate must have the target angle width;
fixed-width mismatches fail during conversion, and runtime counts are checked
when reached. Operations retain register semantics: complement flips exactly
the selected width, shifts discard bits at every intermediate step, and
rotations permute the selected positions. Shift counts must be INT/UINT;
negative counts fail when reached, while counts at least the width produce
zero. Rotation distances follow the existing
[rotation rules](#circular-bit-rotations), including checked UINT-to-INT
conversion before modulo normalization and negative distances reversing direction.

Fixed-width operands include literals, named registers, static and runtime
discrete-set selections, supported explicit `bit[n]` casts, and supported
single-bit/discrete-set selections on sized angles or angle-array elements.
Runtime-dependent ranges on explicitly sized angles are also supported in
ANGLE and UINT casts, including mixed trees with bit-register ranges. UINT casts
also support runtime single-index and discrete-set angle selections directly.
Computed-source indexing, aliases, omitted range endpoints, and scalar-bit
sources remain unsupported.

Register operands are evaluated from left to right. Each range occurrence
captures and checks its own positions and selected count before reading that
operand's bits. A zero step, empty range, invalid generated position, or count
different from the target width fails before those reads. A later operand's
failure may follow earlier operand reads, but all checks and reads finish
before any destination write. Bitwise operators do not short-circuit, even
with a zero mask or repeated source text. Shift counts follow their left
operand, and rotation distances follow their complete register argument.
Position and result storage are bounded by the target width.

The scalar result retains ANGLE semantics, including modular angle arithmetic
and angle-width conversions. It works in scalar and angle-array initializers
and assignments, supported comparisons, explicit Boolean conversion, supported
angle functions, and phase-gate parameters. An enclosing `bit[n]` cast provides
an exact round trip, including overlapping selection assignments; direct
numeric casts from the resulting angle remain unsupported. Runtime bounds or
mutable sources cannot initialize a constant angle. Skipped branches,
short-circuit operands, and empty loops suppress runtime domain checks while
still validating names, types, and widths. Each loop iteration reads fresh
operands and bounds.

<!-- runtime-range-angle-cast-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
bit[8] flags = "11001001";
int start = 7;
int step = -2;
int stop = 1;
angle[4] saved = angle[4](flags[start:step:stop]); // Stored bits: 5, angle: 5*pi/8.
angle[4] from_constant = angle[4](PATTERN[start:step:stop]); // Stored bits: 11.
angle[4] combined = angle[4](rotl(flags[start:step:stop] ^ PATTERN[start:step:stop], 1)); // 13.
bit[4] copied = bit[4](saved);                 // "0101", exact round trip.
array[angle[4], 2] phases = {saved, combined};
int slot = 1;
phases[slot] = angle[4](~flags[start:step:stop]);
angle[4] element = phases[slot];              // Stored bits: 10.
flags[0:3] = bit[4](combined);                 // "11001101".
start = 0;
step = 1;
stop = 3;
angle[4] fresh = angle[4](flags[start:step:stop]); // Stored bits: 13; saved is still 5.
```

#### Angle casts from runtime angle-bit ranges

An ANGLE cast can use a runtime range on an explicitly sized angle variable,
constant, array element, or supported angle iterator. For example,
`angle[4](phase[first:last])` copies the selected pattern exactly. The selected
count must equal the target width; unsized `angle(...)` uses native UINT width.
The source angle itself must have an explicit width.

The range can occur within the same complement, bitwise, shift, and rotation
trees described above, including trees mixing angle and bit-register ranges.
Array-element addresses and the complete source angle are captured before the
range bounds. Each occurrence captures start, step, and stop once, checks all
positions and the selected count, and then extracts bits in selection order.
Negative aliases and repeated source positions are preserved. The complete
expression is evaluated before a destination is written, including overlapping
assignments. Skipped paths defer runtime errors, while names, types, and widths
are still validated. Runtime angle ranges also work in UINT casts and direct
[whole-register copies](#whole-register-copies-from-runtime-angle-bit-ranges).

<!-- runtime-angle-range-cast-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int first = 1;
int last = 4;
angle[4] selected = angle[4](phase[first:last]); // Stored bits: 10.
bit[4] copied = bit[4](selected);               // "1010".
array[angle[8], 2] phases = {phase, phase};
int slot = 1;
phases[slot] = angle[4](~phases[slot][first:last]); // Widened stored bits: 80.
angle[8] changed = phases[slot];
```

#### Static bit selections on sized angles

Named, explicitly sized `angle[n]` variables and constants, sized angle
iterators, and elements of `array[angle[n], N]` support bit-level reads.
Writable scalar variables and array elements also support selection assignment.
The supported selectors are single integer indices, inclusive ranges
`start:stop` or `start:step:stop`, and discrete sets `{i, j, ...}`. Bounds,
steps, and set elements on the static-selector path must be literals or
supported expressions of previously declared constants. Active loop iterators
cannot supply range bounds or steps in fixed-width selection expressions or
assignments, even when unrolled or shadowing a constant. Runtime-dependent
ranges are supported as [scalar-bit loop sources](#runtime-dependent-angle-bit-range-loops).
Single indices and discrete-set entries can also be runtime
expressions, including iterator indices, as described under
[runtime single indices](#runtime-single-bit-indexing-on-sized-angles) and
[runtime-valued discrete sets](#runtime-valued-discrete-set-bit-selections).
A range's omitted step is
`+1`; descending ranges require an explicit negative step.

Bit zero is the least significant stored bit, and bit `n - 1` has weight pi.
Negative indices count from the high end, with `-1` selecting bit `n - 1`.
The first selected bit becomes bit zero of the result, and selection order is
preserved. Repeated source indices are accepted, including repetitions after
negative-index normalization. Destinations must have distinct normalized indices.
Empty selections, omitted range bounds, zero steps, noninteger indices, and
out-of-range bits are rejected during conversion.

Every selection returns `bit[k]`, where `k` is the number of selected bits.
In particular, `a[0]` on an angle returns `bit[1]`, not scalar `bit`; this follows
the [OpenQASM classical-value bit-slicing rules](https://openqasm.com/language/types.html#classical-value-bit-slicing)
and differs from indexing a bit register. Selections can initialize or assign
matching-width registers and participate in supported register bitwise
operations, shifts, equality comparisons, `popcount`, `rotl`, `rotr`, and
same-width angle casts. They also support
[iteration with scalar bit iterators](#angle-bit-selection-for-loops).
They do not implicitly become scalar bits, Boolean
conditions, integers, or angles. Selections of angle constants can initialize
constant registers and participate in supported constant expressions; mutable
sources and angle iterators remain runtime values. To use a selected bit as a
condition, assign it to a named `bit[1]` register and index that register, for
example `if (low[0]) { ... }` with `low` from the example below.

Plain assignment uses `=` with a matching-width bit-register value. Only selected
bits change. The destination array index is evaluated and checked once and
remains captured throughout RHS evaluation. Explicit RHS accesses evaluate
their own indices. The complete result is built in private UINT storage before
one destination write, so overlapping copies use the original source bits.
Constants and active loop iterators remain read-only. Plain and bitwise
assignments to even a single-bit target require a register RHS such as `"1"`;
accepting scalar `bit`, Boolean, or integer values for these RHS operands is
outside this converter subset. Shift assignments use integer counts instead.

Selections also support these compound assignments:

| Operators | RHS | Behavior |
| --- | --- | --- |
| `&=`, `|=`, `^=` | A matching-width bit register | Operate on the selected bits in selection order |
| `<<=`, `>>=` | A supported scalar INT or UINT expression | Shift within the selection width, discarding shifted-out bits and zero-filling vacated positions |

The width is the number of selected bits, not the width of the containing angle.
For example, `a[{3, 0}] <<= count` shifts a two-bit register whose low bit is
`a[3]` and high bit is `a[0]`; only those two angle bits are updated. Single-bit
selections retain `bit[1]` register semantics for all five operators.

Compound assignments snapshot the original selected bits before evaluating the
RHS. Their synthetic LHS read uses this snapshot without re-evaluating the
destination array index; explicit RHS self-references still perform their own
accesses. The full result is computed before the single destination write, so
overlapping operands use the original source bits and unselected bits are
preserved. Failed operand evaluation does not commit a partial update.

Shift counts are evaluated once. A signed count is checked for nonnegativity
on its execution path. Counts at least the selection width produce zero in
the selection, including native UINT_MAX and oversized constant counts, without
an invalid native shift or a premature signed conversion. Skipped operations
do not execute count checks or arithmetic. This follows the existing
[bit-register shift policy](#whole-bit-register-shifts). The specification
describes unsigned counts and discarded bits; accepting nonnegative signed
counts and zeroing oversized shifts are documented converter choices.

The outer array-element index in `phases[index][selector]` retains the existing
static/runtime integer-index rules, including negative indices and runtime
bounds checks. Runtime-dependent ranges are supported only as
[scalar-bit loop sources](#runtime-dependent-angle-bit-range-loops). Names,
types, widths, static bounds, and read-only targets are validated in skipped
bodies and empty loops without evaluating runtime array indices or arithmetic.
Temporary storage is released on success and failure while caller-owned storage
is preserved. Subsequent gate operations see the updated angle; gates queued
before a write retain their captured parameters, including with fusion.

<!-- static-angle-bit-selection-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
angle[4] a = 3 * pi / 4;        // Stored pattern "0110".
bit[1] low = a[0];              // "0", a one-bit register.
bit[2] upper = a[2:3];          // "01".
a[0:2] = a[1:3];               // Overlap-safe: a becomes "0011".
array[angle[4], 2] phases = {0.0, pi};
int index = -1;
phases[index][0:3] = a[3:-1:0]; // phases[1] becomes "1100", or 3*pi/2.
h q;
p(phases[index]) q;
```

<!-- angle-selection-compound-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
angle[4] a = angle[4]("0111");
a[0:2] &= "110";              // "0110".
a[0:1] |= "01";               // "0111".
a[0:2] ^= a[1:3];             // "0100", using the original overlapping bits.
array[angle[4], 1] phases = {a};
int index = -1;
uint count = 1;
phases[index][{2, 3}] <<= count; // "1000"; selection order defines the shift.
phases[index][0:3] >>= count;    // "0100", or pi/2.
h q;
p(phases[index]) q;
```

Bit selections on unsized angles, computed angle expressions or cast results,
further indexing of a selection result, runtime-dependent bit ranges outside
the supported [iteration context](#runtime-dependent-angle-bit-range-loops),
aliases, multidimensional arrays, and INT/UINT bit slicing remain unsupported.
Arithmetic compound assignments to angle selections remain unsupported: a
selection is a bit register, so it does not inherit angle addition or scaling.
Use an explicitly sized temporary before selecting bits from an unsized angle.
No new bra instruction or build option is required.

#### Runtime single-bit indexing on sized angles

The same named, explicitly sized angle sources also support a single runtime
INT or UINT bit index: `a[index]` and `phases[element_index][bit_index]`.
Supported integer expressions include arithmetic, enclosing loop iterators,
integer-array elements, and integer-valued calls such as `popcount`.
Index names follow declaration order and iterator shadowing. Floating-point,
Boolean, and scalar bit indices require a supported explicit integer cast.
Runtime-valued discrete sets are described next. Runtime-dependent ranges are
supported only as [scalar-bit loop sources](#runtime-dependent-angle-bit-range-loops).

The result is always `bit[1]`, including for an angle constant with a runtime
index. It supports the register consumers and equal-width casts described
above, and [single-iteration bit loops](#angle-bit-selection-for-loops), but
cannot initialize a constant register or implicitly become a scalar
bit, Boolean condition, integer, or angle. Constant and iterator angle sources
remain read-only; scalar angle variables and array elements support `=`, `&=`,
`|=`, `^=`, `<<=`, and `>>=` with the same RHS rules as static selections.

For `angle[n]`, valid signed indices are in `[-n, n - 1]`; negative indices
normalize by adding `n`. UINT indices must be less than `n`. Invalid indices
fail on the executed path before bit extraction or modification, including
UINT values too large to convert to QCX INT safely. Index expressions execute
once per access. For an array element, its element index is evaluated and
checked before the bit index. A destination retains both captured indices
throughout RHS evaluation; compound assignments snapshot the selected bit
before evaluating the RHS. Explicit RHS accesses evaluate their own indices.
Only the selected bit is replaced, with one destination write after successful
RHS evaluation. A positive shift count clears the selected bit; a zero count
preserves it, and negative signed counts fail on the executed path.

Skipped branches, short-circuited expressions, and empty loops do not execute
runtime index arithmetic, bounds checks, or shift-count checks. They still
validate names, types, widths, and read-only restrictions without inventing
iterator values. Gates retain parameters captured before a selection write,
including under fusion. This feature uses existing QCX instructions and requires
no bra instruction or build change.

<!-- runtime-angle-bit-selection-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
angle[4] a = angle[4]("0011");
int bit_index = -1;
bit[1] old = a[bit_index];             // "0", a one-bit register.
a[bit_index] ^= "1";                  // a becomes "1011".
array[angle[4], 1] phases = {a};
int element_index = -1;
phases[element_index][bit_index] >>= 1; // "0011": clear just the selected bit.
bit_index = 0;
phases[element_index][bit_index] = "0"; // "0010", or pi/4.
h q;
p(phases[element_index]) q;
```

#### Runtime-valued discrete-set bit selections

The same explicitly sized angle sources support selectors such as
`a[{i, j, -1}]` and `phases[element][{i, j}]`, where entries are supported INT/UINT
expressions. Constant and runtime entries can be mixed. The result width is the
number of entries, including repetitions: the first form returns `bit[3]` and
the second returns `bit[2]`. A singleton set still returns `bit[1]`.
Selected registers can be wider than the source angle when positions repeat.
These fixed-width results follow the existing
[classical-value bit-slicing rules](https://openqasm.com/language/types.html#classical-value-bit-slicing)
and [runtime index-set rules](https://openqasm.com/language/types.html#register-concatenation-and-slicing).

Reads support the existing matching-width register assignments, bitwise
operations, shifts, equality comparisons, `popcount`, `rotl`, `rotr`, and
same-width angle casts. They do not implicitly become scalar bits, Boolean
conditions, or integers, and a runtime-dependent selection cannot initialize a
constant register. Unsized angles and computed angle sources remain unsupported.

For an array element, its element index is captured and checked first. Each set
entry is then evaluated, captured, normalized, and bounds-checked in source
order, once per access. Negative indices count from the high end; UINT indices
must be below the angle width. Explicit RHS accesses evaluate their own indices.
When any entry is runtime- or iterator-dependent, all entries are captured on
the executed path, including literal and constant entries. Bounds and arithmetic
failures in such sets do not execute on skipped paths. Invalid literals too
large for QCX INT produce a reached bounds failure without placing an
unrepresentable number in the QCX program. Entirely constant sets retain their
existing conversion-time validation. Names, types, widths, and read-only targets
are still checked in skipped bodies and empty enclosing loops without inventing
iterator values.

Writable angles and array elements support `=`, `&=`, `|=`, `^=`, `<<=`, and
`>>=` with the same RHS rules as other selections. After capturing and checking
all destination positions, a native UINT bitmap rejects duplicate normalized
positions before evaluating the RHS. For example, if `i == -1` for `angle[4] a`,
`a[{i, 3}] = "11"` fails when reached. Repeated positions remain valid for reads
and iteration. Constants and active angle iterators remain read-only.

Compound assignments snapshot the original selected bits before RHS evaluation.
All RHS bits and calculations complete before one destination write; unselected
bits are preserved, including for overlapping self-copies. Shift assignments act
within the selection width, using the existing nonnegative-count and oversized-
count rules. No destination update is committed if an index, duplicate check,
RHS access, or arithmetic operation fails.

Selection loops capture the source address and all positions at loop entry,
before binding the scalar `bit` iterator. Later changes to selector variables
leave that iteration sequence fixed. Each visit copies the current source bit,
so source writes affect future visits, including repeated positions, without
changing the current iterator. This retains the documented
[live-read policy](#angle-bit-selection-for-loops).

<!-- runtime-angle-bit-set-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
angle[4] a = angle[4]("0011");
int high = -1;
uint low = 0;
bit[3] before = a[{high, low, high}]; // "010": repeated reads are valid.
a[{high, low}] ^= "11";             // a becomes "1010".
array[angle[4], 1] phases = {a};
int element = -1;
phases[element][{high, low}] = phases[element][{low, high}]; // "0011".
uint ones = 0;
int visits = 0;
for bit b in phases[element][{low, high, low}] {
    if (visits == 0) {
        element = 0;
        low = 1;                    // Captured positions remain 0, 3, 0.
        phases[0][0] = "0";
    }
    ones += uint(b);                // b remains a copy of its original bit.
    visits += 1;
}
// Visits 1, 0, 0: ones is 1, visits is 3, phases[0] is "0010".
h q;
p(phases[0]) q;
```

No variable-width register storage, new bra instruction, or build option is
required. Runtime-dependent ranges are supported only as
[scalar-bit loop sources](#runtime-dependent-angle-bit-range-loops);
general variable-width selection results remain separate work.

#### Scalar angle arithmetic

Scalar angles support constant and runtime addition, subtraction, unary
negation, unsigned-integer scaling, and angle ratios, implementing a subset of the
[OpenQASM angle operations](https://openqasm.com/language/classical.html#angles).
Addition and subtraction require two angle operands; there is no implicit
mixing with `int`, `uint`, `float`, `bool`, bit, or complex values. Convert a
float explicitly, for example `a + angle[8](radians)`. Each conversion quantizes
its input before angle arithmetic takes place.

| Operation | Result |
| --- | --- |
| `a + b`, `a - b` | Angle with the wider operand width |
| `-a` | Angle with the same width and sizedness as `a` |
| `a += b`, `a -= b` | Promoted binary result converted back to `a`'s width |

For differing widths, the narrower operand is widened by appending low-order
zero bits, preserving the angle rather than the unsigned integer value.
Binary results are explicitly sized if either operand is explicitly sized;
otherwise they remain unsized. Unsized angles use native UINT width. Assignment
to a narrower angle retains the existing truncation policy, discarding low bits
after the complete expression has been evaluated.

At a result width `n`, aligned bit patterns `left` and `right` produce
`(left + right) mod 2^n` or `(left - right) mod 2^n`. Unary negation produces
`(-bits) mod 2^n`, so negating zero gives zero. Results represent angles in
`[0, 2*pi)`. The principal-phase selection for
[floating gate powers](#floating-powers-of-angle-valued-phase-gates) is a later,
separate operation.

Additive compound assignments behave like `a = a + b` or `a = a - b`, including
promotion before destination narrowing. For example, if `a` is `angle[4]` with bits `0001`
and `b` is `angle[8]` with bits `00000001`, `a -= b` computes
`00010000 - 00000001 = 00001111` at eight bits, then stores `0000` in `a`.
Truncating `b` before subtraction would incorrectly leave `a` at `0001`.
Constants remain read-only, and self-referencing RHS expressions are supported.

Multiplication and division use these operand and result types:

| Operation | Result |
| --- | --- |
| `a * u`, `u * a`, `a / u` | Angle with `a`'s width and sizedness; `u` must be UINT of the same effective width |
| `a / b` | UINT with the wider angle operand width; explicitly sized if either angle is explicitly sized |
| `a *= u`, `a /= u` | Same result as `a = a * u` or `a = a / u` |

Here `a` and `b` are angles and `u` is an unsigned integer. Unsized types use
native UINT width, so a sized and an unsized operand can match at that width.
Otherwise, use an explicit cast such as `uint[4](factor)` to select a matching
factor width. There is no implicit scaling by signed integers, floating-point
values, or Booleans. `angle * angle` and `uint / angle` are unsupported.

At width `n`, an angle with stored bits `bits` multiplied by the ordinary
integer `factor` produces `(bits * factor) mod 2^n`; division produces
`bits // factor`. The UINT factor is not a fractional turn and its bits are
never shifted to align with an angle. Division truncates toward zero because
both stored patterns are nonnegative. Multiplication wraps modulo one full turn.

For `a / b`, the narrower angle is first widened by appending low zero bits,
as for addition. The aligned unsigned patterns then produce `left // right`.
For example, `angle[4]` bits `0111` divided by `angle[8]` bits `00000001`
produces `uint[8]` value `112`. The quotient is an ordinary UINT and can feed
supported integer arithmetic, powers, shifts, bitwise operations, casts,
`popcount`, or rotations. `a /= b` with an angle RHS remains unsupported:
the binary result is UINT and implicit UINT-to-angle conversion is disallowed.

A zero divisor is an error when evaluated. Constant expressions fail during
conversion. Ordinary executable expressions, including divisions of constants,
use bra's UINT division check when reached, so skipped branches, short-circuit
operands, and empty loops do not raise zero-divisor errors. An outer zero
factor does not suppress an evaluated inner division. Compound assignments
write the destination only after the complete RHS succeeds.

Once operands are quantized, constant arithmetic uses exact integer operations.
Runtime arithmetic captures each operand once, in left-to-right order, into
private UINT storage, then uses existing QCX arithmetic, shifts, and masks.
The operands are unchanged; assignment updates its destination only after the
RHS completes. No new bra instruction, storage type, build option, or
floating-point calculation is needed for the angle arithmetic itself.

Expressions work in initializers, assignments, explicit Boolean conversions,
and supported phase-gate parameters, including modifiers, broadcasting, and
fusion. Branches and loops retain normal reached/skipped execution behavior.
Zero results do not suppress runtime conversion or access checks. Empty loops
still validate types, names, widths, operators, and read-only destinations;
temporary declarations are hoisted out of repeated runtime bodies.

<!-- angle-arithmetic-example -->
```qasm
OPENQASM 3.0;
const angle[4] ONE = pi / 8;
const angle[8] FINE = pi / 128;
const angle[4] WRAPPED = ONE + angle[4](15 * pi / 8);
angle[4] wrapped = WRAPPED;     // Bits: 0000.
angle[4] a = ONE;
angle[8] b = FINE;
angle[4] difference = a - b;    // Bits: 0000; subtraction before narrowing.
a -= b;                       // Bits: 0000.
b += angle[8](ONE);            // Bits: 00010001.
angle[4] negative = -ONE;       // Bits: 1111.
bool nonzero = bool(negative);  // true.
```

Computed angles can also feed quantum gates directly:

<!-- angle-arithmetic-phase-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
const angle[4] HALF_TURN = pi;
const angle[8] EPSILON = pi / 128;
angle[4] a = HALF_TURN;
angle[8] fine = EPSILON;
a -= fine;                    // Bits: 0111, representing 7*pi/8.
h q;
p(a) q[0];
p(-a) q[1];
negctrl @ pow(0.5) @ gphase(a - HALF_TURN) q[2];
pragma riken_braket.amplitudes
```

The last operation applies a phase of `-pi/16` when `q[2]` is zero: the
four-bit subtraction represents `15*pi/8`, whose principal square-root phase
is `-pi/16`. The other two operations apply `+7*pi/8` and `-7*pi/8` when
`q[0]` and `q[1]`, respectively, are one, retaining their full phases.

<!-- angle-multiplicative-example -->
```qasm
OPENQASM 3.0;
const angle[4] START = 7 * pi / 8;
const angle[4] UNIT = pi / 8;
const angle[8] FINE = pi / 128;
const uint[4] TWO = 2;
const angle[4] DOUBLE = START * TWO;  // Bits: 1110.
const uint[8] RATIO = START / FINE;   // Ordinary integer: 112.
angle[4] a = START;
uint[4] factor = 2;
angle[4] product = factor * a;        // Bits: 1110.
angle[4] divided = a / factor;        // Bits: 0011.
uint[4] quotient = a / UNIT;          // Ordinary integer: 7.
a *= uint[4](3);                     // Bits: 0101 after wrapping.
a /= factor;                        // Bits: 0010 after truncation.
```

<!-- angle-multiplicative-phase-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
angle[4] a = 5 * pi / 8;
uint[4] factor = 3;
a *= factor;
a /= uint[4](2);                     // Bits: 0111, representing 7*pi/8.
h q;
p(a) q[0];
cp(a) q[1], q[2];
negctrl @ pow(0.5) @ gphase(a) q[2];  // Phase +7*pi/16 when q[2] is zero.
pragma riken_braket.amplitudes
```

#### Scalar angle comparisons

Scalar angle expressions support `==`, `!=`, `<`, `<=`, `>`, and `>=`, producing
Boolean results during constant evaluation or runtime execution. Comparisons use
the represented values in `[0, 2*pi)`, following the
[OpenQASM comparison rules](https://openqasm.com/language/classical.html#comparison-boolean-instructions).
Ordering is unsigned and does not use the principal-phase representative of a
floating gate power. For example, `-pi/2` stored as an angle represents
`3*pi/2`, so it compares greater than an angle representing `pi`.
Wraparound means doubling an angle need not increase it.

For different widths, the narrower angle is widened by appending low zero bits
before comparison. This preserves the angle and discards no precision:
`angle[4]` bits `0100` and `angle[8]` bits `01000000` compare equal. Sized and
unsized operands follow the same rule using their effective widths; unsized
angles use native UINT width. See the
[promotion and casting rules](https://openqasm.com/language/types.html#casting-specifics).

Both operands must be angles in this supported subset. Convert numeric values
explicitly, for example `a == angle[4](pi)`, rather than `a == pi`. No implicit
angle comparison with `int`, `uint`, `float`, `bool`, bit, or complex operands is
provided. An angle ratio such as `a / b` produces UINT, so comparisons of that
result use the ordinary integer rules instead.

Boolean results work in initializers, assignments, Boolean arrays, `if`/`else`,
`while`, and logical expressions, including nested comparisons and
short-circuit evaluation. Constant evaluation compares exact aligned bit
patterns. Runtime lowering captures the left operand once before the right,
aligns private UINT snapshots, and uses existing QCX `JUMPIF` comparisons.
Sources remain unchanged; temporary declarations are hoisted from repeated
runtime bodies. No new bra instruction or floating-point comparison is needed.

A reached comparison evaluates both operands, including conversion, access,
and zero-divisor checks. A skipped branch, short-circuited logical operand, or
empty loop does not execute those checks, but names, types, and widths are still
validated. As with other constant expressions, an evaluated invalid constant
operand is rejected during conversion.

<!-- angle-comparison-example -->
```qasm
OPENQASM 3.0;
const angle[4] QUARTER = pi / 2;
const angle[8] SAME = pi / 2;
const bool EQUAL = QUARTER == SAME;   // true after widening QUARTER.
angle[4] a = 3 * pi / 2;             // Bits: 1100.
angle[8] b = pi;                     // Bits: 10000000.
bool greater = a > b;                // true: 3*pi/2 > pi.
bool wrapped = (a * uint[4](2)) < a;  // true: doubling wraps to pi.
bool fine_order = angle[4](pi / 8) < angle[8](pi / 128); // false.
int count = 0;
while (a > b) {
  a -= angle[4](pi / 8);
  count += 1;
}                                   // a now represents pi; count is 4.
bool equal = a == b;                 // true.
bool skipped = false && (a / uint[4](0) == b); // false; no zero division.
```

<!-- angle-comparison-phase-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
angle[4] a = 5 * pi / 8;
angle[8] b = pi;
h q;
if (a < b) {
  p(a) q[0];
} else {
  p(-a) q[0];
}
if (a != b && a > angle[4](0.0)) {
  negctrl @ pow(0.5) @ gphase(a) q[2]; // Phase +5*pi/16 when q[2] is zero.
}
pragma riken_braket.amplitudes
```

#### Scalar angle bitwise operators

Scalar angles support `~`, `&`, `|`, and `^`, plus `&=`, `|=`, and `^=`, during
constant evaluation and runtime execution. These operate on the stored angle
bits, following the [OpenQASM bitwise rules](https://openqasm.com/language/classical.html#classical-bits-and-registers).
The result is an angle, preserving its use in angle arithmetic, comparisons,
explicit Boolean conversions, and supported phase-gate parameters.

| Operation | Result width and sizedness |
| --- | --- |
| `~a` | Same width and sizedness as `a`; complement only those bits |
| `a & b`, `a \| b`, `a ^ b` | Wider operand width; sized if either operand is sized |
| `a &= b`, `a \|= b`, `a ^= b` | Promoted binary result converted back to `a`'s width |

Binary operands must both be angles. Different widths are aligned by appending
low zero bits to the narrower angle before combining the patterns, using the
same promotion as angle arithmetic and comparisons. Unsized angles use native
UINT width. No implicit bitwise mixing with integer, floating-point, Boolean,
bit, or complex values is provided.

At width `n`, complement is `bits XOR (2^n - 1)`. This is distinct from unary
minus: for four-bit `0111`, complement gives `1000`, whereas negation gives
`1001`. Complement happens at its operand's width before any enclosing promotion.
For example, widening `~angle[4]` bits `0111` to eight bits produces `10000000`;
widening the original angle first and then complementing produces `10001111`.

Compound assignments capture their operands before updating the target,
including self-referencing expressions. Compute at the promoted width, then
discard low bits if the destination is narrower. Constants remain read-only.
Runtime lowering uses private UINT snapshots, existing QCX bitwise instructions,
and width masks, preserving source values and angle metadata. Operands are
evaluated once, left before right for binary expressions. No new bra instruction
or floating-point operation is required for the bitwise calculation itself.

An all-zero operand does not suppress evaluation of the other operand. Reached
conversion, access, and division errors still fail; skipped branches,
short-circuited expressions, and empty loops do not execute those checks.
Names, types, widths, operators, and read-only targets are validated even on
skipped paths. [Angle rotations and population counts](#scalar-angle-rotations-and-population-counts)
are described below.

<!-- angle-bitwise-example -->
```qasm
OPENQASM 3.0;
const angle[4] A = 7 * pi / 8;        // Bits: 0111.
const angle[8] B = pi / 128;          // Bits: 00000001.
const angle[4] INVERTED = ~A;         // Bits: 1000.
const angle[8] JOINED = A | B;        // Bits: 01110001.
const angle[8] BEFORE = ~A | B;       // Bits: 10000001.
const angle[8] AFTER = ~angle[8](A);  // Bits: 10001111.
angle[4] a = A;
angle[8] b = B;
angle[8] result = (~a | b) ^ b;       // Bits: 10000000.
angle[4] narrow = a | b;             // Bits: 0111 after narrowing.
a |= b;                             // Bits: 0111.
a ^= angle[4](pi / 4);               // Bits: 0101.
a &= angle[8](15 * pi / 8);          // Bits: 0101.
bool nonzero = bool(a & b);          // false.
```

<!-- angle-bitwise-phase-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
angle[4] a = 7 * pi / 8;
angle[8] fine = pi / 128;
a |= fine;
a ^= angle[4](pi / 4);
a &= angle[8](15 * pi / 8);          // a represents 5*pi/8.
h q;
p(a) q[0];
if ((~a & fine) != a) {
  negctrl @ inv @ pow(0.5) @ gphase(a) q[2]; // Phase -5*pi/16 when q[2] is zero.
}
pragma riken_braket.amplitudes
```

#### Scalar angle shifts

Scalar angle expressions support `<<`, `>>`, `<<=`, and `>>=` during constant
evaluation and runtime execution. Shifts operate on the stored bit pattern,
following the [OpenQASM shift rules](https://openqasm.com/language/classical.html#classical-bits-and-registers).
The result retains the left operand's angle width and sizedness. The count's
type and width do not promote or resize that angle.

The converter uses this angle-shift policy:

- Counts may be scalar UINT or nonnegative INT expressions. Negative signed
  counts fail before unsigned conversion. Boolean, bit, floating-point,
  complex, and angle counts require an appropriate explicit integer conversion
  where one is supported.
- Left shifts move bits toward higher indices; right shifts move them toward
  lower indices. Vacated positions are zero-filled and shifted-out bits are
  discarded. Left-shift results are masked to the angle width.
- A zero count preserves the angle. A count at least as large as its width
  produces zero, including counts such as native `UINT_MAX`.
- `a <<= count` and `a >>= count` behave like assigning the complete binary
  result to `a`, preserving self-reference and read-only constant checks.

For four-bit angle bits `1001`, shifting left by two gives `0100`, while
shifting right by two gives `0010`. Shifting by four or more gives `0000`.
An explicit cast such as `uint(-1)` produces an unsigned count according to
the existing cast rules; that oversized value produces zero rather than a
negative-count error.

Accepting nonnegative signed counts and producing zero for oversized counts
are documented converter choices. The specification describes unsigned counts
and discarding shifted-out bits but does not explicitly settle the boundary
cases in that section. Our policy matches bit-register shifts; scalar INT/UINT
shifts retain their existing requirement that the count be below the operand's
width. [Angle rotations](#scalar-angle-rotations-and-population-counts) have
separate signed-distance semantics.

Constant folding checks for oversized counts before shifting, avoiding enormous
intermediate integers. Runtime lowering captures the angle before the count,
evaluates each once, and preserves sources in private storage. It checks signed
negativity and branches to a zero result before an oversized native shift.
UINT counts are not narrowed or converted through signed INT. No new bra
instruction or floating-point shift calculation is required.

An oversized count does not suppress source conversion or access errors;
likewise a zero angle does not suppress errors while computing the count.
Evaluated constant errors fail conversion, and reached executable errors fail
at runtime. Skipped branches, short-circuit operands, and empty loops retain
their normal behavior, with names, types, widths, and read-only targets still
validated. Supported arithmetic, bitwise operators, comparisons, loops, and
phase gates can consume the angle result.

<!-- angle-shift-example -->
```qasm
OPENQASM 3.0;
const angle[4] A = 9 * pi / 8;        // Bits: 1001.
const angle[4] LEFT = A << 2;         // Bits: 0100.
const angle[4] RIGHT = A >> 2;        // Bits: 0010.
const angle[4] ZERO = A << 4;         // Bits: 0000.
angle[4] a = A;
uint[8] count = 1;
angle[4] left = a << count;          // Bits: 0010; width remains four.
angle[4] right = a >> count;         // Bits: 0100.
angle[8] after = a >> 1;             // Bits: 01000000; shift before widening.
angle[8] before = angle[8](a) >> 1;   // Bits: 01001000; widen before shifting.
a >>= count;                        // Bits: 0100.
a <<= 1;                            // Bits: 1000.
a >>= uint(-1);                     // Bits: 0000; oversized unsigned count.
bool skipped = false && bool(a << -1); // false; no negative-count error.
```

<!-- angle-shift-phase-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
angle[4] a = 9 * pi / 8;
uint[8] count = 1;
a >>= count;
a <<= 1;
a ^= angle[4](pi / 8);               // a represents 9*pi/8 again.
h q;
p(a) q[0];
if ((a >> 1) < a) {
  negctrl @ pow(0.5) @ gphase(a) q[2]; // Principal phase -7*pi/16 when q[2] is zero.
}
pragma riken_braket.amplitudes
```

#### Scalar angle rotations and population counts

`rotl`, `rotr`, and `popcount` accept constant and runtime scalar angle
expressions, operating on their stored bits. Both `angle[n]` and unsized `angle`
are supported; unsized angles use their resolved native UINT width.

| Function | Result |
| --- | --- |
| `rotl(a, distance)` | Angle with `a`'s width and sizedness; rotate toward higher bit indices |
| `rotr(a, distance)` | Angle with `a`'s width and sizedness; rotate toward lower bit indices |
| `popcount(a)` | Unsized native UINT containing the number of set bits within `a`'s width |

The [OpenQASM classical-instructions section](https://openqasm.com/language/classical.html#classical-bits-and-registers)
includes angles among the operands for these operations, but the
[built-in function table](https://openqasm.com/language/types.html#built-in-constant-expression-functions)
omits angle overloads. The signatures above document this converter's
interpretation. Existing UINT and bit-register overloads retain their rules,
including the explicit-width requirement for UINT inputs.

Rotation distances use the existing INT overload and input promotions from
BOOL and UINT. Conversion to native INT is checked before normalization:
UINT values above `INT_MAX` fail even if their remainder would be small or the
angle has no set bits. Distances are reduced modulo the angle width; negative
distances reverse direction, and zero or a multiple of the width preserves the
value. Unlike shifts, a distance at least the width does not produce zero.
`INT_MIN` is valid: normalization does not negate or take the absolute value of
the signed input. Scalar-bit, floating-point, complex, and angle distances need
an appropriate explicit conversion where supported.

For four-bit angle bits `1001`, `rotl(a, 1)` produces `0011`, `rotr(a, 1)`
produces `1100`, and `popcount(a)` produces ordinary integer `2`. Counting does
not return an angle or narrow the count to the input width. Its UINT result
works with existing arithmetic, comparisons, powers, casts, assignments,
and loop expressions. Cast it explicitly when a sized UINT input is needed by
another bit function, for example `rotl(uint[4](popcount(a)), 1)`.

Runtime lowering reuses existing UINT operations on private copies. Operands
are evaluated once in source order, with sources preserved until the complete
result is ready. A zero-distance guard avoids shifting by the width. Nested
calls, self-assignment, repeated execution, iterator shadowing, and supported
angle arithmetic, comparisons, shifts, bitwise operations, and phase gates are
supported. No new bra instruction is required.

Zero bits or an identity rotation do not suppress operand errors. Reached
runtime conversions, accesses, and arithmetic stay on their execution paths;
short-circuit operands and empty loops can skip them. Names, arity, types, and
widths are validated even on skipped paths. Evaluated invalid constant calls
fail conversion.

<!-- angle-bit-function-example -->
```qasm
OPENQASM 3.0;
const angle[4] A = 9 * pi / 8;        // Bits: 1001.
const angle[4] LEFT = rotl(A, 1);     // Bits: 0011.
const angle[4] RIGHT = rotr(A, 1);    // Bits: 1100.
const uint ONES = popcount(A);       // Integer 2.
const angle NATIVE = pi;
const uint NATIVE_COUNT = popcount(NATIVE); // Integer 1.
angle[4] a = A;
int distance = -1;
angle[4] left = rotl(a, distance);    // Bits: 1100; negative reverses direction.
angle[4] right = rotr(a, distance);   // Bits: 0011.
uint count = popcount(a);            // Integer 2; a remains 1001.
a = rotl(a, int(count));             // Bits: 0110.
bool same = popcount(a) == count;    // true.
uint[4] turned_count = rotl(uint[4](popcount(a)), 1); // Integer 4.
bool skipped = false && bool(rotl(a, uint(-1))); // No UINT-to-INT error.
```

<!-- angle-bit-function-phase-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
angle[4] a = 9 * pi / 8;
uint count = popcount(a);
a = rotl(a, int(count));             // a represents 3*pi/4.
h q;
p(a) q[0];
pow(popcount(a)) @ p(a) q[1];        // Integer power 2 gives phase 3*pi/2.
if (popcount(a) == count) {
  negctrl @ inv @ pow(0.5) @ gphase(a) q[2]; // Phase -3*pi/8 when q[2] is zero.
}
pragma riken_braket.amplitudes
```

### Angle-valued phase gates

`p`, `phase`, and `u1` accept scalar angle parameters and apply the diagonal
matrix `diag(1, exp(i*a))`. `cp` and `cphase` apply the corresponding controlled
phase, multiplying only the all-one control/target state. These are full
phase gates, not `rz`/`crz` operations with a discarded global phase.
As with the other standard gates, include `stdgates.inc` before using them.

With no floating-point powers, each set bit at position `j` of an `angle[n]`
with stored bits `m` contributes the phase $2\pi/2^{n-j}$.
The converter emits QCX `R` instructions with exponent
`n-j`; controlled forms use `CR`/`C<n>R`, and constant inversions use their
adjoint forms. For example, a four-bit angle with stored bits `0101` represents
$5\pi/8$ and decomposes into `R q 4` and `R q 2`. The decomposition is
mathematically exact; native complex amplitudes still use backend floating-point
precision. Phase exponents are not limited by the circuit's qubit count.

Named constants and width casts of those constants emit only set-bit
contributions. Runtime angle variables and executable angle casts are evaluated
once and copied into private UINT storage. Integer bit tests and `JUMPIF`
conditionally execute each contribution. Register broadcasting reuses the same
snapshot, and neither bit testing nor power evaluation changes the source.
Executable float-to-angle casts, including literal inputs, retain native
`ANGLEBITS` evaluation rather than substituting host constant folding.

`inv`, `ctrl`, `negctrl`, and constant/runtime integer `pow` modifiers are
supported, including nested combinations and controls added to `cp`/`cphase`.
Control operands retain their source order and polarity; negative controls use
QCX `wire#0`. All operands of a controlled operation must be distinct, and
broadcast register selections must have equal lengths.

An integer power `k` has angle bits $(m k) \bmod 2^n$. Constant powers are
reduced during conversion; runtime powers use unsigned multiplication and
masking, with separate sign handling. This avoids signed overflow and a gate
loop proportional to the exponent, including `INT_MIN`, `UINT_MAX`, and large
nested products. The emitted bit-test work is bounded by angle width and
broadcast size, subject to the existing output/expansion budgets.

Runtime integer exponents and angle expressions are still evaluated on reached
classical paths even for zero powers or inactive quantum controls. A zero
angle or zero power does not hide invalid qubit operands. Skipped branches
and empty loops do not execute runtime conversion, access, or arithmetic
checks, but names, types, widths, modifiers, and statically checkable qubit
operands are validated. Normal and fused execution are supported.

<!-- angle-phase-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
const angle[4] STEP = 5 * pi / 8;
angle[4] live = STEP;
int exponent = -3;
h q;
p(STEP) q[0];
negctrl @ pow(exponent) @ p(live) q[2], q[1];
cp(angle[8](live)) q[0], q[2];
pragma riken_braket.amplitudes
```

The negative-control operation above applies a phase of `pi/8` when `q[2]`
is zero and `q[1]` is one: `-3 * 5` reduces to `1` modulo `16`.

Constant floating-point powers, including integer-valued floats such as
`pow(2.0)`, instead use the complete angle's principal phase; see
[floating powers of angle-valued phase gates](#floating-powers-of-angle-valued-phase-gates).
Principal fractional powers cannot generally be distributed over the
individual `R` contributions.
Angle-valued rotations and U-family parameters remain unsupported.
Existing non-angle parameters keep their previous radian-based lowering and
floating-power behavior; they are not implicitly quantized into fixed-point angles.

#### Angle-valued global phase

The built-in `gphase(a)` also accepts constant and runtime scalar angles,
including width casts and executable float-to-angle casts. It multiplies the
entire state by `exp(i*a)`, without allocating an ancillary qubit. No
`stdgates.inc` include or declared qubits are required for uncontrolled `gphase`.
The same inverse, control, and integer-power rules above apply. Constant
floating-point powers use the whole-angle lowering described below.

Uncontrolled global phase uses the existing QCX `PHASE` instruction. With no
floating-point powers, each binary contribution starts from native `:TWO_PI` and is scaled by exact
powers of two in REAL storage; radian coefficients are not folded at Python
precision. Constant angles emit only their nonzero contributions. Runtime
angles use a private UINT snapshot and conditional bit tests, with the REAL
contribution halved at every bit position, including zero bits. Inversion
negates the REAL contributions. Both `QUBITS 0` and fusion are supported.

Controlled global phase has only condition qubits, not an additional target:

| OpenQASM operation | QCX contribution with exponent `k` |
| --- | --- |
| `ctrl @ gphase(a) c` | `R c k` |
| `ctrl(2) @ gphase(a) c1, c2` | `CR c1 c2 k` |
| `negctrl @ ctrl @ gphase(a) c1, c2` | `CR c1#0 c2 k` |

More controls use the corresponding multi-controlled `R` form. Inverse
contributions use adjoint forms, and control-register broadcasting reuses
the captured angle and exponent values. The complete global phase is retained
because it becomes a relative phase when quantum controls are added.

This zero-qubit example produces the scalar amplitude `-i`, up to backend
rounding: the two phases sum to `-5*pi/2`, equivalent to `-pi/2`.

<!-- angle-global-phase-zero-example -->
```qasm
OPENQASM 3.0;
const angle[4] STEP = 5 * pi / 8;
angle[4] live = STEP;
int exponent = -3;
inv @ gphase(STEP);
pow(exponent) @ gphase(live);
pragma riken_braket.amplitudes
```

Here a negative control applies the global phase only when `q[2]` is zero.
The inverse two-control operation applies its phase only when both `q[0]`
and `q[1]` are one; `q[2]` is not an extra target of that operation.

<!-- angle-global-phase-controlled-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
const angle[4] STEP = 5 * pi / 8;
angle[4] live = STEP;
h q;
negctrl @ gphase(live) q[2];
ctrl(2) @ inv @ gphase(STEP) q[0], q[1];
pragma riken_braket.amplitudes
```

Integer powers reuse modular UINT arithmetic, so work is bounded by angle
width and control broadcasting rather than the exponent's magnitude. Runtime
inputs are evaluated once on each reached classical path, even for zero powers
or inactive quantum controls. Skipped classical paths do not execute runtime
conversion, index, or arithmetic checks. Width/type/modifier validation,
zero-identity operand checks, source preservation, temporary cleanup, and
output budgets follow the angle-valued phase-gate rules above. This adds no
new QCX instruction or native storage type, and changes no build options.

#### Floating powers of angle-valued phase gates

Finite compile-time floating-point `pow` exponents are supported for `p`,
`phase`, `u1`, `cp`, `cphase`, and `gphase` with constant or runtime scalar
angles, including width casts and executable float-to-angle casts. A float
factor, even `2.0` or `0.0`, selects this path. Integer factors in the same
chain remain exact integers; integer-only chains keep the modular-bit lowering
above. Runtime floating exponents and mixed runtime-integer/floating chains
remain unsupported, including in empty loops.

For stored angle bits `m` and width `n`, the initial eigenphase is the exact
turn fraction `m / 2^n`, reduced to `(-1/2, 1/2]`. Every power selects the
principal phase of its input operation again, with the negative-real endpoint
assigned the positive half-turn. Modifiers act from rightmost to leftmost.
Inversions and nested powers retain their order, rather than combining all
exponents. For an angle equal to `pi`:

| OpenQASM modifier chain | Nontrivial final phase |
| --- | --- |
| `pow(0.5) @` | `pi/2` |
| `inv @ pow(0.5) @` | `-pi/2` |
| `pow(0.5) @ inv @` | `pi/2` |
| `pow(0.5) @ pow(2) @` | Zero: the inner square is identity |

The converter captures the raw angle bits once, then invokes native
`APPLY phase <- ANGLEBITS2PHASE bits width ...` for the complete chain. Use a bra
version providing this native function. Literal
`INV` and `POW` descriptors are serialized in evaluation order, opposite to
OpenQASM source order. Exact dyadic arithmetic preserves the initial and every
intermediate branch decision, even when REAL could not distinguish neighboring
angle bit patterns. The source is unchanged, and broadcasting reuses one
native result. Final gates use `U1`/controlled forms, or `PHASE` for uncontrolled
`gphase`. Positive and negative controls retain their original operands and
polarities; zero-qubit global phases and fusion are supported.

Floating exponent expressions are folded at Python binary64 precision, and
the native evaluator preserves those stored binary values. Integer factors
are serialized as exact decimal integers, rather than promoted to floats.
Only the final exact turn fraction is multiplied by the backend REAL
`:TWO_PI` and rounded once, nearest with ties to even. It is not requantized
to the source angle width. Final underflow is allowed; native gate arithmetic
still follows the existing floating-point environment. No compiler options or
floating-point controls are changed.

Metadata limits match bra: 1,024 `inv`/`pow` descriptors and 4,096 characters per
serialized exponent, including its sign. Intermediate reduced numerators and
denominators are each limited to 65,536 bits. Static metadata/type/operand
validation also applies in empty loops; runtime conversions, accesses, and
data-dependent arithmetic-size checks occur only on reached calls. Even zero
powers and inactive quantum controls evaluate the angle and complete chain.
A later zero cannot hide an earlier arithmetic-size error. Ordinary output
budgets and temporary-storage cleanup also apply.

<!-- angle-floating-power-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
const angle[4] HALF_TURN = pi;
angle[4] live = HALF_TURN;
h q;
pow(0.5) @ inv @ p(live) q[0];
inv @ pow(0.5) @ p(live) q[1];
negctrl @ pow(0.5) @ gphase(live) q[2];
pragma riken_braket.amplitudes
```

This applies `+pi/2` when `q[0]` is one, `-pi/2` when `q[1]` is one, and
`+pi/2` when `q[2]` is zero. The source `live` remains the half-turn bit pattern.
No qubits are needed for the following global-phase example; its sole amplitude
is `-i`:

<!-- angle-floating-global-example -->
```qasm
OPENQASM 3.0;
const angle[1] HALF_TURN = pi;
inv @ pow(0.5) @ gphase(HALF_TURN);
pragma riken_braket.amplitudes
```

The [native function documentation](../../docs/bra.md#classical-function-invocation)
and [design contract](../../docs/anglebits2phase-design.md) specify its literal
grammar, rounding, validation, and resource policies. Existing non-angle
floating powers retain their radian-based numerical behavior.

### Mathematical function calls

The following OpenQASM built-ins are supported. Overloads are selected
from the input types, never from the assignment destination:

| OpenQASM call | Input type | Result type | Runtime QCX operation |
| --- | --- | --- | --- |
| `exp(x)` | `float` | `float` | `APPLY result <- EXP operand` |
| `exp(z)` | `complex` | `complex` | `APPLY result <- EXP operand` |
| `sqrt(x)` | `float` | `float` | `APPLY result <- SQRT operand` |
| `sqrt(z)` | `complex` | `complex` | `APPLY result <- SQRT operand` |
| `log(x)` | `float` | `float` | `APPLY result <- LOG operand` |
| `sin(x)` | `float` | `float` | `APPLY result <- SIN operand` |
| `cos(x)` | `float` | `float` | `APPLY result <- COS operand` |
| `tan(x)` | `float` | `float` | `APPLY result <- TAN operand` |
| `sin(a)` / `cos(a)` / `tan(a)` | `angle[n]` or `angle` | `float` | Integer quadrant reduction, exact-axis handling, then existing `SIN`/`COS`/`TAN` calls |
| `arcsin(x)` | `float` | `float` | `APPLY result <- ARCSIN operand` |
| `arccos(x)` | `float` | `float` | `APPLY result <- ARCCOS operand` |
| `arctan(x)` | `float` | `float` | `APPLY result <- ARCTAN operand` |
| `ceiling(x)` | `float` | `float` | `APPLY result <- CEILING operand` |
| `floor(x)` | `float` | `float` | `APPLY result <- FLOOR operand` |
| `mod(x, y)` | `int`, `int` | `int` | Copy dividend to a temporary, then `LET temporary %= divisor` |
| `mod(x, y)` | `float`, `float` | `float` | `APPLY result <- MOD dividend divisor` |
| `popcount(bits)` | `bit[n]` | `uint` | Unsigned accumulation over register bits |
| `popcount(value)` | `uint[n]` | `uint` | Low-bit accumulation and right shifts of a private UINT copy |
| `popcount(value)` | `angle[n]` or `angle` | `uint` | Count the stored bits within the effective angle width |
| `rotl(value, distance)` / `rotr(value, distance)` | `bit[n]`, `int` | `bit[n]` | Snapshot and circular indexed copy |
| `rotl(value, distance)` / `rotr(value, distance)` | `uint[n]`, `int` | `uint[n]` | Guarded native shifts, bitwise OR, and width masking |
| `rotl(value, distance)` / `rotr(value, distance)` | `angle[n]` or `angle`, `int` | Input angle type | Reuse unsigned rotation lowering while preserving angle width and sizedness |
| `real(z)` | `complex` | `float` | `LET result := :REAL:operand` |
| `imag(z)` | `complex` | `float` | `LET result := :IMAG:operand` |

`real` and `imag` extract the real and imaginary components; they reuse bra's
existing component operators rather than introducing new native functions.
Standard scalar `bool`, `int`, and `uint` arguments promote to `float` for
`exp`, `sqrt`, `log`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`,
`ceiling`, and `floor`, or to `complex` for component extraction. A `float`
argument to `real` or `imag` also promotes to `complex`,
with a zero imaginary component. Thus
`complex z = exp(1);` evaluates the real exponential first, then converts its
result to complex. Use `exp(complex(1))` to select the complex overload.
Scalar `bit` arguments require an explicit numeric cast.

Calls can be nested and used in assignments, indexed array elements and
initializers, gate parameters, numeric comparisons, explicit casts, and
supported loop expressions. Whole arrays and bit registers are not scalar
arguments; `popcount`, `rotl`, and `rotr` explicitly accept register values.
A floating-point result is not itself a Boolean condition: compare
it with a number or use `bool(...)`.

Constant expressions, such as `const float E = exp(1);`, are evaluated by the
converter using exact integer arithmetic for integer `mod`, and Python's
`float`/`complex` precision for floating-point functions. Supported constant calls
can also supply declaration sizes, loop bounds, and modifier arguments where
their result type is accepted (use an integer cast when required). In
executable expressions, even literal calls such as `float x = exp(1);` use
native bra evaluation, not host mathematical-function folding. Runtime
promotion and evaluation therefore use bra's configured `REAL`/`COMPLEX` precision; declared
OpenQASM floating-point widths remain unenforced.

`exp` requires a finite argument (both components for complex inputs) and a
finite result. Invalid evaluated constant calls are conversion errors; reached
runtime failures are native `EXP` errors, currently uncaught by the bra CLI.
Underflow follows the applicable host or backend floating-point environment.
Component extraction does not add these finiteness checks. Classical branches,
short-circuit operators, and empty loops can skip runtime evaluation and its
failures; names, arity, and types are still validated in skipped code.

`sqrt` returns the principal square root. Its real overload requires a
nonnegative input, accepting real `-0` and preserving its sign. Negative
nonzero real inputs are errors, including stored negative subnormals. A
complex destination does not change this rule: `complex z = sqrt(-1);` is
invalid when reached. Use `sqrt(complex(-1))` to select the complex overload
and obtain `0 + 1im`. Constant calls use Python's `math.sqrt` or `cmath.sqrt`;
executable calls, including literal arguments, use native `SQRT`. Both paths
reject nonfinite inputs and nonfinite results, with the same distinction
between conversion-time errors and reached runtime errors as for `exp`.

The complex overload has a nonnegative real result component. The
converter/native signed-zero policy distinguishes the two sides of the
negative-real branch cut: input imaginary `+0` selects the positive imaginary
root, while `-0` selects the negative imaginary root. For example,
`sqrt(complex(-4))` is `0 + 2im`, whereas `sqrt(-complex(4))` has imaginary
component `-2`. On the nonnegative real axis, the imaginary zero sign is
preserved. Complex zero inputs return real `+0` and retain the imaginary zero
sign. Nearby nonzero components are not snapped to the cut; subnormal
arithmetic still follows the applicable floating-point environment.

`log` computes the natural logarithm and supports only a real argument. It
requires a strictly positive finite input: both signs of zero, negative
values (including stored negative subnormals), and nonfinite values are
errors. A complex destination merely converts the real result; it does not
enable a complex logarithm. Complex arguments are rejected even if their
imaginary component is zero; use `real(z)` explicitly when appropriate.
Constant calls use Python's `math.log`; executable calls, including literal
arguments, use native `LOG`. Invalid evaluated constants fail conversion,
whereas executable domain checks occur only when the call is reached.
Native `LOG` accepts positive stored subnormals for IEEE binary `REAL`
representations even under flush-to-zero/denormals-are-zero modes, by
normalizing them before library evaluation without changing those modes.
Other `REAL` representations reject subnormal inputs explicitly; normal
positive inputs remain supported. Arithmetic that has already flushed an
argument to zero is not repaired by `log`.

`sin`, `cos`, and `tan` accept both real-valued arguments in radians and
scalar angle arguments, returning `float` in either case. Complex arguments
are rejected, even with a zero imaginary part or a complex assignment destination.
For real-valued arguments, constant expressions use Python's `math.sin`, `math.cos`, and
`math.tan`; executable calls, including literal arguments, use native bra
`SIN`, `COS`, and `TAN`. Both paths reject nonfinite inputs and nonfinite
results. Invalid evaluated constants fail conversion; executable failures
occur only when reached, including in conditions and loop expressions.

For exact real zero, `sin` and `tan` preserve its sign and `cos` returns positive
one. Subnormal arithmetic follows the host or backend floating-point
environment and may flush to zero. Native calls use the C library at the
configured precision, without preliminary angle reduction or changes to
build flags or floating-point control modes. Large finite arguments remain
valid, subject to library accuracy. Values near zeros and tangent poles are
not snapped: an approximation of π/2 is not an exact pole, so its large
finite tangent is accepted. Constant and runtime results can differ because
their precisions and floating-point environments can differ.

For an `angle[n]` argument with stored pattern `bits`, these functions evaluate
the quantized angle `2*pi*bits/2^n`. Unsized `angle` uses native UINT width.
The original float used to initialize an angle is not recovered. Both constant
and runtime paths first reduce the pattern to a quadrant and a small signed
residual using exact integer arithmetic. Only that residual is converted to
radians, avoiding cancellation near axes and tangent poles. Runtime arguments
are captured once, and their source storage is preserved. Existing QCX math
functions evaluate the reduced argument using bra's configured REAL precision;
the radians conversion uses native `:PI` and an exact power-of-two scale.
Constant evaluation uses Python's floating-point precision.

At exact quadrantal angles, sine and cosine return exact `0`, `1`, or `-1`.
All exact-axis zero results, including tangent at zero and π, are positive zero.
Tangent at exactly π/2 or 3π/2 is an error, determined from the stored bits.
For example, `tan(angle[4](pi/2 + 1e-12))` fails when evaluated because the
conversion rounds onto a pole. Adjacent representable angles remain valid;
there is no tolerance-based pole snapping. These exact-axis and pole rules
are converter numerical policies. A failed constant call is a conversion error;
an executable call, even with a constant angle argument, checks poles only
when reached using QCX `ASSERT`. Skipped branches, short-circuit operands,
and empty loops suppress evaluation while still validating names, arity,
types, and widths. General angle-to-float casts remain unsupported.

`arcsin`, `arccos`, and `arctan` accept real-valued arguments and return
principal `float` values in radians, not `angle` values. Complex arguments
are rejected even if their imaginary component is zero or the destination
is complex; use `real(z)` explicitly when appropriate. Constant calls use
Python's `math.asin`, `math.acos`, and `math.atan`. Executable calls, including
literal arguments, use native `ARCSIN`, `ARCCOS`, and `ARCTAN`. The OpenQASM
function names are `arcsin`, `arccos`, and `arctan`, not the Python spellings.

Both paths require finite inputs and results. `arcsin` and `arccos` require
inputs in `[-1, 1]`, including the endpoints; values immediately outside the
domain are errors and are never clamped. `arctan` accepts every finite real
input. Principal result ranges are [-π/2, π/2], [0, π], and (-π/2, π/2),
respectively; rounding can produce ±π/2 for large finite `arctan` arguments.
`arcsin` and `arctan` preserve signed zero, `arccos(±0)` returns π/2, and
`arccos(1)` returns positive zero. Invalid evaluated constants fail conversion;
native runtime checks occur only when reached, so short-circuiting, branches,
and empty loops can skip failures. Input types and arity are still validated
in skipped expressions. Native calls preserve the existing build flags and
floating-point control modes; subnormal arithmetic follows the applicable
host or backend environment. General complex inverse functions and `atan2`
remain unsupported.

`ceiling` rounds toward positive infinity and `floor` toward negative infinity.
Both accept real-valued arguments and return `float`, not `int`; use an
explicit cast such as `int(floor(x))` where an integer is required. The
OpenQASM spelling is `ceiling`, not Python's `ceil`. Complex arguments are
rejected, independently of the assignment destination. Finite inputs and
results are required. Large finite values outside native integer ranges are
supported; native evaluation does not round through `INT` or `UINT`.

Both functions preserve exact `+0` and `-0`. In addition, `ceiling` returns
`-0` for values in (-1, 0), and `floor` returns `+0` for values in (0, 1).
Positive stored subnormals round to `1`/`+0`, and negative stored subnormals
to `-0`/`-1`, for `ceiling`/`floor` respectively, even under DAZ. Native calls
classify stored values and use same-sign normal representatives for subnormal
inputs without changing floating-point modes. Earlier arithmetic that has
already flushed an argument to zero is not repaired by these functions.

Constant calls use Python's `math.ceil`/`math.floor`, convert their arbitrary-
precision integer results back to `float`, and restore the appropriate zero
sign. Executable calls, including literal arguments, use native `CEILING` and
`FLOOR` at bra's configured precision. Invalid evaluated constants fail
conversion; native checks occur only when reached, so branches, short-circuit
operators, and empty loops can skip failures. Names, arity, and operand types
are still validated in skipped code. Native results are independent of the
caller's rounding direction, and build flags and floating-point control modes
remain unchanged.

`mod(x, y)` returns the truncation-toward-zero remainder. Its integer overload
comes first, so `int`, `uint`, and `bool` inputs select the signed `int` result.
Unsigned arguments must fit native QCX `INT`; conversion is checked and does not
fall back to the real overload when a value is too large. This differs from
the unsigned arithmetic promotions of `%`: for example, `mod(uint(-1), 3)` fails
its signed conversion even though the unsigned `%` operation is supported.
If either argument is `float`, both are promoted to `float` and the result is
`float`. Complex arguments are rejected; scalar `bit` arguments require an
explicit numeric cast. The destination never changes overload selection.

Integer constant evaluation uses exact integer arithmetic without converting
through floating point. Runtime calls, even with literal inputs, use native
`LET ... %= ...`; the minimum native signed integer modulo `-1` returns zero.
Real constant evaluation uses `math.fmod`, not Python's `%`, and runtime calls
use native `APPLY ... <- MOD ...`. A nonzero remainder has the dividend's sign;
real zero results preserve that sign, including `mod(-6.0, 3.0) == -0.0`.

Both real inputs must be finite, and both overloads reject zero divisors
(including either sign of real zero). Invalid evaluated constants fail
conversion. Reached integer zero-divisor failures identify `LET ... %= ...`;
real domain failures identify `APPLY ... <- MOD ...`. Calls skipped by branches,
short-circuiting, or empty loops do not perform arithmetic or conversions, but
both arguments' names and types are still validated. Temporaries preserve both
arguments and captured destination indices through nested calls and assignments.

Real calls avoid overflowing intermediate quotients and support stored subnormal
divisors without mistaking them for exact zero. Native final subnormal results
may flush to signed zero in the backend's existing arithmetic environment; host
constant evaluation and runtime need not agree on such underflow. Earlier
arithmetic that has already flushed an operand cannot be repaired. Declared
floating-point widths, compiler options, and floating-point control modes remain
unchanged. Floating-point `%` and `%=` remain unsupported.

`popcount(value)` counts the set (`1`) bits of an angle, bit register, or an explicitly
sized `uint[n]` without modifying it. It returns native, unspecified-width
`uint`, independently of the input width and destination type.

Register inputs can be bit-string literals, whole registers (including `bit[1]`),
static range/set selections, direct runtime ranges, arbitrary complement/shift
or complement/rotation chains on named mutable or constant
registers, or supported fixed-width register
`~`, `&`, `|`, `^`, `<<`, `>>`,
`rotl`, and `rotr` expressions. Repeated positions in a discrete selection are counted each
time they appear. Scalar `bit`, scalar bit-register elements, signed integers,
Booleans, and whole classical arrays are not accepted; use a one-position range
such as `bits[0:0]` when a one-bit register selection is intended.

Constant calls accept literal or named constant-register expressions and constant
sized-UINT or angle expressions, using exact bit counting. Mutable named bit registers are
runtime values even if their initializers are literals. Fixed-width runtime
register calls use a `UINT`
temporary initialized to the literal-bit subtotal, then add each live bit with
an explicit unsigned cast. Registers wider than native `UINT` are supported
without packing their contents into an integer; the register/selection width
must fit the native `UINT` range so the count cannot overflow.

Unsigned inputs support all existing sized-UINT widths, from one bit through
native QCX UINT width. Constants, variables, indexed UINT array elements,
and supported arithmetic, bitwise, shift, and rotation expressions are accepted
when their resulting type is explicitly sized `uint[n]`. Input expressions obey
their existing promotion and width-normalization rules before counting; no bits
above that resulting width contribute. For example, `popcount(uint[6](127))` is
six, and `popcount(uint[6](63) + uint[6](1))` is zero.

Unspecified-width `uint` inputs remain unsupported, including the native UINT
result of another `popcount` call. An explicit `uint[n](...)` cast can supply
the desired input width. Likewise, mixing a sized UINT with a native integer
literal can produce an unspecified-width result: `popcount(u + 0)` is rejected,
while `popcount(uint[6](u + 0))` and `popcount(u + uint[6](0))` accept a six-bit
input when `u` is `uint[6]`.

Angle inputs may be explicitly sized or unsized and use the same private-copy
counting algorithm. Their results are unsized native UINT. See
[scalar angle rotations and population counts](#scalar-angle-rotations-and-population-counts).

Runtime unsigned counting evaluates the input once, snapshots it into private
UINT storage, and repeatedly adds its low bit to a native UINT accumulator before
shifting the copy right by one. Zero inputs take no counting iterations; all-one
native UINT inputs require at most the native width. No signed conversion or
shift by the input width is used, including for `uint[1]`. The result retains
native UINT arithmetic semantics, and only assignment or an explicit cast to a
narrow destination masks the count. Indexed and self-referencing assignments
retain their existing checked-index and snapshot behavior.

The result works in unsigned arithmetic, powers, casts, numeric conditions,
indexed assignments, and supported loop expressions. As with other numeric
results, compare it with a number or use `bool(...)` for a Boolean condition.
Register-expression and count temporaries remain reserved through counting.
Skipped calls do not evaluate runtime shifts or their errors, but names, types,
widths, and static selections are validated even in empty loops. Other
expressions over raw runtime ranges, runtime angle-bit ranges, concatenation,
and aliases remain outside this `popcount` subset. No new native bra instruction
is required.

<!-- popcount-example -->
```qasm
OPENQASM 3.0;
bit[8] flags = "10110100";
uint count = popcount(flags);          // 4
uint selected = popcount(flags[2:5]); // 3
uint[6] value = 37;                   // "100101"
uint unsigned_count = popcount(value); // 3; value remains 37
const uint FULL_COUNT = popcount(uint[6](-1)); // 6
uint full_count = FULL_COUNT;
```

#### Population counts of runtime bit-register ranges

`popcount(flags[start:stop])` and `popcount(flags[start:step:stop])` accept
direct ranges of named mutable or constant bit registers. The selected length
may vary at runtime; no fixed-width UINT cast or destination width is required.
The result is an unsized native UINT, including when the source or selected
length exceeds the native UINT bit width.

Both endpoints are required. Start, step, and stop use supported INT/UINT
expressions and are captured once in that order; an omitted step is one.
The inclusive raw progression is generated before negative positions are
normalized against the source width. Only generated positions are accessed:
`1:2:6` selects `1, 3, 5`. Repeated normalized positions contribute each time,
so `flags[-1:3]` on a four-bit source counts bit 3 twice.

The complete selection is checked before any source bit is read or any
constant-register lookup words are prepared. Reached zero steps, empty
selections, invalid positions, or exceeded position capacity fail before
destination writes. An uncomplemented all-zero selected pattern returns zero;
an empty selection is an error.

The existing position buffer has capacity `min(2*N, MAX_RUNTIME_BIT_RANGE_POSITIONS)`
for source width `N`; the configured limit is currently 1,000,000 positions.
Source width and buffer capacity must fit QCX INT indexing. A larger reached
selection fails rather than allocating more storage. After validation, a runtime
loop reads each selected bit and adds it to a private UINT accumulator. Constant
sources reuse native UINT-word lookup storage. The selection is never packed
into a UINT or materialized as variable-width bit-register storage.

Calls use fresh source values and bounds on each loop visit. Their results work
in the existing scalar arithmetic, cast, array-index, loop, and gate-parameter
contexts. Logical short-circuiting and skipped branches/loops suppress runtime
capture and errors, while source visibility, bound types, and unsupported forms
remain checked. A runtime selection of a constant source is still a runtime
value and cannot initialize a `const` result. Temporary storage and caller state
are preserved on lowering failures.

<!-- runtime-range-popcount-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
bit[8] flags = "11010011";
int start = 7;
int step = -2;
int stop = 1;
uint from_constant = popcount(PATTERN[start:step:stop]); // 3: selected "1011".
uint selected = popcount(flags[start:step:stop]);       // 2: selected "1001".
start = -1;
step = 1;
stop = 7;
uint repeated = popcount(flags[start:step:stop]);       // 6: bit 7 occurs twice.
flags = "00000000";
uint fresh = popcount(flags[start:step:stop]);          // 0; selected is still 2.
```

Unary complement chains around the direct range are also supported:

```qasm
uint zeros = popcount(~flags[start:step:stop]);
uint ones = popcount(~~flags[start:stop]);
```

Complement applies to the selected positions, using their actual runtime
length, independently of the source-register and destination widths. For a
selection of length `n` containing `k` set bits, an odd number of `~` operators
returns `n - k`, and an even number returns `k`. Every occurrence of a repeated
normalized position contributes separately. Even complement parity still
validates the source and all bounds; it does not suppress reached errors.

Both parities capture and validate the range once, then read each selected bit
once. Odd parity initializes the native UINT accumulator to the captured
selection length and subtracts each original bit. Even parity uses the direct
counting path. This never requires a complemented register, packed UINT input,
second capture, or second source pass. The accumulator cannot underflow because
the number of set bits cannot exceed the validated selected length. All-zero
selected patterns produce `n` for odd parity; all-one patterns produce zero.
Constant sources, short-circuiting, storage limits, and cleanup retain the rules
above, and source registers remain unchanged.

<!-- runtime-range-popcount-complement-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
bit[8] flags = "11010011";
int start = 7;
int step = -2;
int stop = 1;
uint zeros = popcount(~flags[start:step:stop]);           // 2: selected "1001".
uint constant_zeros = popcount(~PATTERN[start:step:stop]); // 1: selected "1011".
uint twice = popcount(~~PATTERN[start:step:stop]);        // 3.
uint thrice = popcount(~~~PATTERN[start:step:stop]);      // 1.
start = -1;
step = 1;
stop = 7;
uint repeated_zeros = popcount(~flags[start:step:stop]);  // 3: nine selected positions.
uint repeated_ones = popcount(~~flags[start:step:stop]);  // 6: bit 7 occurs twice.
flags = "11111111";
uint fresh = popcount(~flags[start:step:stop]);           // 0; zeros is still 2.
```

A single left or right shift directly around a runtime range is supported:

```qasm
uint left = popcount(flags[start:step:stop] << distance);
uint right = popcount(flags[start:stop] >> distance);
```

The actual runtime selection length `n` is the shift width. Shifted-out bits
are discarded and vacated positions contain zero. For a nonnegative distance
`d < n`, a left shift counts original selected positions `[0, n-d)`, while a
right shift counts `[d, n)`. These are positions in the selected value, not
physical source indices. Selection order and repeated normalized positions are
preserved. For these uncomplemented shifts, zero distance counts all selected
bits and distance at least `n` returns zero, including when the source or
selection is wider than native UINT.

The complete range is captured and validated first, then the distance is
evaluated and captured once. Distances follow the existing register-shift rules:
supported INT/UINT expressions are accepted, negative INT values fail when
reached, and oversized distances produce an all-zero shifted pattern. Full-width
UINT counts are compared with the selected length before signed conversion or
index arithmetic. Arbitrarily large nonnegative literal counts also produce an
all-zero shifted pattern without putting an invalid native literal into the
emitted QCX. Invalid ranges still fail before distance
evaluation, even when the distance would yield zero. All-zero source patterns
do not suppress distance arithmetic or checks.

Only surviving selected positions are read after these checks; vacated zeros
need no storage or counting. Constant lookup initialization is also skipped on
the oversized-count path. Temporary declarations remain available to later
expressions that reuse storage after a zero result. Source registers remain
unchanged, and failures precede destination writes. Existing storage limits,
short-circuit behavior, and caller-state cleanup apply. No shifted bit-register
buffer, packed UINT input, or new bra instruction is required.

<!-- runtime-range-popcount-shift-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
bit[8] flags = "11010011";
int start = 7;
int step = -2;
int stop = 1;
int distance = 1;
uint left = popcount(flags[start:step:stop] << distance);     // 1: "1001" becomes "0010".
uint right = popcount(flags[start:step:stop] >> distance);    // 1: "1001" becomes "0100".
uint from_constant = popcount(PATTERN[start:step:stop] << distance); // 2: "1011" becomes "0110".
start = -1;
step = 1;
stop = 7;
uint repeated = popcount(flags[start:step:stop] << distance); // 5: nine selected positions.
uint huge = uint(-1);
uint zero = popcount(PATTERN[start:step:stop] >> huge);       // 0; range checks still run.
distance = 0;
uint fresh = popcount(PATTERN[start:step:stop] << distance);  // 5, after the oversized-count branch.
```

Unary complement chains may appear before and/or after that single shift:

```qasm
uint before = popcount((~flags[start:step:stop]) << distance);
uint after = popcount(~(flags[start:step:stop] << distance));
uint both = popcount(~((~flags[start:step:stop]) >> distance));
```

The two complement parities are tracked independently. Inner complement affects
the original selected bits before shifting. Outer complement also flips the
shift's zero-filled positions, so its count includes those positions as ones.
For selected length `n`, surviving length `m = max(n-d, 0)`, and `k` original
set bits in the surviving positions, the counts are:

| Inner complement parity | Outer complement parity | Result |
| --- | --- | --- |
| Even | Even | `k` |
| Odd | Even | `m - k` |
| Even | Odd | `n - k` |
| Odd | Odd | `n - m + k` |

An oversized distance has `m = k = 0`: odd outer parity returns `n`, and even
outer parity returns zero, regardless of inner parity. This result is initialized
before the early-result branch. No source bits or constant lookup words are read
on that branch, but the complete range and distance still undergo their checks.

All four cases reuse one range capture and one distance evaluation. Each surviving
selected position is read at most once; filled positions are accounted for through
the initial accumulator value. No complemented or shifted register is created.
The accumulator and its intermediate adjustments stay between zero and the
validated selected length.
Even complement chains and cancelling complements do not suppress validation,
distance arithmetic, or errors. Negative indices, repeated normalized positions,
mutable/constant sources, storage limits, short-circuiting, and cleanup follow
the same rules above.

<!-- runtime-range-popcount-complement-shift-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
int start = 7;
int step = -2;
int stop = 1;
int distance = 1;
uint plain = popcount(PATTERN[start:step:stop] << distance);       // 2: "1011" -> "0110".
uint before = popcount((~PATTERN[start:step:stop]) << distance);  // 1: "0100" -> "1000".
uint after = popcount(~(PATTERN[start:step:stop] << distance));   // 2: "0110" -> "1001".
uint both = popcount(~((~PATTERN[start:step:stop]) << distance)); // 3: "1000" -> "0111".
uint even = popcount(~~((~~PATTERN[start:step:stop]) << distance)); // 2.
uint huge = uint(-1);
uint full = popcount(~((~PATTERN[start:step:stop]) >> huge));     // 4; range checks still run.
uint empty_pattern = popcount((~PATTERN[start:step:stop]) >> huge); // 0.
distance = 0;
uint fresh = popcount(~((~PATTERN[start:step:stop]) << distance)); // 3, after the oversized-count branch.
```

`rotl` and `rotr` are also accepted inside `popcount` on a runtime bit-register
range, with complements before and after each call. For a single call:

```qasm
uint plain = popcount(rotl(flags[start:step:stop], distance));
uint before = popcount(rotr(~flags[start:stop], distance));
uint after = popcount(~rotl(flags[start:stop], distance));
uint both = popcount(~rotr(~flags[start:stop], distance));
```

A circular rotation preserves every selected bit. For selected length `n` and
original population count `k`, the result is `k` when the combined inner/outer
complement parity is even and `n - k` when it is odd. Complements commute with
rotation because no zero-filled positions are introduced.

The complete range is captured and validated once before evaluating the distance
once. The distance follows the existing rotation input rules: `int`, or `bool`
and `uint` promoted to `int`, with checked UINT-to-INT conversion. Negative
distances and native `INT_MIN`/`INT_MAX` are valid. A UINT distance above native
`INT_MAX` fails when reached, even though the distance cannot affect the count.
Distance arithmetic and conversions are still evaluated for all-zero sources
and cancelling complement chains. Invalid ranges fail before distance evaluation;
short-circuited and skipped paths suppress runtime evaluation while retaining
name, type, and syntax validation.

The converter counts the original selected positions using combined complement
parity. It does not materialize a rotated register or normalize or negate the
distance. The native UINT result, bounded position storage, negative indices,
repeated normalized positions, constant-source lookup, and cleanup follow the
direct runtime-range counting rules. Selections wider than native UINT remain
supported.

<!-- runtime-range-popcount-rotation-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
int start = 7;
int step = -2;
int stop = 1;
int distance = -1;
uint plain = popcount(rotl(PATTERN[start:step:stop], distance));   // 3: selection "1011".
uint before = popcount(rotr(~PATTERN[start:step:stop], distance)); // 1.
uint after = popcount(~rotl(PATTERN[start:step:stop], distance));  // 1.
uint both = popcount(~rotr(~PATTERN[start:step:stop], distance));  // 3.
distance = 1000;
uint even = popcount(~~rotl(~~PATTERN[start:step:stop], distance)); // 3.
uint checked_distance = 2;
uint checked = popcount(rotr(PATTERN[start:step:stop], checked_distance)); // 3; conversion still runs.
bool skipped = false && bool(popcount(rotl(PATTERN[start:step:stop], uint(-1)))); // false.
```

Nested `rotl` and `rotr` calls are supported over one named runtime range,
including complement chains before, between, and after the calls:

```qasm
uint plain = popcount(rotl(rotr(flags[start:stop], first), second));
uint middle = popcount(rotl(~rotr(flags[start:step:stop], first), second));
uint both = popcount(~rotl(~rotr(flags[start:stop], first), second));
```

Every rotation preserves the input width and population count. Complements
commute with rotations, so the result is the original count `k` when the total
complement parity is even, or `n - k` when it is odd, where `n` is the selected
length. This includes even complement groups and longer chains in either
direction. The converter counts the original selected bits directly; no
intermediate register, distance normalization, summation, or negation is needed.
Native `INT_MIN` and `INT_MAX` distances remain valid independently, even when
their sum or negation would exceed native INT limits.

The complete range is captured and validated once, before evaluating each
distance once from the innermost call outward. Every distance independently
undergoes the existing rotation promotions and checked INT conversion. A UINT
distance above native `INT_MAX` fails when reached, regardless of the selected
width, any other distance, or complement parity. Arithmetic and conversion
errors in later distances remain observable, including for all-zero sources
and cancelling rotations or complements. Evaluation stops at the first reached
error. Types, names, arity, and supported syntax are checked in skipped paths,
while runtime evaluation remains on reached paths.

The result remains native, unspecified-width UINT. Bounded position storage,
negative indices, repeated normalized positions, wide mutable/constant sources,
declaration placement, and temporary cleanup follow the existing range-counting
rules. Counting the main range and preparing its constant lookup follow all
distance evaluations and checks.

<!-- runtime-range-popcount-nested-rotation-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
int start = 7;
int step = -2;
int stop = 1;
int first = -1;
uint second = 2;
uint plain = popcount(rotl(rotr(PATTERN[start:step:stop], first), second)); // 3: selection "1011".
uint middle = popcount(rotl(~rotr(PATTERN[start:step:stop], first), second)); // 1.
uint both = popcount(~rotl(~rotr(PATTERN[start:step:stop], first), second)); // 3.
uint inner = popcount(rotl(rotr(~PATTERN[start:step:stop], first), second)); // 1.
uint even = popcount(~~rotl(~~rotr(~~PATTERN[start:step:stop], first), second)); // 3.
bool one = true;
uint promoted = popcount(rotr(rotl(PATTERN[start:step:stop], one), second)); // 3; both distances convert to INT.
first = 1000;
uint longer = popcount(rotr(~rotl(~rotr(PATTERN[start:step:stop], first), second), -7)); // 3.
bool skipped = false && bool(popcount(rotl(rotr(PATTERN[start:step:stop], first), uint(-1)))); // false.
first = 0;
second = 0;
uint fresh = popcount(rotl(rotr(PATTERN[start:step:stop], first), second)); // 3, after changing both distances.
```

Chains of two or more `<<` and `>>` operations are also accepted inside
`popcount` on a runtime range:

```qasm
uint count = popcount((flags[start:stop] << first) >> second);
uint longer = popcount(((flags[start:step:stop] >> first) << second) >> third);
uint complemented = popcount(~(((~flags[start:stop]) << first) >> second));
```

The shifts keep the original selected width at every step. Each shift discards
bits and fills vacated positions with zero. Opposite shifts cannot simply cancel:
`("1000" << 1) >> 1` is `"0000"`, not `"1000"`. Parentheses determine grouping;
unparenthesized shifts associate to the left.

For chains with complements only before the first shift or after the complete
chain, the four-case table above still applies, where `m` now means
the number of original selected positions surviving **every** shift and `k` is
their original population count. The converter tracks this surviving interval
and its current position with arithmetic bounded by the selected length; it
neither sums unbounded displacements nor constructs intermediate registers.

The complete range is captured and validated once. Distances are then evaluated
and checked once each, from the innermost shift outward, using the existing
INT/UINT shift rules. Even after an oversized or cumulative shift discards all
original bits, later distances still execute. For example, an expression such
as `popcount((flags[start:stop] << huge) >> (distance / zero))` still fails on
reached division by zero. A later negative distance also fails; neither an
all-zero pattern nor endpoint complements suppress those checks. Evaluation
stops at the first reached error.

Counting the main range's surviving bits and preparing its constant lookup
occur after all distance checks. For these endpoint-only chains, an empty
survivor interval returns zero, or the original selected length when the outer
complement parity is odd.
Declarations remain safe for reuse after that branch. Native UINT results,
bounded position storage, negative indices,
repeated normalized positions, mutable/constant sources, short-circuiting, and
cleanup follow the existing runtime-range counting rules. Selections wider than
native UINT remain supported.

<!-- runtime-range-popcount-nested-shift-example -->
```qasm
OPENQASM 3.0;
const bit[4] PATTERN = "1000";
int start = 0;
int stop = 3;
int left = 1;
uint right = 1;
uint original = popcount(PATTERN[start:stop]);                  // 1.
uint lost = popcount((PATTERN[start:stop] << left) >> right);   // 0; the high bit was discarded.
uint before = popcount(((~PATTERN[start:stop]) << left) >> right); // 3.
uint after = popcount(~((PATTERN[start:stop] << left) >> right)); // 4.
uint both = popcount(~(((~PATTERN[start:stop]) << left) >> right)); // 1.
uint third = popcount(((PATTERN[start:stop] >> left) << right) >> left); // 1.
uint huge = uint(-1);
uint filled = popcount(~((PATTERN[start:stop] << huge) >> right)); // 4; right is still evaluated.
bool skipped = false && bool(popcount((PATTERN[start:stop] << huge) >> -1)); // false; reached -1 would fail.
left = 0;
right = 0;
uint fresh = popcount((PATTERN[start:stop] << left) >> right);   // 1, after the empty-window branch.
```

Complements may also appear between successive shifts. Any chain of `~`, `<<`,
and `>>` over one named runtime bit-register range is supported inside `popcount`:

```qasm
uint middle = popcount((~(flags[start:stop] << first)) >> second);
uint longer = popcount(~((~(flags[start:step:stop] >> first)) << second));
uint even = popcount((~~(flags[start:stop] << first)) >> second);
```

Each complement flips every bit of its immediate operand, including zeros
introduced by earlier shifts. Later shifts can discard these generated ones.
Complement placement therefore matters: `popcount((~("0000" << 1)) >> 1)` is
three, while `popcount(~(("0000" << 1) >> 1))` is four. Parenthesize intermediate
shift results to complement them; `~` binds more tightly than `<<` and `>>`.
Consecutive complements reduce to parity, but complements separated by shifts
cannot generally be combined into endpoint parity.

The converter captures and validates the complete range, then evaluates and
checks each distance once in execution order. The original INT/UINT shift rules
apply: negative distances fail when reached, and oversized nonnegative
distances discard all input bits. UINT distances are bounded by the selected
length before signed conversion. Later distances still execute after complete
original-bit loss, even when complement parity or a known result could make
their values irrelevant. Even complement groups retain range and distance
validation. Short-circuited and skipped paths suppress runtime evaluation while
still validating names, types, and supported syntax.

After distance capture, the converter traces the final positions backward
through the chain. Positions supplied by a shift's zero fill contribute ones
under odd complement parity and are then removed from further tracing. The
remaining interval reaches the original selected bits and is counted with its
remaining parity. Generated-one contributions are retained when no original
bits survive. Interval coordinates, effective distances, and counts remain
bounded by the selected length, without packing the source into native UINT or
constructing intermediate registers. During the final counting pass, only
surviving original positions are read from the main source; its constant lookup
is skipped when none survive.
Existing storage limits, repeated normalized positions, wide mutable/constant
sources, declaration placement, and temporary cleanup are preserved.

<!-- runtime-range-popcount-interleaved-complement-example -->
```qasm
OPENQASM 3.0;
const bit[4] PATTERN = "0000";
bit[4] flags = "1011";
int start = 0;
int stop = 3;
int left = 1;
uint right = 1;
uint plain = popcount((PATTERN[start:stop] << left) >> right);    // 0.
uint middle = popcount((~(PATTERN[start:stop] << left)) >> right); // 3: "0000" -> "1111" -> "0111".
uint after = popcount(~((PATTERN[start:stop] << left) >> right)); // 4.
uint even = popcount((~~(PATTERN[start:stop] << left)) >> right);  // 0.
uint inverted = popcount(~((~(PATTERN[start:stop] << left)) >> right)); // 1.
uint huge = uint(-1);
uint generated = popcount((~(PATTERN[start:stop] << huge)) >> 0); // 4, with no original bits left.
uint retained = popcount((~(PATTERN[start:stop] << huge)) >> right); // 3.
uint longer = popcount((~((~(PATTERN[start:stop] << huge)) >> right)) << left); // 0.
bool skipped = false && bool(popcount((~(PATTERN[start:stop] << huge)) >> -1)); // false; reached -1 would fail.
left = 0;
right = 0;
uint fresh = popcount((~(flags[start:stop] << left)) >> right);   // 1, after a call with no original survivors.
```

Mixed `~`, `<<`, `>>`, `rotl`, and `rotr` chains over one named runtime
bit-register range are also supported inside `popcount`. Every operation acts
on the same selected length `n`. Rotation placement matters when a later shift
discards bits: rotating before shifting can change which bits survive, even
though rotation alone preserves the population count. Complements flip the
zero fill of earlier shifts, and later operations can move or discard those
new ones. Parenthesize intermediate shift results to complement them.

Range bounds are captured and validated once, then each distance is evaluated
once in execution order, from the innermost operation outward. Shift distances
must be INT or UINT and nonnegative when reached; values at least `n` discard
all bits without narrowing an oversized UINT. Rotation distances retain their
separate rules: INT, UINT, or Boolean inputs undergo checked INT conversion
before reduction modulo `n`. Negative rotations, including native `INT_MIN`,
are valid; a UINT rotation distance above native `INT_MAX` fails even when its
remainder would fit. All later distance checks still run after a shift discards
every bit. Skipped calls and short-circuited conditions suppress runtime errors
while still validating names, types, arity, and supported syntax.

After distance capture, the converter traces surviving logical positions
backward through disjoint intervals. Inverse shifts clip these intervals;
inverse rotations can split them at the circular wrap point. At most one new
interval is needed per rotation, and the number of intervals is also bounded
by the maximum selected length. Zero-fill contributions under odd complement
parity are counted separately. No intermediate bit register or packed UINT is
needed. Main-source bits are read only for surviving logical positions, with
repeated physical positions counted separately. Existing position-storage and
output limits, temporary cleanup, and native unspecified-width UINT result
semantics are preserved for mutable and constant sources.

<!-- runtime-range-popcount-mixed-operations-example -->
```qasm
OPENQASM 3.0;
const bit[4] PATTERN = "1011";
bit[4] flags = "1011";
int start = 0;
int stop = 3;
int rotation = 1;
uint shift = 1;
uint before = popcount(rotl(PATTERN[start:stop], rotation) << shift); // 3: "1011" -> "0111" -> "1110".
uint after = popcount(rotl(PATTERN[start:stop] << shift, rotation)); // 2: "1011" -> "0110" -> "1100".
uint middle = popcount(~(rotr((~PATTERN[start:stop]) << shift, -1) >> shift)); // 4.
uint even = popcount(~~(rotl(~~PATTERN[start:stop], rotation) << shift)); // 3.
uint huge = uint(-1);
uint filled = popcount(rotr(~(PATTERN[start:stop] << huge), -1) >> shift); // 3, with no original bits left.
bool skipped = false && bool(popcount(rotl(PATTERN[start:stop] << huge, huge))); // false; reached rotation conversion would fail.
start = -4;
uint repeated = popcount(rotl(flags[start:stop], rotation) << shift); // 6: eight logical positions, including aliases.
start = 0;
rotation = 0;
shift = 0;
flags = "0010";
uint fresh = popcount(rotl(flags[start:stop], rotation) << shift); // 1, using fresh bits and distances.
```

`popcount` also accepts `&`, `|`, or `^` between one direct runtime
bit-register range and an otherwise supported register expression with a known
width. The range can appear on either side. The fixed operand supplies the
required selected length `k`; the runtime range must have exactly that many
entries, without padding, truncation, or scalar broadcasting. Bit-string
literals, mutable/constant registers, static selections, runtime index sets,
and supported fixed-width bitwise, shift, rotation, and cast expressions can
supply that width. Scalar bits, integers, and whole classical arrays cannot.

Both operands are captured in written order. Each operand is completely
snapshotted before the next is evaluated, including borrowed reads from a fixed
register. At the range's evaluation point, start, step, and stop are captured
once; selected positions and the exact count are checked before reading its
bits or preparing its constant lookup. If the range is on the left, its checks
and snapshot precede all RHS evaluation. If it is on the right, fixed-LHS
accesses and arithmetic occur first. This capture/check timing is a converter
policy. A reached failure stops evaluation before later operands and before
assigning the population-count result to its destination.

Bits are paired by selection order, including repeated normalized source
positions. Neither source is changed. The result is native unspecified-width
UINT, even for registers wider than native UINT. The fixed width must fit QCX
INT and the existing range-position storage limit (currently 1000000). Position
storage is bounded by `k`, independently of the containing source width.
Snapshot and counting code grows with `k` and is also checked against the QCX
output limit. Scratch storage is released on success and failed lowering.
Skipped paths suppress runtime value checks while retaining static validation;
operands are recaptured on each reached visit.

<!-- runtime-range-popcount-bitwise-example -->
```qasm
OPENQASM 3.0;
bit[8] flags = "11010011";
const bit[8] PATTERN = "10110010";
bit[4] mask = "1010";
int start = 0;
int stop = 3;
uint left = popcount(flags[start:stop] & mask);       // 1: "0011" & "1010".
uint right = popcount(mask & flags[start:stop]);      // 1; reversed operand order.
uint united = popcount(flags[start:stop] | mask);     // 3: "1011".
uint different = popcount(mask ^ flags[start:stop]);  // 2: "1001".
uint constant_count = popcount(mask & PATTERN[start:stop]); // 1.
uint expression = popcount(flags[start:stop] & rotl(mask, -1)); // 1: "0011" & "0101".
start = -1;
uint repeated = popcount(flags[start:stop] ^ "10101"); // 2: five logical positions give "00111".
const bit[65] WIDE = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001";
start = 0;
stop = 64;
uint wide = popcount(WIDE[start:stop] & WIDE); // 2, without packing into native UINT.
bool skipped = false && bool(popcount(flags[start:0:999] & (mask << -1))); // false.
stop = 3;
flags = "00000001";
uint fresh = popcount(flags[start:stop] & mask); // 0, using the updated bits.
```

`popcount` also accepts `&`, `|`, and `^` between two direct runtime
bit-register ranges. Neither selected width needs to be known during conversion.
Each source may independently be mutable or constant, have a different containing
width, and use its own start, step, and stop expressions. Both ranges require
explicit start/stop bounds; their selected lengths must be equal when reached.
Bits are paired in selection order, preserving descending/nonunit strides,
negative indices, and repeated normalized source positions. Both operands may
select the same register, including overlapping or reversed selections.

The converter captures and validates the complete left range and snapshots all
its selected bits before evaluating any right-range bound. It then captures and
validates the right range, checks that its selected length equals the left
length, and only then prepares its constant lookup and reads its selected bits.
Each range's start, step, and stop are evaluated once in that order. After both
snapshots, a runtime loop combines corresponding bits and accumulates native
unspecified-width UINT. Sources remain unchanged, and reached failures prevent
later evaluation and assignment of the count to its destination. This timing
is a converter policy; the specification supplies the equal-width register
operation and slicing rules.

Each operand uses a position buffer and a bit-snapshot buffer, both with
capacity `min(2*N, L)`, where `N` is its containing register width and `L` is the
existing range-position limit (currently 1000000). The factor of two allows
negative aliases to select the same physical bit twice. Containing widths,
capacities, and selected counts must fit QCX INT. Oversized selections fail
before source reads for that operand; short selections from wider sources
remain supported. Runtime loops keep generated capture, snapshot, and counting
code independent of these capacities, apart from existing constant-word lookup
preparation. Output limits, declaration placement, scratch cleanup, skipped-path
validation, and fresh captures on loop reentry retain the existing rules.
No new bra instruction or build option is required.

<!-- runtime-range-pair-popcount-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[8] B = "11011101";
int first = 0;
int last = 3;
int start = 0;
int stop = 3;
uint both = popcount(a[first:last] & B[start:stop]); // 2: "1011" & "1101".
uint either = popcount(B[start:stop] | a[first:last]); // 4.
uint different = popcount(a[first:last] ^ B[start:stop]); // 2.
uint reversed = popcount(a[first:last] ^ B[stop:-1:start]); // 0: both selections are "1011".
first = -4;
stop = 7;
uint repeated = popcount(a[first:last] & B[start:stop]); // 4: eight entries; a's bits occur twice.
first = 0;
uint same = popcount(a[first:last] ^ a[first:last]); // 0; same-source selections are allowed.
bit[1] one = "1";
int index = -1;
uint singleton = popcount(one[index:index] & a[first:first]); // 1.
const bit[65] WIDE = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001";
int high = 64;
uint wide = popcount(WIDE[first:high] & WIDE[high:-1:first]); // 2, without native UINT packing.
int zero = 0;
bool skipped = false && bool(popcount(a[first:0:999] & B[start/zero:stop])); // false.
start = 4;
a = "0001";
uint fresh = popcount(a[first:last] & B[start:stop]); // 1, using fresh bounds and bits.
```

Both binary subsets also accept `~` chains on either operand and around the
binary result. Each complement flips only the bits within that operand or
result's selected width. Operand complements apply before `&`, `|`, or `^`;
result complements apply afterward. Consecutive complements at each position
reduce to parity, independently of complements at the other positions. Fixed
operands retain support for their existing nested register expressions and
internal complements.

Complement chains do not change selected lengths or relax any capture, bounds,
capacity, equal-length, or operand checks. Even complement groups and identities
whose result is known still evaluate both operands in written order. Reached
arithmetic and index errors remain observable; skipped calls and short-circuited
conditions suppress runtime evaluation while retaining static validation.
Sources and private snapshots remain unchanged. Counting flips individual 0/1
scratch bits with XOR by one, avoiding signed native `~` and bits beyond the
selected width. Results remain native unspecified-width UINT. Existing storage,
output, declaration-placement, and temporary-cleanup rules are preserved.

<!-- runtime-popcount-bitwise-complement-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[8] B = "11011101";
bit[4] mask = "1101";
int first = 0;
int last = 3;
int start = 0;
int stop = 3;
uint plain = popcount(a[first:last] & B[start:stop]); // 2: "1001".
uint left = popcount((~a[first:last]) & B[start:stop]); // 1: "0100".
uint right = popcount(a[first:last] & (~B[start:stop])); // 1: "0010".
uint both = popcount((~a[first:last]) & (~B[start:stop])); // 0.
uint outer = popcount(~(a[first:last] & B[start:stop])); // 2: "0110".
uint all = popcount(~((~a[first:last]) & (~B[start:stop]))); // 4: "1111".
uint even = popcount(~~((~~a[first:last]) & (~~B[start:stop]))); // 2; checks still execute.
uint odd = popcount(~~~((~~~a[first:last]) & (~~~B[start:stop]))); // 4.
uint fixed = popcount(~((~a[first:last]) & mask)); // 3.
uint reversed = popcount(~((~mask) & a[first:last])); // 3.
uint legacy = popcount(a[first:last] & ~mask); // 1; existing fixed-operand complements still work.
first = -4;
stop = 7;
uint repeated = popcount(~((~a[first:last]) | (~B[start:stop]))); // 4 across eight logical entries.
const bit[65] WIDE = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001";
int low = 0;
int high = 64;
uint wide = popcount(~(WIDE[low:high] ^ WIDE[high:-1:low])); // 65, complemented within the selected width.
int zero = 0;
bool skipped = false && bool(popcount(~~((~~a[first:0:999]) & (~~B[start/zero:stop])))); // false.
first = 0;
stop = 3;
a = "0001";
uint fresh = popcount(~((~a[first:last]) & B[start:stop])); // 2, using fresh bits.
```

Both binary subsets also accept finite chains of `<<`, `>>`, and `~` around
the binary result, for example `popcount(~((a[first:last] & mask) << distance))`
or `popcount(((a[first:last] ^ b[start:stop]) >> right) << left)`. Every shift
preserves the common selected width, discards displaced bits, and fills with
zeros. Complements flip every bit within that width, including zeros introduced
by earlier shifts. Complements separated by shifts therefore cannot be reduced
to a single parity for the whole expression.

The converter completely captures both binary operands in written order, with
all their existing bounds, capacity, equal-length, and read checks, before
evaluating any outer shift distance. Distances are then captured once each,
from the innermost shift outward. They must be INT or UINT; negative reached
distances fail. Distances at least the selected width discard all input bits,
including UINT distances above `INT_MAX` and arbitrarily large positive literal
distances. Runtime UINT distances are clamped before any conversion to INT.
Even after a shift discards all bits, later distances and their reached errors
still execute. These capture/check timings are converter policy. A reached
failure prevents assignment of the count to its destination. Skipped paths
retain static validation while suppressing runtime evaluation.

Counting traces the surviving interval backward through the outer operations
and accounts separately for complemented zero-fill bits. It combines surviving
snapshot bits without constructing an intermediate register or packing into
UINT. Both sources and their snapshots remain unchanged, and operands and
distances are recaptured on loop reentry. Results remain native unspecified-width
UINT, including for selections wider than native UINT. Pair counting uses a
runtime loop; the fixed-width counterpart retains its code growth with the
known width. Existing storage/output limits, declaration placement, and scratch
cleanup apply. No new bra instruction or build option is required.

<!-- runtime-popcount-bitwise-result-shift-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[8] B = "11011101";
bit[4] mask = "1101";
int first = 0;
int last = 3;
int start = 0;
int stop = 3;
uint distance = 1;
uint left = popcount((a[first:last] & B[start:stop]) << distance); // 1: "1001" -> "0010".
uint right = popcount((a[first:last] & B[start:stop]) >> distance); // 1: "0100".
uint outer = popcount(~((a[first:last] & B[start:stop]) << distance)); // 3: "1101".
uint before = popcount((~(a[first:last] & B[start:stop])) << distance); // 2: "0110" -> "1100".
uint mixed = popcount((~((a[first:last] ^ B[start:stop]) << distance)) >> distance); // 1: "0001".
uint operands = popcount(((~a[first:last]) ^ (~B[start:stop])) << distance); // 2.
uint fixed = popcount((a[first:last] & mask) << distance); // 1.
uint reversed = popcount(((~mask) & a[first:last]) >> distance); // 1.
uint united = popcount((mask | a[first:last]) >> distance); // 3.
uint huge = uint(-1);
uint filled = popcount((~((a[first:last] & mask) << huge)) >> distance); // 3; complemented zero fill survives.
uint literal = popcount((a[first:last] & mask) << 10000000000000000000000000000000000000000); // 0.
first = -4;
stop = 7;
uint repeated = popcount((a[first:last] & B[start:stop]) << distance); // 3 across eight logical entries.
const bit[65] WIDE = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001";
int low = 0;
int high = 64;
uint wide = popcount(~((WIDE[low:high] & WIDE[low:high]) << distance)); // 64, without native UINT packing.
bool skipped = false && bool(popcount(((a[first:last] & B[start:stop]) << huge) >> -1)); // false; a reached negative distance would fail.
first = 0;
stop = 3;
a = "0001";
uint fresh = popcount((a[first:last] & mask) << distance); // 1, using fresh bits.
```

Outer chains around either binary subset may also contain `rotl` and `rotr`,
mixed with `<<`, `>>`, and `~` in any order. Rotations wrap bits within the
common selected width; they do not introduce zero fill. A rotation alone does
not change population count, but a rotation followed by a shift can change
which bits survive. For example, with the four-bit result `1001`, rotating left
by one then shifting left by one gives `0110` (count 2), while shifting first
then rotating gives `0100` (count 1).

As for outer shift chains, both binary operands are fully captured and checked
before any outer distance is evaluated. Distances are then captured once each
from innermost to outermost, including nested distance expressions. Rotation
distances follow the existing `rotl`/`rotr` overload rules: signed INT with
ordinary BOOL/UINT promotions. Conversion to INT is checked before modulo
normalization, so a UINT above native `INT_MAX` fails even when its remainder
would fit. Negative distances reverse direction; zero, width multiples, and
`INT_MIN` are valid. Shift distances retain their separate nonnegative INT/UINT
rules, including clamping oversized UINT values before narrowing.

All reached distances and their conversions remain observable even for pure
rotations, even complement groups, or after an earlier shift discards all bits.
Skipped calls and short-circuited conditions suppress runtime errors while
retaining static validation. Reached operand or distance failures prevent
assignment of the population-count result to its destination. These evaluation
timings and native conversion checks are converter policy.

Rotations can split the surviving interval, so counting traces disjoint
intervals backward through the outer operations. It combines snapshot bits
within those intervals and accounts separately for complemented zero-fill
contributions. Four INT interval buffers have capacity bounded by the number
of rotations plus one, register widths, and the existing range-position limit.
Outer-result tracing needs no additional bit-register buffer or packed UINT.
Existing snapshot capture order, source preservation, native unspecified-width UINT results,
declaration placement, scratch cleanup, and storage/output limits apply.
Pair counting uses runtime loops; the fixed-width counterpart keeps code growth
with its known width. No new bra instruction or build option is required.

<!-- runtime-popcount-bitwise-result-rotation-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[8] B = "11011101";
bit[4] mask = "1101";
int first = 0;
int last = 3;
int start = 0;
int stop = 3;
int rotation = -1;
uint shift = 1;
uint left = popcount(rotl(a[first:last] & B[start:stop], 1)); // 2: "1001" -> "0011".
uint right = popcount(rotr(a[first:last] & B[start:stop], rotation)); // 2; negative reverses direction.
uint outer = popcount(~rotl(a[first:last] & B[start:stop], 1)); // 2: "1100".
uint before = popcount(rotl(a[first:last] & B[start:stop], 1) << shift); // 2: "0110".
uint after = popcount(rotl((a[first:last] & B[start:stop]) << shift, 1)); // 1: "0100".
uint mixed = popcount(rotr(~(rotl((~a[first:last]) ^ B[start:stop], rotation) << shift), 1) >> shift); // 2: "0101".
uint fixed = popcount(rotr(a[first:last] & mask, rotation) >> shift); // 1.
uint reversed = popcount(~rotl(mask ^ a[first:last], rotation)); // 2.
uint united = popcount(rotl(mask | a[first:last], rotation) >> shift); // 3.
int minimum = -2147483648;
uint endpoint = popcount(rotr(a[first:last] & B[start:stop], minimum)); // 2; no negation of INT_MIN.
bool distance = true;
uint promoted = popcount(rotl(a[first:last] & mask, distance) << shift); // 2; BOOL distance promotes to INT.
uint huge = uint(-1);
uint filled = popcount(rotr(~((a[first:last] & mask) << huge), rotation) >> shift); // 3; complemented zero fill survives.
first = -4;
stop = 7;
uint repeated = popcount(rotl(a[first:last] & B[start:stop], 1) >> shift); // 3 across eight logical entries.
const bit[65] WIDE = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001";
int low = 0;
int high = 64;
uint wide = popcount(~(rotl(WIDE[low:high] & WIDE[low:high], 1) >> shift)); // 64, without native UINT packing.
bool skipped = false && bool(popcount(rotl((a[first:last] & B[start:stop]) << huge, huge))); // false; reached rotation conversion would fail.
first = 0;
stop = 3;
a = "0001";
uint fresh = popcount(rotl(a[first:last] & mask, rotation) >> shift); // 1, using fresh bits.
```

Either or both runtime-range operands of `&`, `|`, and `^` inside `popcount`
may now have independent chains of `<<`, `>>`, `rotl`, `rotr`, and `~`.
For example, `popcount((a[first:last] << shift) & b[start:stop])` and
`popcount(rotl(a[first:last], rotation) ^ mask)` are supported. Each chain
preserves its range's selected width; both binary operand widths must still
match, even when shifts discard every bit. There is no padding, truncation,
or scalar broadcasting. Known-width register operands retain their existing
expression rules in either operand order. The outer-result chains described
above can wrap these binary combinations as well.

The converter completes the left operand before starting the right. For each
runtime-range operand, it captures and checks start, step, and stop; snapshots
all selected source bits; then captures every operation distance from innermost
to outermost and constructs the transformed snapshot. A known fixed width is
checked against the selected length before any range bit read or inner distance.
For two runtime ranges, the right length is checked against the left before
right-side reads or inner distances, after the entire left chain has completed.
Fixed-register operands, including their reads and arithmetic, execute at their
written position. Only after both operands are complete are outer-result
distances evaluated. These capture/check timings are converter policy.

Inner distances use the same shift and rotation rules as outer distances:
shifts accept nonnegative INT/UINT counts and clamp oversized counts before
narrowing; rotations use checked INT conversion followed by modulo
normalization, with ordinary BOOL/UINT promotions. All reached distances remain
observable after earlier shifts discard all bits, regardless of complement
parity or binary identities. Errors stop later operand/outer evaluation and
prevent assignment of the population count to its destination. Skipped paths
suppress runtime evaluation while retaining static validation.

Each transformed runtime-range operand uses one additional INT buffer for its
result bits. For two runtime ranges, each operand's position, raw-snapshot, and
transformed-snapshot buffers have capacity `min(2*N, L)`, with its containing
source width `N` and range-position limit `L`. A known fixed counterpart of
width `k` bounds these buffers by `k`, independently of the containing source
width. The factor of two in range-pair storage permits repeated negative aliases.
Buffer sizes do not grow with chain length. Runtime loops map output positions
back through the chain using bounded indices, and complemented zero-fill bits
are produced without reading an original position. Neither original sources
nor raw snapshots are modified, and no operand is packed into UINT.
Existing outer-result tracing, native unspecified-width UINT results,
declaration placement, temporary cleanup, storage/output limits, and fresh
captures on loop reentry apply. No new bra instruction or build option is required.

<!-- runtime-popcount-operand-chain-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[8] B = "11011101";
bit[4] mask = "1101";
int first = 0;
int last = 3;
int start = 0;
int stop = 3;
uint shift = 1;
int rotation = -1;
uint left = popcount((a[first:last] << shift) & B[start:stop]); // 1: "0110" & "1101".
uint right = popcount(a[first:last] & (B[start:stop] >> shift)); // 1: "1011" & "0110".
uint both = popcount(rotl(a[first:last], 1) ^ rotr(B[start:stop], rotation)); // 2: "0111" ^ "1011".
uint before = popcount(((~a[first:last]) << shift) & B[start:stop]); // 1: "1000" & "1101".
uint after = popcount((~(a[first:last] << shift)) & B[start:stop]); // 2: "1001" & "1101".
uint huge = uint(-1);
uint filled = popcount((~(a[first:last] << huge)) & B[start:stop]); // 3; complemented zero fill.
uint fixed = popcount(rotl(a[first:last], rotation) ^ mask); // 0.
uint reversed = popcount(mask ^ rotl(a[first:last], rotation)); // 0; reversed operand order.
uint nested = popcount(rotl(a[first:last], popcount((B[start:stop] << shift) & a[first:last])) & mask); // 2.
uint mixed = popcount(rotr((~rotl(a[first:last], rotation)) ^ (B[start:stop] >> shift), 1) << shift); // 1.
bool distance = true;
uint promoted = popcount(rotl(a[first:last], distance) | (B[start:stop] << shift)); // 4.
uint oversized = popcount(rotl(a[first:last], 1) & (~(B[start:stop] << huge))); // 3.
int minimum = -2147483648;
uint endpoint = popcount(rotr(a[first:last], minimum) ^ (B[start:stop] >> 0)); // 2; INT_MIN is valid.
uint self = 1;
self = popcount(rotl(a[first:last], self) & mask); // 2; distance uses the previous value.
first = -4;
stop = 7;
uint repeated = popcount(rotr(a[first:last], rotation) ^ (B[start:stop] >> shift)); // 3 across eight logical entries.
const bit[65] WIDE = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001";
int low = 0;
int high = 64;
uint wide = popcount(~(rotl(WIDE[low:high], 1) ^ (WIDE[low:high] >> shift))); // 62, without native UINT packing.
bool skipped = false && bool(popcount(rotl(a[first:0:999], huge) & B[start:stop])); // false; invalid bounds are not reached.
first = 0;
stop = 3;
a = "0001";
uint fresh = popcount(rotl(a[first:last], 1) & mask); // 0, using fresh bits.
```

`popcount` also accepts nested register-valued expression trees containing
`&`, `|`, and `^`, with `~`, shifts, and rotations at any level. For example,
`popcount((a[first:last] & mask) ^ b[start:stop])` and
`popcount(rotl((a[first:last] | b[start:stop]) & mask, distance))` need no
explicit cast or globally known selected width. Leaves may be direct runtime
bit-register ranges, runtime ranges on explicitly sized angle variables,
constants, array elements, and supported angle iterators, or existing supported
fixed-width register expressions. Direct runtime angle ranges are also accepted.
Both operands of every binary bitwise node must have equal lengths; unary
register operations preserve their child's length. There is no padding,
truncation, or scalar broadcasting. Scalar distance expressions may contain
supported nested calls, including other population counts.

For these new tree forms, the converter completes the left subtree before the
right subtree. A runtime-range leaf captures and validates all bounds/positions
before reading its selected bits; for an angle range, the angle value and any
array-element address are captured before the bounds. A fixed-width leaf captures its existing
expression, including borrowed register reads. Each binary node checks equal
lengths **after both children complete**, then combines their private snapshots.
A parent shift or rotation evaluates its distance only after its register child
is complete. Consequently, a right-child distance error can occur before that
parent binary node's length-mismatch check. Earlier failures suppress later
children, parent operations, and assignment to the count's destination. This
timing is converter policy. Existing simple-expression paths retain their
earlier capture and selected-length checks described above.

Runtime lengths are not inferred from containing widths, neighbouring fixed
leaves, or buffer capacities. A runtime-range leaf uses position and snapshot
buffers bounded by `min(2*N, L, INT_MAX)`, with containing width `N` and the
existing range-position limit `L`; a fixed leaf's snapshot is bounded by its
known width. Unary nodes inherit their child's capacity bound. A binary node's
private result capacity is the smaller child bound, safe once their lengths
have been checked. These bounds are not selected lengths or global width
constraints. Child storage is released after producing a private parent result,
so completed siblings can reuse buffers. Combining, unary transformations, and
counting use runtime loops; fixed-leaf capture and existing constant-lookup
preparation can still grow with known widths. Existing storage/output limits
apply to the complete generated program.

Sources remain unchanged, and parent operations do not modify their live child
snapshots. The tree's root bits are counted
into native unspecified-width UINT without packing the register into an integer,
including for selections wider than native UINT. Existing checked distance
conversions, later errors after oversized shifts, skipped-path validation,
declaration placement, caller-state preservation, temporary cleanup, and fresh
captures on loop reentry apply. No new bra instruction or build option is required.

<!-- angle-range-popcount-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int first = 1;
int last = 4;
bit[4] mask = "1101";
uint direct = popcount(phase[first:last]);          // 2
uint inverted = popcount(~phase[first:last]);       // 2
uint mixed = popcount(phase[first:last] ^ mask);    // 3
angle[4] small = angle[4]("1011");
first = -4;
last = 3;
uint repeated = popcount(small[first:last]);        // 6
```

<!-- runtime-popcount-tree-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[8] B = "11011101";
bit[4] c = "0011";
bit[4] mask = "0101";
int first = 0;
int last = 3;
int start = 0;
int stop = 3;
int rotation = -1;
uint shift = 1;
uint left = popcount((a[first:last] & B[start:stop]) ^ c[first:last]); // 2: "1001" ^ "0011".
uint right = popcount(a[first:last] ^ (B[start:stop] | c[first:last])); // 1: "1011" ^ "1111".
uint balanced = popcount((a[first:last] & B[start:stop]) | (c[first:last] ^ mask)); // 4: "1111".
uint outer = popcount(rotl((a[first:last] & B[start:stop]) ^ c[first:last], rotation) >> shift); // 1.
uint inner = popcount(rotl(a[first:last] & B[start:stop], 1) ^ (c[first:last] << shift)); // 2.
uint complemented = popcount(~((a[first:last] | B[start:stop]) & c[first:last])); // 2.
uint fixed = popcount((mask | "1000") ^ (a[first:last] & B[start:stop])); // 1; fixed subtree uses its existing reader.
bool distance = true;
uint promoted = popcount(rotl((a[first:last] & B[start:stop]) ^ c[first:last], distance)); // 2.
int minimum = -2147483648;
uint endpoint = popcount(rotr((a[first:last] & B[start:stop]) | c[first:last], minimum)); // 3.
uint huge = uint(-1);
uint oversized = popcount(((a[first:last] & B[start:stop]) << huge) ^ c[first:last]); // 2; right child still executes.
uint nested = popcount(rotl((a[first:last] & B[start:stop]) ^ c[first:last], popcount((a[first:last] ^ B[start:stop]) & c[first:last])) >> shift); // 1.
uint self = 1;
self = popcount((rotl(a[first:last] & B[start:stop], self) ^ c[first:last]) | mask); // 2; distance uses the previous value.
int zero = 0;
bool skipped = false && bool(popcount((a[first:last] & mask) ^ rotl(B[start:1], 1/zero))); // false; reached RHS distance would fail before the parent width check.
first = -4;
stop = 7;
uint repeated = popcount((a[first:last] & B[start:stop]) ^ c[first:last]); // 4 across eight logical entries.
const bit[65] WIDE = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001";
int low = 0;
int high = 64;
uint wide = popcount((WIDE[low:high] & WIDE[high:-1:low]) ^ (~WIDE[low:high])); // 65, without native UINT packing.
first = 0;
stop = 3;
a = "0001";
uint fresh = popcount((a[first:last] & B[start:stop]) ^ c[first:last]); // 1, using fresh bits.
```

This extends `popcount` only; other register-consuming contexts retain their
existing rules. Computed-result indexing and extensions to casts are outside
these tree forms; fixed-width leaves retain their existing supported cast rules.
For a known selected width, an explicit UINT cast can bridge to existing
sized-UINT operations, such as `popcount(uint[4](~(flags[start:stop] << count)))`.

#### Circular bit rotations

`rotl(value, distance)` moves bits toward higher indices and `rotr` toward lower
indices, wrapping displaced bits around to the other end. Both preserve the
input type and width, independently of the destination. Supported inputs are
bit-string literals, whole `bit[n]` registers (including `bit[1]`), static
range/set selections, nested register bitwise/shift/rotation expressions, and
explicitly sized `uint[n]` expressions within the supported native width, and
scalar `angle[n]` or `angle` expressions.
Scalar `bit`, signed integers, unspecified-width `uint`, and whole classical arrays
are not supported. An indexed UINT array element is a scalar
input; a one-position bit range is a register input, unlike a scalar bit element.
Runtime bit-register ranges can also supply rotation operands inside
[UINT casts](#casts-between-uint-values-and-bit-registers), where the cast
provides the required selection width, or inside
[population counts](#population-counts-of-runtime-bit-register-ranges), where
rotation chains with complements need no fixed selected width.

The distance parameter is `int`, with the usual built-in input promotions from
`bool` and `uint`. UINT-to-INT conversion is checked before normalization;
a UINT distance above native `INT_MAX` is an error, even if its remainder
would fit. Distances are reduced modulo the input width: negative values reverse
the direction, zero and multiples of the width leave the value unchanged, and
oversized magnitudes are valid within native INT range. For example, rotating
an eight-bit value left by `10` is equivalent to rotating left by `2`, and
rotating left by `-2` is equivalent to rotating right by `2`.

Constant calls use exact integer arithmetic. Literal and named constant-register
expressions can be constant inputs; mutable registers remain runtime values.
Runtime calls evaluate each operand once and
hold temporary results through the operation. Register snapshots protect
self-assignment and partially overlapping or reordered destinations. Selection
order defines bit significance, and repeated source positions remain distinct
positions in the selected value.

Unsigned and angle rotations use existing QCX shifts and bitwise OR, masking narrow
results before subsequent operations. A zero-distance guard avoids shifting by
the width. Register rotations do not pack into integers, so registers wider
than native UINT are supported. Runtime register distances require a register
width at most native `INT_MAX`; constant distances use direct bit remapping.
Signed-distance normalization does not negate the distance, so `INT_MIN` is
valid. No new bra instruction is required.

Types, names, widths, and static selections are validated even in empty loops.
Runtime conversions, distance arithmetic, and copy loops stay on reached paths,
including under Boolean short-circuiting. Existing arithmetic-overflow rules
still apply when computing a distance. Computed-result indexing and bare
register-valued conditions or loop bounds remain unsupported. Compare register
results using `==`/`!=`, use `popcount` to count set bits, or use an
[explicit same-width UINT cast](#casts-between-uint-values-and-bit-registers)
to interpret a supported register expression as an integer.

<!-- rotation-example -->
```qasm
OPENQASM 3.0;
bit[8] flags = "10001111";
int distance = -2;
bit[8] rotated = rotl(flags, distance); // "11100011"
bool restored = rotr(rotated, distance) == flags; // true
uint[6] value = 37;
uint[6] turned = rotl(value, 3); // 44 ("101100")
uint count = popcount(rotated); // 5
```

#### Complex construction and mathematical examples

The converter uses native `APPLY ... <- COMPLEX real imaginary` to construct
nontrivial complex literals directly, preserving component signs and avoiding
arithmetic assembly that could discard zeros or subnormals. This also applies
to variable and array initializers and constants copied into runtime storage.
Runtime complex negation uses a negative-zero seed and componentwise
subtraction instead of complex multiplication by `-1`, so `sqrt(-z)` retains
the branch-cut side. The constructor is native QCX functionality, not an
additional OpenQASM function: `complex(value)` remains the existing
one-argument OpenQASM cast.

This example combines a constant real root and a runtime complex root. The
complex root is `1 + 2im`, so the rotation angles are approximately `0.7` and `2`:

<!-- sqrt-function-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
const float SCALE = sqrt(0.49);
complex z = -3 + 4im;
complex root = sqrt(z);
rx(SCALE * real(root)) q;
rz(imag(root)) q;
```

This example uses a constant exponential, a runtime complex exponential, and
both component extractors. The two rotations use the real and imaginary parts
of the unit complex value:

<!-- math-function-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
const float SCALE = exp(-0.7);
float theta = 0.7;
complex unit = exp(theta * 1im);
float cosine = real(unit);
float sine = imag(unit);
rx(SCALE * cosine) q;
rz(sine) q;
```

This example combines a constant logarithm with a runtime logarithm. Both
rotation angles are natural logarithms of positive real values:

<!-- log-function-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
const float HALF_LOG = log(0.5);
float x = 2;
float theta = log(x);
rx(theta) q;
rz(HALF_LOG) q;
```

This example combines a constant cosine with runtime sine and tangent calls:

<!-- trigonometric-function-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
const float SHIFT = cos(0.5);
float x = 0.5;
rx(sin(x)) q;
rz(SHIFT + tan(x)) q;
```

Angle arguments also work in gate parameters. Here `a` stores exactly `3*pi/8`;
the constant cosine returns zero, and both executable calls return floats:

<!-- angle-trigonometric-function-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
const angle[4] QUARTER = pi / 2;
const float SHIFT = cos(QUARTER);  // Exactly zero.
angle[4] a = 3 * pi / 8;
rx(sin(a)) q;
rz(SHIFT + tan(-a)) q;
```

This example combines a constant inverse cosine with runtime inverse sine
and inverse tangent calls:

<!-- inverse-trigonometric-function-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
const float SHIFT = arccos(0.5);
float x = 0.5;
rx(arcsin(x)) q;
rz(SHIFT + arctan(x)) q;
```

This example combines a constant floor with runtime ceiling and floor calls.
The rotations use angles `1` and `-1` radians:

<!-- rounding-function-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
const float SHIFT = floor(-0.25);
float x = 0.25;
rx(ceiling(x)) q;
rz(SHIFT + floor(x)) q;
```

This example combines integer constant evaluation and a real runtime call.
The rotation angle is `1.5` radians:

<!-- mod-function-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
const int CONTROL = mod(7, 3);
qubit[CONTROL + 1] q;
float x = 7.5;
float y = 3.0;
rx(mod(x, y)) q[1];
```

Use a bra version with native `APPLY`, `EXP`, `SQRT`, `LOG`, `SIN`, `COS`, `TAN`,
`ARCSIN`, `ARCCOS`, `ARCTAN`, `CEILING`, `FLOOR`, `MOD`, integer `%=`, and `COMPLEX` support; see
[classical functions in bra](../../docs/bra.md#classical-function-invocation).
`COMPLEX` is also needed for general complex-literal construction in programs
without `sqrt` calls.
Other OpenQASM mathematical built-ins, user-defined functions, extern calls,
and discarded function-call statements remain unsupported. Native bra `ABS`
and `ARG` are not exposed as OpenQASM function extensions by this converter.

### Scalar bit constants

`const bit` declares an immutable compile-time scalar bit. Initializers may be
Boolean or scalar-bit constant expressions, or integer constant expressions
whose value is zero or one. This includes earlier scalar constants and static
scalar elements of constant bit registers. Existing explicit casts can be used
within the initializer, for example `bit(bool(2))`; direct `bit(2)` and implicit
floating-point initializers remain unsupported. A mutable value or measurement
result cannot initialize a constant. Even `false && live` is nonconstant when
`live` is mutable: short-circuiting does not remove its type dependency.

Named constants retain scalar BIT type and produce literal values without QCX
storage. They support the same expression subset as mutable scalar bits:
logical operations, supported comparisons, scalar-bit `~`, `&`, `|`, and `^`,
explicit casts, conditions, and copies into mutable scalar bits, indexed bit
destinations, and Booleans. Bitwise binary operands must both be scalar bits;
use `bit(true)` when a Boolean operand needs conversion. Arithmetic, numeric
function arguments, gate parameters, and integer indices/bounds require the
existing explicit casts. This feature does not broaden the supported operators
or conversions for scalar bits.

<!-- constant-scalar-bit-example -->
```qasm
OPENQASM 3.0;
const bit ON = true;
const bit[4] FLAGS = "1011";
const bit LOW = FLAGS[0];
const bit MASK = ON ^ LOW;         // zero, still scalar bit type
const bool READY = ON && !MASK;
bit live = MASK;
if (READY) { live = LOW; }          // live becomes one
bool ready = READY;                // true
uint value = uint(LOW);            // 1
```

Scalar `bit` remains distinct from `bit[1]`: `"1"`, a whole constant register,
or a one-position range/set selection cannot initialize `const bit`. Scalar
bits cannot be indexed or passed to register-only operations such as
`popcount` or register rotations.

Declarations retain source-order visibility; self-references, forward references,
and duplicate names are rejected. Constants are published only after successful
initialization. All writes, including compound assignments, indexed targets,
and measurement destinations, are rejected even in skipped branches, empty
loops, and after loop transfers. Loop iterators can shadow a constant, but are
themselves read-only; the outer constant is restored afterward. Constant-only
short-circuit expressions skip unreached value arithmetic while still checking
names and types. General block-local declarations remain unsupported.

### Constant bit registers

`const bit[n]` declares an immutable, compile-time register. Its width must be a
positive constant integer, and its initializer must be a constant register
expression of exactly that width. Bit-string literals and earlier constant
registers support `~`, `&`, `|`, `^`, `<<`, `>>`, `rotl`, and `rotr`, static
selections, `popcount`, and equal-width `==`/`!=` comparisons. Widths above the
native integer width are supported. No QCX storage is allocated for the constant;
runtime uses supply literal bits, including in mixed constant/mutable register
expressions, copies into mutable registers, conditions, and gate parameters.

As for mutable registers, index zero is the least significant bit. Range bounds
are inclusive, negative static indices count from the end, and the first position
in a range/set selection becomes the least significant selected bit. Repeated
source positions are allowed. `FLAGS[0]` is a scalar bit, whereas `FLAGS[0:0]`
and `FLAGS[{0}]` are one-bit registers.

Declarations retain source-order visibility: self-references, forward references,
and duplicate names are rejected. Widths are fixed in declaration scope, even
when loop iterators later shadow width constants. An iterator can also shadow
the register name without changing the original constant. All assignments,
compound assignments, indexed writes, and measurement destinations targeting
the constant are rejected, including in skipped branches and empty loops.
Short-circuiting suppresses unreached value arithmetic, but names, types, widths,
and static selection bounds are still validated.

<!-- constant-bit-register-example -->
```qasm
OPENQASM 3.0;
const bit[8] FLAGS = "10001111";
const bit[8] ROTATED = rotl(FLAGS, 2);
const bit[4] SELECTED = FLAGS[7:-2:1]; // "1101"
const uint COUNT = popcount(FLAGS);
bit[8] live = ROTATED;
bool restored = rotr(live, 2) == FLAGS; // true
uint count = COUNT;                    // 5
uint selected = popcount(SELECTED);    // 3
```

Named constant registers also support `for bit` iteration as described below.
Their runtime-dependent ranges can be [copied into registers or selections](#copies-from-runtime-bit-register-ranges)
or used as [bitwise compound-assignment RHS values](#runtime-range-rhs-bitwise-assignments).
Same-width [UINT casts](#casts-between-uint-values-and-bit-registers) also
accept constant registers, static selections, runtime index sets, and runtime
ranges, including supported bitwise, complement, shift, and rotation expressions.
Runtime ranges, arbitrary complement/shift/rotation chains, and bitwise
combinations of two runtime ranges or a range and a known-width register
operand, including operand/result complement chains, support [population counts](#population-counts-of-runtime-bit-register-ranges).
Supported runtime-range expressions can also be used in
[comparisons](#equality-with-runtime-bit-register-and-angle-bit-ranges),
[whole-register assignments and initializers](#whole-register-assignments-from-runtime-range-expressions),
[static-selection assignments](#assigning-runtime-range-expressions-to-static-selections),
[runtime index-set assignments](#assigning-runtime-range-expressions-to-runtime-index-sets), and
[runtime-range destination assignments](#assigning-runtime-range-expressions-to-runtime-ranges).
Other runtime-dependent constant-register range expressions, aliases, concatenation,
whole-array constants, and general block-local declarations remain unsupported.

### Unsigned integers

Scalar variables, constants, and casts support `uint` and `uint[n]`. Unsized
`uint` uses the backend's native C++ `unsigned int` width, denoted `W` here.
The converter derives `W` from the host's C `unsigned int` through Python's
`ctypes`; on a 32-bit unsigned-int host, `uint` ranges from zero through
`4294967295`. Generated programs require a bra version with native `UINT`
support and the same signed/unsigned integer representations as the converter
host; target widths are not detected automatically. See
[bra.md](../../docs/bra.md) for the native storage and conversion instructions.

A specified width must be a positive compile-time integer with `1 <= n <= W`.
Previously declared integer constants and supported constant expressions may
supply it; runtime variables and loop iterators may not. Nonpositive,
noninteger, runtime-dependent, and oversized widths are rejected, not ignored.

Unsigned assignment, integer-to-unsigned casts, unary negation, and unsigned
arithmetic normalize modulo `2**n` (or `2**W` for unsized `uint`). Narrow
values are stored in QCX `UINT` and masked with `LET name &= (2**n - 1)`.
Masks apply to intermediate unsigned results as well as completed assignments,
so folding and runtime evaluation agree:

```qasm
OPENQASM 3.0;
const uint[8] MASK = ~uint[8](0); // 255
uint[8] a = MASK;
uint[8] one = 1;
uint[8] wrapped = (a + one) / uint[8](2); // 0, not 128
a += one;                               // 0
uint[8] negative = -uint[8](1);           // 255
uint full = uint(-1);                    // native UINT maximum
```

Supported operations are `+`, `-`, `*`, `/`, `%`, `&`, `|`, `^`, `<<`, `>>`, unary `-`
and `~`, comparisons, and the corresponding compound assignments. Unsigned
division and remainder operate on nonnegative values. Complement flips the
bits of the unsigned operand's width. One-dimensional UINT arrays and
matching-width `for uint` iteration follow the same scalar rules, as described below.
General signed-width emulation remains unsupported.

For two unsigned operands, the wider width determines the result width.
When mixing an unsigned operand with native `int`, a narrower unsigned operand
promotes to `int`, which can represent all its values; a native-width unsigned
operand instead promotes the signed operand to native `uint`. Arithmetic,
bitwise operations other than shifts, and comparisons use the same rules. Shifts
retain the left operand's type and width, independently of the count. Bare integer literals
have `int` type, so they can change the intermediate result type:

```qasm
OPENQASM 3.0;
uint[8] a = 255;
uint[8] wrapped = (a + uint[8](1)) / uint[8](2); // 0
uint[8] promoted = (a + 1) / 2;                 // 128
int n = -1;
bool narrow = a > n;                          // true: signed comparison
bool native = uint(-1) == n;                   // true: unsigned comparison
```

Mixing UINT with an explicitly sized, non-native signed operand is rejected
because signed-width emulation is not implemented. `int[n](unsigned_value)`
is also rejected, including at the native width: this converter does not yet
implement fixed-width signed bit reinterpretation. Unsized `int(unsigned_value)`
and assignment to QCX `INT` are checked conversions, not bit reinterpretation;
values above `INT_MAX` produce an error. Compound assignments compute using
the operand promotion rules (or the left operand's type for shifts) before converting and, for narrow UINT targets,
masking the completed result.

Explicit `uint(...)` and `uint[n](...)` casts accept supported integer,
Boolean, scalar-bit, floating-point, and complex expressions. Integer casts
preserve integer precision and normalize to the target width. A floating-point
cast truncates toward zero, then requires a finite result in the native UINT
range before applying a narrow-width mask. Complex casts use the real
component. Thus `uint[8](257.9)` is `1`, `uint(-0.9)` is `0`, and `uint(-1.0)`
fails. Floating-point and complex assignments to UINT require an explicit
cast. Casts to `float` or `complex` use native bra conversions; `bool(uint_value)`
tests for nonzero, but bare UINT conditions remain unsupported.

Invalid conversions in evaluated constant expressions are converter errors.
In runtime expressions, checks execute only on paths that reach the conversion;
branches and logical short-circuiting can skip them. Native bra throws
`std::out_of_range` for an executed out-of-range conversion; its
CLI currently leaves this exception uncaught. Integer zero-divisor checks also
remain at the division instruction. Unsigned wraparound is defined by the
width normalization; this does not add general signed-arithmetic overflow checks.

UINT expressions can supply supported array/bit indices and `for int` range
bounds, steps, and set elements. Runtime captures convert to QCX `INT` with
range checks before index assertions or loop execution. Constant UINT loop
operands must also fit QCX `INT`; expansion never silently retypes an
unrepresentable unsigned value as an INT iterator. Negative indices and negative
steps require signed values: unsigned negation wraps, rather than producing a
negative value. `for uint` is supported for UINT arrays with matching or wider iterators, not
for integer ranges or discrete sets.

### Casts between UINT values and bit registers

`uint[n](bits)` and `bit[n](number)` explicitly reinterpret a pattern between a
bit-register expression and a scalar UINT value. Both constant folding and
runtime evaluation preserve every bit, including the highest native UINT bit.
The first register bit is the least significant bit: the integer value is
`sum(bit[i] * 2**i)`. These follow the OpenQASM
[bit-to-integer](https://openqasm.com/language/types.html#casting-from-bit) and
[integer-to-bit](https://openqasm.com/language/types.html#casting-from-int-uint)
cast rules.

Source and target widths must match, with `1 <= n <= W` for the native QCX UINT
width `W`. Unsized `uint(bits)` requires a register of width `W`, and
`bit[W](number)` accepts a source of unsized `uint` type. Cast widths must be
positive compile-time integers; runtime values and active iterators cannot
supply them, even when an iterator is unrolled or shadows a width constant.
The widths of previously declared variables and arrays remain fixed in their
declaration scope.

A width mismatch is rejected even if the current value would fit. There is no
implicit truncation, padding, or resizing. For example, converting a `bit[8]`
value directly with `uint[4](bits)` is invalid; select four bits explicitly.
Conversely, to resize a UINT numerically before extracting its pattern, use an
explicit numeric conversion such as `bit[4](uint[4](number))`. Scalar `bit` and
`bit[1]` remain distinct: these reinterpretations require the register form;
existing scalar bit/Boolean numeric conversions retain their separate rules.

Register operands may be bit strings, named mutable or constant registers,
static range/discrete-set selections, runtime discrete-set selections of named
mutable or constant registers, and supported fixed-width bitwise, shift,
rotation, and cast expressions. Selection order determines bit significance;
repeated source positions are retained. A selection from a register wider than
`W` is supported when its selected width is within `W`. Static selection
expansion is bounded by `W` before width inference materializes its positions.
Runtime ranges are supported directly and in bitwise, complement, shift, and
rotation expressions inside UINT casts, as described below. Runtime ranges in other
register-expression contexts, indexing or
iterating over computed register values, and aliases remain outside this UINT
subset. [ANGLE casts](#angle-casts-from-runtime-bit-register-ranges) have
their separate supported operand forms.

For `uint[n](flags[{i, j, ...}])`, the set's entry count must equal `n` and
fit within `W`; unsized `uint` requires exactly `W` entries. Each index must
have a supported INT or UINT type. Every occurrence is captured in source order,
including repeated indices and constant entries in a mixed runtime set. Signed
negative indices count from the end of the source. Checked integer conversion,
normalization, and bounds checks finish for all entries before selected bits
are read. The converter then snapshots those bits before packing the UINT,
without modifying the source. Duplicate source indices are permitted. Runtime
sets also work inside supported register expressions, such as
`uint[4](~flags[{i, j, 0, 1}])` and
`uint[4](rotl(flags[{i, j, 0, 1}], count))`.

Direct runtime ranges on named mutable or constant bit registers are also
accepted by `uint[n](flags[start:stop])` and
`uint[n](flags[start:step:stop])`. The target width must still be a positive
compile-time integer within native UINT capacity. Its width supplies the
required selection count; the converter does not infer a fixed register width
from runtime bounds. Unsized `uint` requires exactly `W` selected entries.

Start, step, and stop may independently use supported INT/UINT expressions.
Both endpoints are required; the default step is one. Operands are captured
once in start, step, stop order. The range generates its inclusive raw integer
progression before negative indices are normalized against the source width.
The stop bound need not itself be selected: `1:2:6` selects `1, 3, 5`. Only
generated positions are source accesses. Repeated normalized positions are
retained, so `flags[-1:3]` on a `bit[4]` source selects five entries,
including bit 3 twice.

For each runtime range operand, a reached zero step, empty selection, invalid
generated position, or count different from the target width fails before that
operand's selected bits are read. Its positions are captured and checked, then
its count is checked against the target width, then its selected bits are
snapshotted before use in the expression. The complete expression is evaluated
before any destination writes. Position and result storage are bounded by the target
width; a longer selection fails when it exceeds that capacity. Constant sources
reuse the existing native UINT-word lookup machinery. Raw range advancement
preserves native INT/UINT operands without signed overflow or narrowing
full-width UINT values.

Within the UINT cast, a runtime range may also be wrapped in unary complement
(`~`) and left/right shifts (`<<`, `>>`), including nested combinations:

```qasm
uint[4] inverted = uint[4](~flags[start:stop]);
uint[4] shifted = uint[4](flags[start:step:stop] << count);
uint[4] chain = uint[4](~((flags[start:stop] << left_count) >> right_count));
```

The target width supplies the required count throughout the chain. Each
operation preserves that width; shifted-out bits are discarded and vacated
positions contain zero. In particular, `(range << a) >> b` cannot recover bits
lost in its first shift, and `~(range << a)` can differ from `(~range) << a`.
Both sized and unsized UINT targets follow the width rules above.

The converter first captures and validates the range and snapshots all selected
bits, then applies operations from the innermost expression outward. Each
shift-count occurrence is evaluated once after its left operand, even when
multiple counts use identical source text. Counts may use supported INT/UINT
expressions; negative counts fail when reached, and counts at least the selected
width produce zero. An earlier zero result does not suppress later shift-count
arithmetic or checks. Nested UINT casts in counts perform their own range
captures, including when they access the same named register.

Binary register operations `&`, `|`, and `^` may also combine runtime ranges
inside a UINT cast. Ranges may appear on either or both sides, alongside
supported fixed-width register operands, with nested complement and shifts:

```qasm
uint[4] masked = uint[4](flags[start:stop] & "1010");
uint[4] combined = uint[4](left[a:b] ^ right[c:d]);
uint[4] nested = uint[4](~((left[a:b] | right[c:d]) << count));
```

Every register operand must have the target UINT width. Fixed-width mismatches
are rejected during conversion; runtime ranges are checked against that width
when reached. There is no broadcasting, padding, truncation, or width promotion
between register operands. Scalar INT/UINT values and scalar bits cannot be
mixed directly with registers; use a matching-width bit string or a supported
explicit `bit[n](number)` conversion. Existing fixed-width operand forms remain
available, including named registers, static and runtime index-set selections,
rotations of fixed-width registers, and supported angle/UINT-to-bit conversions.

Runtime value evaluation follows the expression tree from left to right. The
left subtree is completed, including its snapshots, shift counts, rotation
distances, and arithmetic, before the right subtree begins. Each range
occurrence captures its own bounds and bits when reached, even when accesses
use identical source text. Fixed register
reads are also snapshotted at their operand's position. A left-side failure
prevents right-side evaluation; a right-side failure may occur after left-side
reads but still precedes all destination writes. Binary bitwise operators do
not short-circuit, so zero masks, all-one masks, or repeated operands do not
suppress range checks or other operand evaluation.

`rotl` and `rotr` may wrap runtime ranges and supported register expression
trees inside a UINT cast. They preserve the required cast width and rotate
within the selected positions, rather than within the entire source register:

```qasm
uint[4] left = uint[4](rotl(flags[start:stop], distance));
uint[4] right = uint[4](rotr(left_bits[a:b] ^ right_bits[c:d], distance));
```

Each call evaluates and snapshots its complete register argument before
evaluating its distance once. Nested rotations finish from the inside outward.
Distances follow the existing [rotation rules](#circular-bit-rotations): the
parameter type is INT, with supported Boolean and UINT input promotions.
UINT-to-INT conversion must succeed before normalization, even if the remainder
would fit. Negative distances reverse direction; distances are reduced modulo
the selected width, and zero or multiples of that width preserve the bits.
These rules also apply at native INT endpoints and for width-one selections.
A nested UINT cast in the distance captures its own operands at that point.

Intermediates retain the target width and private storage until their parent
operation is complete. Source registers remain unchanged, overlapping
assignments use original values, and failed expressions do not partially
update the destination. Temporary storage and caller snapshot bindings are
restored after lowering failures.

Other register functions applied directly to raw runtime ranges,
aliases, and further indexing of range results
remain outside this subset. The resulting scalar UINT may supply supported
arithmetic, indices, loop bounds, or an enclosing `bit[n]` cast. General
variable-width register expressions remain limited to the explicitly documented
contexts, including [ANGLE casts](#angle-casts-from-runtime-bit-register-ranges).

#### UINT casts from runtime angle-bit ranges

`uint[n](phase[first:last])` and `uint[n](phase[first:step:last])` accept
runtime ranges on explicitly sized angle variables, constants, array elements,
and supported angle iterators. The selected count must equal the target width;
unsized `uint(...)` uses native UINT width. The first selected bit becomes the
least significant result bit. Negative indices and repeated source positions
are preserved, and there is no floating-point conversion.

Ranges can occur inside nested complement, bitwise, shift, and rotation trees,
including mixed expressions with bit-register ranges. The target width applies
to every register operand and intermediate. Source and range operands are
captured once in source order. Each range checks its positions and count before
extracting bits, and the complete result is ready before any destination write.
Skipped paths defer runtime domain checks while still validating names, types,
and widths. The result retains UINT width and arithmetic semantics.

Runtime angle-bit single-index and discrete-set UINT casts are also supported,
as described below. Unsized angle sources, computed-source
indexing, omitted range endpoints, and direct numeric casts from angles remain
outside this subset.

<!-- runtime-angle-range-uint-cast-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
bit[4] mask = "0101";
int first = 1;
int last = 4;
uint[4] saved = uint[4](phase[first:last]); // 10.
uint[4] combined = uint[4](rotl(phase[first:last] ^ mask, 1)); // 15.
bit[4] copied = bit[4](saved); // "1010".
array[angle[8], 2] phases = {phase, phase};
int slot = 1;
uint[4] element = uint[4](phases[slot][first:last]); // 10.
phase = angle[8]("00000000");
uint[4] fresh = uint[4](phase[first:last]); // 0; saved is still 10.
```

#### UINT casts from runtime angle-bit indices and sets

`uint[1](phase[index])` copies a selected angle bit to a one-bit UINT.
`uint[n](phase[{i,j,...}])` requires exactly `n` entries; an unsized target
uses native UINT width. Sources may be explicitly sized mutable or constant
angles, angle-array elements, or supported angle iterators. The first selected
bit becomes the least significant UINT bit. Negative indices are normalized
against the source width, and repeated source positions are retained.

The selections can appear in supported complement, bitwise, shift, and
rotation expressions. Matching-width `bit[n](angle[n](...))` round trips can
also supply register operands to a UINT cast. The source value and each index
occurrence are captured once in source order, including repeated indices. All
indices are validated before extracting any selected bits, and the complete result is computed before
destination writes. This preserves overlapping assignments. Skipped paths defer
runtime domain checks while still validating names, types, and widths. The
result retains UINT width and arithmetic semantics. Bit selections on unsized
angle sources and direct numeric casts from angles remain unsupported.

<!-- runtime-angle-selection-uint-cast-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int index = -1;
uint[1] one = uint[1](phase[index]); // 1.
uint[4] saved = uint[4](phase[{index,0,0,1}]); // 7: "0111".
bit[4] mask = "0101";
uint[4] combined = uint[4](rotl(phase[{index,0,0,1}] ^ mask, 1)); // 4.
array[angle[8], 2] phases = {phase, phase};
int slot = 1;
uint[4] element = uint[4](phases[slot][{index,0,0,1}]); // 7.
phase = angle[8]("00000000");
uint[4] fresh = uint[4](phase[{index,0,0,1}]); // 0; saved is still 7.
```

UINT operands may use supported scalar expressions, including indexed UINT
array elements and supported UINT array iterators. The source width is
inferred from the complete expression before any conversion: shifts preserve
the left operand's width, while binary UINT arithmetic promotes widths according
to the existing UINT rules. For example, `bit[4](u4 + u8)` is invalid because the
sum has width eight. Narrow UINT arithmetic still wraps at every operation,
so `bit[4](uint[4](15) + uint[4](1))` produces `"0000"`. Mixed signed/unsigned
arithmetic can produce INT and then needs an explicit numeric UINT conversion.

Register results work in existing register initializers, assignments, bitwise
expressions, comparisons, and register functions. UINT results work in existing
numeric contexts, including arithmetic, conditions with an explicit comparison
or `bool`, supported indices, integer loop bounds/steps/sets, gate parameters,
and shift counts. A shift count's integer width need not match the shifted
register's width:

<!-- uint-bit-cast-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
const uint[4] SELECTED = uint[4](PATTERN[7:-2:1]); // 11: selected bits are "1011".
bit[4] bits = bit[4](SELECTED);
uint[4] packed = uint[4](bits);                   // 11.
uint[4] wrapped = uint[4](bit[4](packed + uint[4](7))); // 2, after modulo-16 addition.
bits[3:-1:0] = bit[4](uint[4](bits));             // Overlap-safe reversal: "1101".
bit[8] flags = "10001111";
bit[4] count_bits = "0001";
flags <<= uint[4](count_bits);                   // Shift by one: "00011110".
array[uint[4], 2] values = {uint[4](bits), wrapped};
uint total = 0;
for uint[4] value in values {
  total += uint[4](bit[4](value));                // Final total is 15.
}
```

A runtime set can read either a constant or mutable source. Saving the UINT
retains its value after the index or source changes:

<!-- runtime-bit-set-uint-cast-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
bit[8] flags = "11001001";
int i = 7;
uint j = 0;
uint[4] from_constant = uint[4](PATTERN[{i, j, -1, 1}]); // 13: "1101".
uint[4] packed = uint[4](flags[{i, j, -1, 1}]);          // 7: "0111".
bit[4] back = bit[4](packed);
flags[0:3] = bit[4](uint[4](flags[{i, j, -1, 1}]));     // Snapshot before writes: "11000111".
i = 0;
uint[4] fresh = uint[4](flags[{i, j, -1, 1}]);           // 15; packed is still 7.
```

A direct runtime range preserves its selected order and checks its width
before any overlapping assignment:

<!-- runtime-bit-range-uint-cast-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
bit[8] flags = "11001001";
int start = 7;
int step = -2;
int stop = 1;
uint[4] from_constant = uint[4](PATTERN[start:step:stop]); // 11: "1011".
uint[4] packed = uint[4](flags[start:step:stop]);          // 5: "0101".
bit[4] back = bit[4](packed);
flags[0:3] = bit[4](uint[4](flags[start:step:stop]));      // Snapshot first: "11000101".
start = 4;
step = 1;
stop = 7;
uint[4] fresh = uint[4](flags[start:step:stop]);           // 12; packed is still 5.
int negative = -1;
uint[5] repeated = uint[5](flags[negative:3]);            // 11: "01011".
```

This example distinguishes complement placement and intermediate truncation:

<!-- runtime-range-unary-shift-uint-cast-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
bit[8] flags = "11001001";
int start = 7;
int step = -2;
int stop = 1;
int distance = 1;
uint[4] from_constant = uint[4](~PATTERN[start:step:stop]); // 4: complement of "1011".
uint[4] inverted = uint[4](~flags[start:step:stop]);       // 10: complement of "0101".
uint[4] left = uint[4](flags[start:step:stop] << distance); // 10.
uint[4] after = uint[4](~(flags[start:step:stop] << distance)); // 5.
uint[4] before = uint[4]((~flags[start:step:stop]) << distance); // 4.
uint[4] truncated = uint[4]((PATTERN[start:step:stop] << distance) >> distance); // 3, not 11.
flags[0:3] = bit[4](uint[4](~(flags[start:step:stop] << distance))); // "11000101".
start = 0;
step = 1;
stop = 3;
distance = 4;
uint[4] ones = uint[4](~(flags[start:step:stop] << distance)); // 15; after is still 5.
```

This example combines ranges on mutable and constant sources and performs
an overlapping assignment only after the complete result is ready:

<!-- runtime-range-bitwise-uint-cast-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
bit[8] flags = "11001001";
int start = 7;
int step = -2;
int stop = 1;
int count = 1;
uint[4] both = uint[4](flags[start:step:stop] & PATTERN[start:step:stop]); // 1.
uint[4] either = uint[4](flags[start:step:stop] | PATTERN[start:step:stop]); // 15.
uint[4] different = uint[4](flags[start:step:stop] ^ PATTERN[start:step:stop]); // 14.
uint[4] with_mask = uint[4](flags[start:step:stop] ^ "0011"); // 6.
uint[4] chain = uint[4](~((flags[start:step:stop] ^ PATTERN[start:step:stop]) << count)); // 3.
flags[0:3] = bit[4](uint[4](flags[start:step:stop] ^ PATTERN[start:step:stop])); // "11001110".
start = 0;
step = 1;
stop = 3;
uint[4] fresh = uint[4](flags[start:step:stop] ^ PATTERN[start:step:stop]); // 12; different is still 14.
```

Rotations use the selected order and preserve snapshots during overlapping
assignments:

<!-- runtime-range-rotation-uint-cast-example -->
```qasm
OPENQASM 3.0;
const bit[8] PATTERN = "10110010";
bit[8] flags = "11010011";
int start = 7;
int step = -2;
int stop = 1;
int distance = 1;
uint[4] left = uint[4](rotl(flags[start:step:stop], distance)); // 3: rotate "1001" left.
uint[4] right = uint[4](rotr(flags[start:step:stop], distance)); // 12.
uint[4] from_constant = uint[4](rotl(PATTERN[start:step:stop], distance)); // 7.
uint[4] restored = uint[4](rotr(rotl(flags[start:step:stop], distance), distance)); // 9.
uint[4] combined = uint[4](rotr(flags[start:step:stop] ^ PATTERN[start:step:stop], distance)); // 1.
uint[4] shifted = uint[4](rotl(flags[start:step:stop] << 1, distance)); // 4.
flags[0:3] = bit[4](uint[4](rotr(flags[start:step:stop], distance))); // "11011100".
start = 0;
step = 1;
stop = 3;
uint[4] fresh = uint[4](rotl(flags[start:step:stop], distance)); // 9; left is still 3.
```

Each register-operand occurrence is evaluated once per cast. Runtime packing
uses existing QCX UINT shifts and OR; it does not shift signed INT values. Unpacking snapshots the
complete UINT before extracting bits, masks each extracted value to one bit,
and only then converts it to INT storage. All result bits are available before
any destination write, preserving original values in overlapping assignments.
Temporary ownership and expression state are restored after evaluation and
failed lowering; output budgets are checked before returning a runtime result.
No new bra instruction or build option is required.

Supported branches and loops take fresh values on each visit. Skipped paths,
logical short-circuiting, and empty loops suppress operand domain checks and
arithmetic, while names, types, widths, visibility, and static selections remain
validated. A constant cast requires constant operands; converting mutable
storage or using runtime indices on a constant register does not make it
constant. In a mixed runtime set or range, operand domains (including constant
entries or bounds) are checked only when reached. Skipped paths still validate
names, types, visibility, and the target width; runtime set counts are known
during conversion, while runtime range counts are checked when reached.
Reached operand failures, including bounds, shift-count, and division failures,
occur before destination writes. Signed
integer/register reinterpretations and widths beyond native UINT remain
outside this feature.

### Integer remainder

The `%` operator supports scalar `int` and `uint` operands, including
literal expressions, named constants, variables, and supported explicit integer
casts. It follows the converter's truncation-toward-zero integer division
convention:

```qasm
const int negative = -7 % 3; // -1
const int positive = 7 % -3; // 1
int value = 7 % 3;          // folded to 1 during conversion
int divisor = 3;
value = value % divisor;    // evaluated by QCX at runtime
value %= divisor;           // computes remainder, then assigns back to value
```

The remainder follows `a - trunc(a / b) * b`, so a nonzero signed remainder
has the sign of the dividend. Unsigned remainder is nonnegative after operand
promotion. Constant evaluation uses integer calculations without floating-point
conversion; unsigned widths follow the normalization rules above.

Runtime remainder is lowered to a temporary initialized from the dividend,
followed by native QCX `LET temporary %= divisor`. Both operands remain live
until the complete result is available; no intermediate quotient is evaluated.
Computations remain in their original branches
and short-circuit operands; only temporary declarations may be moved earlier.

`%=` is supported for scalar `int` and `uint` variables with integer
right-hand operands. It uses the same lowering as `%`, followed by assignment
of the completed result back to the target. Self-references such as `a %= a`
or `a %= (a % b) + 1` therefore read the original target throughout evaluation.

Evaluated constant `/` and `%` expressions with zero divisors produce
converter-specific errors, reported by the command-line tool without a Python
traceback. Skipped operands of constant Boolean expressions are still validated
without performing the arithmetic. Runtime division and remainder remain deferred
to QCX; `bra` checks integer zero divisors when the respective instruction executes
and throws `bra::integer_zero_divisor_error`. Remainder diagnostics identify
`LET ... %= ...`, rather than a division instruction. The native minimum signed
integer modulo `-1` produces zero, consistently with constant evaluation, without
overflowing an intermediate quotient. Other signed arithmetic, including division
overflow, remains unchecked; unsigned arithmetic wraps at its supported width.
Literal zero-divisor `/` and `%` operations in runtime
expressions are also deferred so short-circuited operands can skip them.
Non-integer remainder operands are
rejected unless explicitly cast to an integer type first. Indexed
integer-array elements also support `%` and `%=`.

### Scalar bitwise operators

The supported subset provides `&`, `|`, `^`, and unary `~` for scalar `int`,
`uint`, and scalar bit expressions. Binary operands must either
both have integer type or both have scalar bit type. Integer results retain
integer type; bit results retain scalar bit type and remain zero or one.
Mixing bits and integers requires an explicit cast. Boolean operands require
an explicit `bit(...)` or integer cast; Boolean variables are not bitwise
assignment destinations. Floating-point and complex operands require a
supported explicit integer cast. Whole registers and arrays are not scalar
operands, including singleton registers.

Angle operands follow the separate [scalar angle bitwise rules](#scalar-angle-bitwise-operators).

```qasm
OPENQASM 3.0;
const int MASK = 7 & 3;
int a = -3;
int b = 7;
int c = ~(a & b) | (a ^ b);
a &= MASK;
b ^= a;
// c is -6; a is 1; b is 6.
```

Signed integer operations use the backend's native two's-complement `INT`
representation, including negative values. Literal integer operands, including
folded constant expressions, must fit the host-derived QCX `INT` range when
used in signed bitwise operations. Signed complement flips the native-width
bits (`~0` is `-1`); declared `int[n]` widths remain unenforced. Unsigned
operations retain their unsigned width, and complement uses its width's mask
(`~uint[8](0)` is `255`). Mixed integer operands follow the promotion rules
above.
Generated programs assume a backend with the same integer representation as
the converter host. General signed arithmetic overflow checks remain separate work.

Bit complement flips only the single bit: `~bit(false)` is one and
`~bit(true)` is zero. Scalar bits and individually indexed bit-register
elements can be used in these expressions, directly in conditions and logical
expressions, or assigned to scalar bits and Booleans. An integer bitwise result
still requires a comparison or `bool(...)` cast to become a condition.
Supported numeric casts allow bitwise results in arithmetic and gate parameters.

```qasm
OPENQASM 3.0;
bit[3] flags = "101";
array[int, 3] positions = {0, 1, 2};
int i = -1;
bit selected = ~flags[i];
flags[i - 1] |= flags[i];
flags[positions[0]] ^= bit(true);
bool ready = flags[i] & ~flags[0];
// selected is 0; flags is "110"; ready is true; i is still -1.
```

Compound `&=`, `|=`, and `^=` assignments accept scalar `int`,
`uint`, and scalar bit targets, including static and runtime-indexed integer-array
and bit-register elements. Their RHS must have integer type for integer targets
or scalar bit type for bit targets. For example, `flag ^= bit(true)` is accepted,
but `flag ^= 1` requires that explicit bit conversion. Whole-register and static
selection operations are described below. Active loop
iterators remain read-only.

Bitwise binary operands evaluate left to right and are eager: `&` and `|`
do not short-circuit like `&&` and `||`. Only the surrounding logical
operators, branches, and loop transfers can skip their evaluation. Literal
expressions fold during conversion; runtime expressions use private storage
and the QCX `LET &=`, `|=`, and `^=` operators. Integer complement uses XOR
with `-1` for INT or the width mask for UINT, while bit complement uses XOR
with `1`.

Compound assignments evaluate the complete RHS before modifying the target.
Runtime destination indices are captured and bounds-checked before the RHS
and remain reserved until the write; they are not reevaluated. Separate RHS
accesses retain their own captures and checks. Self-references and overlapping
accesses read the original target throughout RHS evaluation. Types, names, and
supported index syntax are validated even in empty loop bodies, without
evaluating value-dependent bounds or arithmetic there. Whole-array bitwise
operations, aliases, and general
fixed-width signed integer semantics remain unsupported.

### Whole-bit-register bitwise operators

Fixed-size `bit[n]` registers support binary `&`, `|`, and `^`, unary `~`,
and compound assignments `&=`, `|=`, and `^=`. Operands may be whole register
names, bit-string literals, or nested register bitwise expressions. Binary
operands must have equal widths, and the result preserves that width.
Complement flips every bit without changing the width.

Register expressions can initialize a top-level `bit[n]` declaration or be
assigned with `=` to a whole register. The destination must have the same
width as the result. Compound assignments also require an equal-width register
RHS. A complete runtime range can also supply that RHS through the
[checked range subset](#runtime-range-rhs-bitwise-assignments) below.
No implicit padding, truncation, or scalar broadcasting is performed.
`bit[1]` remains a register, distinct from scalar `bit`: `one ^ "1"` is valid
for a singleton register, but `one ^ bit(true)` is not. Leading zeroes in a
bit-string literal count toward its width.

```qasm
OPENQASM 3.0;
bit[4] a = "1010";
bit[4] b = "1100";
bit[4] c = a & b;          // "1000"
bit[4] d = ~(a ^ b);       // "1001"
a |= b;                   // "1110"
b ^= b & "0011";          // "1100"
```

Operations use the existing one-`INT`-element-per-bit representation and bra's
scalar bitwise instructions. Registers are not packed into native integers, so
their width is not restricted to native `INT` or `UINT` width. As in existing
register assignments, index zero is the rightmost bit of a literal.
Literal-only expressions fold during conversion; runtime expressions evaluate
left to right into private storage. The complete result is computed before
any destination element is written, including for `a = ~(a & b)` and
`a ^= a & b`. Temporary storage is released and reused after evaluation and
assignment. No new bra instruction is required.

Ordinary and compound register assignments are supported in branches and
supported `for` and `while` bodies, including nested loops and
measurement-controlled execution. Branches and loop transfers can skip their execution;
binary bitwise operators themselves remain eager. Types, names, and widths
are checked even in skipped branches and empty loop bodies. Register sizes
remain tied to declaration scope when iterators shadow size constants, and
active iterators remain read-only.

Register values are supported in `popcount`, `rotl`/`rotr`, register assignments, and the equality
comparisons described below; they are not bare conditions. Explicit
[same-width UINT casts](#casts-between-uint-values-and-bit-registers) interpret
a supported register pattern as an integer, including for indices or loop
bounds. Signed integer/register casts, indexing of computed results, and bare
register expressions as numeric loop bounds remain unsupported. Static selection
operations are described below.
Concatenation and aliases remain outside this subset. Existing direct
bit-register copies and indexed scalar-bit operations retain their behavior.

### Whole-register copies from runtime angle-bit ranges

A named mutable bit register can be initialized or assigned directly from a
runtime range on an explicitly sized angle variable, constant, angle-array
element, or supported angle iterator. Both endpoints are required; an omitted
step is `+1`. The first selected bit becomes bit zero of the destination.
Negative indices are normalized after generating the inclusive progression,
and repeated source positions, including negative aliases, are preserved.

The source angle and array-element address are captured before the range
operands. Start, step, and stop are each captured once. A reached zero step,
empty range, invalid generated position, or count different from the declared
destination width fails before extracting selected bits or writing the
destination. The complete result is snapshotted before the write loop. Each
loop iteration captures fresh operands; skipped paths defer runtime checks while
still validating names, types, and known widths.

Destination widths are not limited to native UINT width. A source angle has at
most native UINT width `W`, but negative and nonnegative positions can refer to
the same bit, so a valid range can contain up to twice the source width. Position
and snapshot storage are bounded independently of larger destination widths;
the generated copy loop does not grow with the destination width. Scalar `bit`
destinations, compound assignments, and unsized angle sources remain outside
this direct whole-register copy subset. Assignments to static bit-register
selections and nested expressions are described below.

<!-- runtime-angle-range-register-copy-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int first = 1;
int last = 4;
bit[4] saved = phase[first:last]; // "1010".
array[angle[8], 2] phases = {phase, phase};
int slot = 1;
bit[4] element;
element = phases[slot][first:last]; // "1010".
phase = angle[8]("00000000");
bit[4] fresh = phase[first:last]; // "0000"; saved is still "1010".
angle[4] small = angle[4]("1011");
first = -4;
last = 3;
bit[8] repeated = small[first:last]; // "10111011".
```

### Whole-register assignments from runtime-range expressions

A top-level `bit[n]` initializer or an ordinary `=` assignment to a named mutable
whole bit register may use a runtime-dependent register expression. The RHS
may contain ranges on named mutable or constant bit registers or explicitly
sized angle variables, constants, array elements, and supported angle iterators;
nested `&`, `|`,
`^`, `~`, `<<`, `>>`, `rotl`, and `rotr` operations, and supported known-width
register operands. Both range bounds are required; an omitted step means `1`.
Signed and unsigned integer bounds and steps retain the existing checked range
rules, including negative positions, nonunit strides, and repeated source bits.
Shifts and rotations retain their existing count types and domain rules.

The declared destination width must fit QCX INT indexing. It is resolved in
declaration scope and does not change when a loop iterator shadows a size
constant. A singleton `bit[1]` is a register; scalar `bit` destinations cannot
receive a register value. Statically known invalid types, widths, and selection
bounds are rejected during conversion.

The converter evaluates the RHS left to right into private bit snapshots.
Each nested binary register operation checks equal operand lengths after its
own children complete. After the complete RHS and its runtime checks finish,
QCX `ASSERT` requires the final length to equal the declared destination width.
Only then does a loop write the destination bits in order. There is no padding,
truncation, or scalar broadcasting. The destination width never determines a
dynamic leaf's selected length or skips a later bound or count expression.
A reached failure before copying leaves every destination bit untouched,
including when a source overlaps the destination.

For an angle-range leaf, the source value and any array-element address are
captured before its bounds, and selected bits are snapshotted before the next
operand is evaluated. Each leaf has its own storage capacity, independent of
the destination width. Unlike a direct angle-range copy, the final length is
checked only after the complete expression has been evaluated. Expressions
containing angle ranges also support static selections, runtime index sets, and
runtime-range destinations as described below. Bitwise compounds on whole
registers, static selections, runtime index sets, and runtime ranges also accept
these expressions. `popcount` accepts them as described above; other expression
contexts remain outside this angle-range subset.

<!-- runtime-angle-range-expression-assignment-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int first = 1;
int last = 4;
bit[4] mask = "1101";
bit[4] inverted = ~phase[first:last];       // "0101"
bit[4] result = phase[first:last] ^ mask;  // "0111"
result = rotl(phase[first:last] ^ mask, 1); // "1110"
```

Assignments work in supported branches and loops; initializers remain top-level.
An unreached assignment performs no runtime reads, checks, or writes. Names,
types, known widths, and wholly static selection bounds are still validated on
skipped paths and in empty loop bodies. Logical short-circuiting inside scalar
bounds or distances is preserved. Loop reentry captures current bounds, counts,
and source bits, rather than reusing a previous result.

<!-- runtime-range-expression-assignment-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[4] B = "1101";
int first = 0;
int last = 3;
bit[4] initialized = (a[first:last] & B) ^ "0011"; // "1010"
bit[4] rotated = rotl(a[first:last], 1);           // "0111"
bit[4] result = "0000";
result = ~a[first:last];                         // "0100"
a = rotl(a[first:last], int(a[first:last] != B)); // "0111"; original a is read first
result = (a[first:last] | B) ^ "0011";           // "1100"
if (false) { result = ~a[first:0:last]; }         // No runtime step check or write.
int visits = 0;
while (visits < 2) {
    result = ~result[first:last];
    visits += 1;
}
bit[1] single = ~a[first:first];                  // "0"; result is still "1100"
bool preserved = B == "1101";                    // true
```

The path reuses the snapshots used by runtime-range `popcount` and equality.
Widths may exceed native UINT width. Snapshot storage and generated output
remain bounded by the existing limits; the final copy uses a loop rather than
one emitted instruction per destination bit. Temporary declarations are hoisted
outside branches and loops, and cleanup preserves caller-owned storage on
success and failure. No new bra instruction or build option is required.

Existing [direct runtime-range copies](#copies-from-runtime-bit-register-ranges)
retain their separate capture/check timing, and fixed-width assignments retain
their existing paths. Static selections also accept the
[expression assignment subset](#assigning-runtime-range-expressions-to-static-selections)
below, as do [runtime index-set destinations](#assigning-runtime-range-expressions-to-runtime-index-sets)
and [runtime-range destinations](#assigning-runtime-range-expressions-to-runtime-ranges).
Whole registers, static selections, runtime index sets, and runtime ranges also
support these angle-range expressions in bitwise compound assignments. These destinations support
[bitwise compounds with runtime-range expressions](#bitwise-compound-assignments-with-runtime-range-expressions).
Shift and arithmetic compound extensions remain separate work. Runtime-dependent `const`
initializers, declaration widths, general block-local declarations, runtime
angle-bit ranges in other contexts, aliases, concatenation, and computed-result indexing remain
outside this subset.

### Whole-bit-register shifts

`<<` and `>>` shift fixed-size `bit[n]` register values, including bit-string
literals and nested register bitwise/shift expressions. Results preserve the
left operand's width and can initialize a top-level register declaration or
be assigned to an equal-width whole register. Whole registers also support
`<<=` and `>>=`. Singleton `bit[1]` values remain registers, distinct from
scalar `bit` values.

The converter uses the following register-shift policy:

- Left shifts move bits toward higher indices; right shifts move them toward
  lower indices. Index zero is the rightmost bit of a literal.
- Vacated positions are filled with zeroes, and shifted-out bits are discarded.
- A zero count preserves the value. A count at least as large as the register
  width produces an all-zero result.
- Counts may be scalar `uint` or nonnegative `int` expressions. Negative signed
  counts are rejected, not converted to unsigned values. Boolean and scalar
  bit counts require an explicit integer cast; floating-point and complex
  counts likewise require a supported integer cast.

This oversized-count policy differs from scalar integer shifts, which require
the count to be below the shifted integer's width. Scalar-integer semantics
are unchanged. Register widths may exceed native integer width, and counts
such as native `UINT_MAX` safely produce zero rather than causing an invalid
native shift or an out-of-range signed conversion.

```qasm
OPENQASM 3.0;
bit[4] a = "1011";
uint count = 1;
bit[4] b = a << count;       // "0110"
bit[4] c = a >> count;       // "0101"
a <<= int(a[0]);            // "0110": count reads the original low bit
a >>= 4;                    // "0000"
```

Literal-only expressions fold during conversion; constant counts use direct
element assignments. Runtime counts are evaluated and captured once. A signed
count is checked with `ASSERT count >= 0`; negative literal counts in runtime
expressions also emit a check so unreachable code can skip them. An executed
failure throws `bra::assertion_error`, currently uncaught by bra's CLI.

Runtime lowering uses private source/result bit arrays and a bounded copy
loop. Oversized counts reach the zero-result path before any count-to-`INT`
conversion or index arithmetic. Runtime register shifts require a width at
most the host-derived QCX `INT_MAX` because indices use signed native storage.
The complete result is computed before any destination element is written,
preserving self-references such as `a <<= int(a[0])`. Buffer storage is reused
only after its result is consumed. No new bra instruction is required.

These operations support branches and supported loops, including transfers,
measurement-controlled execution, and nested expressions. Runtime checks and
copy loops stay on reached paths. Empty loop bodies validate types, names,
and index syntax without evaluating count or index values; statically invalid
indices in converted bodies retain existing conversion-time errors. Active
iterators remain read-only, and bit-register iteration retains live element
reads when a shift modifies the iterated register.

Shifts on scalar bits and computed-result indexing remain unsupported. Static
selection shifts are described below; rotations have their own rules above.
This does not add aliases, concatenation, register
casts, or register-valued scalar
conditions/loop bounds. General count-expression overflow checks remain
separate work.

### Static bit-register selections

Constant index ranges and discrete index sets can be register-valued operands
in `&`, `|`, `^`, `~`, `<<`, and `>>` expressions. They can also be destinations
of ordinary `=` and compound `&=`, `|=`, `^=`, `<<=`, and `>>=` assignments.
Selected values may be combined with whole registers, other selections, and
bit-string literals, including in nested expressions and top-level initializers.
No new bra instruction is required.

Selection order is significant: ranges follow their specified progression,
and discrete sets retain their listed order. The first selected element is
position zero (the low bit) of the selected value. Negative indices are
normalized against the original container size after generating the range or
set. Descending ranges and nonunit steps are supported; range endpoints are
inclusive when reached by the step. For example, `a[3:-1:0]` reverses the
element order, and `a[{3, 0}]` has `a[3]` as its low bit and `a[0]` as its high bit.

The selected width is the number of selected elements, not the container width.
Binary bitwise operands and assignment destinations require matching widths;
there is no padding, truncation, or scalar broadcasting. `a[0]` is a scalar
bit, whereas `a[0:0]` and `a[{0}]` are one-bit register selections. These views
also work for singleton storage. Shift counts follow the register policy above:
counts at least the selected width produce zero, and negative signed counts
fail only when their emitted check executes.

```qasm
OPENQASM 3.0;
bit[4] a = "1010";
bit[2] b = "11";
bit[2] c = a[0:1] ^ b;     // "01"
a[1:2] = ~a[0:1];          // overlapping source/destination; a stays "1010"
a[{0, 2}] &= b;            // a stays "1010"
a[0:1] <<= 1;              // a becomes "1000"
```

The complete RHS is evaluated before any destination write, including direct
copies such as `a[1:2] = a[0:1]` and `a = a[3:-1:0]`. Overlapping direct copies
snapshot their sources; computed expressions use existing private result
storage. Untouched container elements are preserved, and temporary storage is
released and reused after assignment.

The converter preserves repeated source indices in order, including entries
that become equal after negative-index normalization. Ordinary and compound
assignment destinations must not repeat a normalized index: `a[{3, -1}]` is
therefore rejected as a destination for a four-bit register. This is an explicit
converter subset policy; it does not change measurement selection handling.

Range bounds, steps, and set elements must be literals or supported expressions
of previously declared constants. Runtime variables and active loop iterators
are not allowed in these selections, even when an iterator is unrolled or
shadows a constant. Runtime scalar-element indexing remains supported separately.
Empty selections, zero steps, omitted bounds, noninteger indices, and out-of-range
elements are rejected during conversion. Invalid large ranges are checked
incrementally rather than fully allocated before a bounds failure.

Supported branches and loops may contain these assignments. Widths, types,
selection bounds, and repeated destinations are checked on skipped paths and
in empty loop bodies; runtime shift-count arithmetic and checks stay on reached
paths. Active iterators remain read-only. Static selections also accept
[a complete runtime range RHS for bitwise compound assignments](#runtime-range-rhs-bitwise-assignments).
Runtime-range expressions outside the supported scalar consumers,
[whole-register assignments](#whole-register-assignments-from-runtime-range-expressions),
[static-selection assignments](#assigning-runtime-range-expressions-to-static-selections),
[runtime index-set assignments](#assigning-runtime-range-expressions-to-runtime-index-sets), and
[runtime-range destination assignments](#assigning-runtime-range-expressions-to-runtime-ranges), multidimensional
indexing, indexing computed register results, aliases, concatenation, register
casts remain separate work. Static selections also
support [iteration with scalar bit iterators](#static-selection-sources).
Runtime singleton and multi-entry sets have separate
[loop-source subsets](#runtime-singleton-selection-sources).
Runtime ranges also support [scalar-bit iteration](#runtime-dependent-bit-register-range-loops).
Their [plain and bitwise compound assignments](#runtime-bit-register-range-assignments)
and [copies into registers or selections](#copies-from-runtime-bit-register-ranges) are described below.

#### Assigning runtime-range expressions to static selections

An ordinary `=` assignment to a static range or discrete index set on a named
mutable `bit[n]` register may use the runtime-range RHS expression subset
described for [whole-register assignments](#whole-register-assignments-from-runtime-range-expressions).
This includes nested bitwise, complement, shift, and rotation expressions
mixing ranges on mutable or constant bit registers with supported known-width
register operands. Direct and nested runtime ranges on explicitly sized angle
variables, constants, array elements, and supported angle iterators are also
accepted, including mixed angle and bit-register range expressions.

The destination selection is resolved and validated before RHS evaluation.
Its positions must be supported constant expressions; runtime variables and
active loop iterators are not allowed in the destination bounds or set entries.
Existing rejection of empty selections, zero steps, omitted bounds, invalid
indices, read-only targets, and duplicate normalized destinations is retained.
A singleton selection such as `a[0:0]` remains a register, distinct from `a[0]`.
Known invalid types, widths, and static selections are rejected during conversion,
including in skipped branches and empty loop bodies.

The complete RHS is evaluated left to right into private snapshots. Nested
binary operations check operand lengths after their children finish. The final
RHS length must then equal the destination selection width, enforced by QCX
`ASSERT` when runtime-dependent. The selection width never determines a dynamic
source leaf's length, and no padding, truncation, or scalar broadcasting occurs.
Only after successful RHS evaluation and length validation are selected bits
written: RHS bit zero goes to the first selected position, bit one to the next,
and so on. Negative indices are normalized without reordering the selection.
All writes read the completed snapshot, preserving original source values during
self-overlap and leaving every unselected destination bit unchanged.
Angle sources and array-element addresses are captured before their bounds;
each leaf is snapshotted independently of the destination width.

<!-- angle-range-static-selection-assignment-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int first = 1;
int last = 4;
bit[8] result = "01010101";
result[3:-1:0] = ~phase[first:last]; // "01011010"
bit[8] copied = "00000000";
copied[0:3] = phase[first:last];   // "00001010"
```

<!-- static-selection-expression-assignment-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[4] B = "1101";
bit[8] result = "01010101";
int first = 0;
int last = 3;
result[7:-2:1] = rotl(a[first:last], 1);         // "11111101"
bool range_ok = result == "11111101";
result[{7, 1, 5, 0}] = (a[first:last] & B) ^ "0011"; // "01011111"
bool set_ok = result == "01011111";
a[3:-1:0] = ~a[first:last];                     // "0010"; original a is read first
if (false) { result[0:3] = ~a[first:0:last]; }   // No runtime step check or write.
bit[1] one = "1";
one[{-1}] = ~a[first:first];                    // "1"; singleton storage works
bool untouched = result[{6, 4, 2}] == "111";    // true
```

Supported branches and loops retain their usual execution behavior. An
unreached assignment performs no runtime reads, checks, or writes; logical
short-circuiting within scalar RHS bounds and distances is preserved. Each
reached execution captures current source values and counts. A reached RHS
error or length mismatch causes no selected destination writes, including on a
later loop iteration after earlier assignments succeeded.

Destination expansion is bounded by QCX INT indexing and the existing runtime
selection storage limit. RHS snapshots use the shared bounded storage. Static
destinations emit one final write per selected position, with the output budget
checked before the entire write sequence. Temporary declarations are hoisted
outside branches and loops, and cleanup preserves caller-owned state and storage
on success and failure. Widths may exceed native UINT; no new bra instruction
or build option is required.

Existing [direct runtime-range copies](#copies-from-runtime-bit-register-ranges)
retain their capture/check timing, and fixed-width selected assignments retain
their existing paths. [Runtime index-set destinations](#assigning-runtime-range-expressions-to-runtime-index-sets)
and [runtime-range destinations](#assigning-runtime-range-expressions-to-runtime-ranges)
also accept this RHS subset. Static selections, runtime index sets, and runtime ranges
support these angle-range RHS expressions in bitwise compound assignments. These destinations support
[bitwise compound assignments](#bitwise-compound-assignments-with-runtime-range-expressions).
Existing direct-copy and fixed-width subsets remain available. Scalar
elements, angle-bit destinations, aliases, concatenation, computed-result
indexing, and general block-local declarations remain outside this subset.

### Runtime bit-register index-set expressions

Named mutable and constant `bit[N]` registers support runtime-dependent
discrete-set selections as register-valued expression operands:

```qasm
bit[2] selected = flags[{i, j}];
bit[2] result = flags[{i, j}] ^ "11";
uint ones = popcount(C[{i, -1, i}]);
```

The result width is the number of entries, including repetitions. The first
entry supplies the low bit of the selected value. Negative indices normalize
against the source width, preserving order and aliases. Selected results can
be wider than their sources. A singleton set `flags[{i}]` produces `bit[1]`,
while ordinary `flags[i]` remains scalar `bit`. These forms follow the
specification's [register slicing rules](https://openqasm.com/language/types.html#register-concatenation-and-slicing).

Each entry must independently have supported INT/UINT type. Entries may mix
constants, runtime variables, enclosing iterators, integer-array elements, and
supported integer-valued functions. Boolean, floating-point, and scalar-bit
values require a supported explicit integer conversion first; register operands
are invalid as indices.
Sets must be nonempty. Source widths and result widths must fit QCX INT indexing.

Every access resolves the source and captures each index occurrence once in
source order. All indices are normalized and bounds-checked before reading any
selected bit. Signed indices must be in `[-N, N - 1]`, and UINT indices must be
below `N` and pass checked conversion to QCX INT. When any entry depends on
runtime storage or an enclosing iterator, all entries remain on the reached
path, including literal and constant entries. Entirely constant sets retain
their existing static validation and lowering.

The complete selected value is copied into private bit storage before its
consumer operates or any assignment destination is written. Each explicit
access performs its own captures and reads. Subsequent source/index changes
cannot alter an earlier result. Overlapping assignments such as
`flags[0:1] = flags[{i, j}]` therefore use the original RHS bits. This expression
snapshot behavior differs from selection loops, which capture positions once
and read source bits live on each visit.

Supported consumers are matching-width register initialization and assignment,
`~`, `&`, `|`, `^`, shifts, `==`/`!=`, `popcount`, `rotl`/`rotr`, and supported
same-width angle casts. Whole registers, static selections, supported angle
selections, and bit-string literals may be combined with these operands.
Existing static and supported runtime index-set destinations and compound
assignments may consume the results. Widths must match: there is no implicit padding, truncation, scalar
broadcasting, or conversion from a selected register to a Boolean or integer.
A runtime-dependent selection cannot initialize a constant register, even when
its source register is constant.

Runtime arithmetic and bounds checks execute only on reached paths, including
under short-circuit conditions. Skipped bodies and empty enclosing loops still
validate names, widths, types, and supported syntax without fabricating iterator
values. Source and index names must be declared before use; active iterators
shadow outer declarations and remain read-only.

<!-- runtime-bit-set-expression-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit[4] flags = "1010";
int i = -1;
uint j = 1;
bit[3] saved = flags[{i, j, i}]; // "111", including the repeated high bit.
flags[0:2] = flags[{i, 0, j}];   // RHS is captured first; flags becomes "1101".
i = 0;
bit[3] after = flags[{i, 0, j}]; // "011"; saved is still "111".
const bit[65] C = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001";
int high = -1;
bit[2] pair = C[{high, 0}];      // "11", without truncating the wide source.
uint ones = popcount(C[{high, 0}]); // 2.
angle[2] phase = angle[2](pair); // Exact stored pattern "11".
h q;
if (saved == "111" && after == "011" && pair == "11") { p(pi / 4) q; }
```

Mutable registers are read through their existing element storage; `bit[1]`
retains scalar QCX storage while checking each index. Constant sources use a
private native UINT-word table prepared once per access and reused across all
selected bits. Full and partial words are preserved without truncating wide
sources. Generated capture, lookup, and snapshot code is checked against the
QCX output limit. Temporaries remain reserved throughout the consumer and are
released on success and failure without releasing caller- or consumer-owned
results. No new bra instruction or build option is required.

Runtime ranges support [plain and bitwise compound assignments](#runtime-bit-register-range-assignments).
They also support [copies into registers or selections](#copies-from-runtime-bit-register-ranges),
where the destination supplies the required width. Explicit
[UINT casts](#casts-between-uint-values-and-bit-registers) also accept runtime
ranges directly or in bitwise/complement/shift expressions, with the target
UINT width supplying the required count for every register operand.
Variable-length runtime ranges also support `popcount`,
[comparisons](#equality-with-runtime-bit-register-and-angle-bit-ranges),
[whole-register assignments and initializers](#whole-register-assignments-from-runtime-range-expressions),
[static-selection assignments](#assigning-runtime-range-expressions-to-static-selections),
[runtime index-set assignments](#assigning-runtime-range-expressions-to-runtime-index-sets), and
[runtime-range destination assignments](#assigning-runtime-range-expressions-to-runtime-ranges).
Other runtime range expressions with variable result widths, computed register sources,
aliases, concatenation,
and further indexing of selection results remain
unsupported. Runtime ranges also retain their separate
[loop-source subset](#runtime-dependent-bit-register-range-loops).

### Copies from runtime bit-register ranges

Plain `=` assignment can copy a runtime-dependent range from a named mutable or
constant `bit[N]` register into a whole mutable `bit[k]` register or a static or
runtime range/discrete-set selection on a named mutable bit register:

```qasm
result = flags[start:stop];
result = C[start:step:stop];
result[0:2] = flags[start:step:stop];
result[5:-2:1] = flags[start:step:stop];
result[{5, 1, 3}] = flags[start:step:stop];
result[{i, j, k}] = flags[start:step:stop];
result[{7, index, 0}] = C[start:stop];
result[{index}] = flags[first:first];
result[dstart:dstop] = flags[sstart:sstop];
result[dstart:dstep:dstop] = C[sstart:sstep:sstop];
```

The complete RHS must be the range selection. Whole registers, static selections,
and runtime index sets supply a required entry count `k` known during conversion.
For runtime range destinations, both selected counts are determined during
execution and must match, as described below. A range is
runtime-dependent when a bound or step
uses runtime storage or an enclosing iterator. Start, step, and stop may
independently use supported INT/UINT expressions; both bounds are required and
the default step is one. Descending ranges, nonunit steps, negative indices,
and full-width native UINT operands are supported.

After capturing and checking any runtime destination indices, the converter
captures source start, step, and stop once in that order. It constructs
the inclusive raw progression before normalizing negative positions, so
`1:2:6` selects `1, 3, 5`. Every selected position must be in `[-N, N - 1]`;
an unselected stop bound is not an array access. Zero steps, empty selections,
invalid selected positions, and counts different from `k` fail before the first
selected source read or destination write. The runtime count must match exactly,
without padding, truncation, or scalar broadcasting. A one-entry range requires
a `bit[1]` register or register-selection destination, rather than scalar `bit`,
`bool`, or `int` storage. For example, `result[4:4]` and `result[{4}]` are suitable
one-bit register destinations, while scalar `result[4]` is rejected.

Destination ranges and discrete sets follow their declared selection order:
source entry zero maps to the first selected destination, entry one to the
second, and so on. Descending/nonunit ranges, reordered sets, and negative
destination indices are supported. Both ranges and discrete sets may use
supported runtime INT/UINT expressions. Negative indices are normalized against the containing register
width. Entirely static destinations reject empty selections, invalid indices,
and duplicate normalized positions during conversion, including in skipped bodies.
Rejecting duplicate destinations is converter policy; the specification does
not explicitly resolve conflicting writes through repeated indices. Unselected
destination bits are preserved. Active integer loop iterators may supply runtime
range bounds/steps or set entries, including when they shadow constants or their
loops are unrolled. Iterator-dependent selections remain runtime selections;
the iterators themselves remain read-only.

When any destination set entry depends on runtime storage or an enclosing
iterator, every entry is retained until execution, including constant entries.
The fixed entry count supplies `k` without evaluating index values. The converter
uses this execution order:

1. Capture and bounds-check each destination entry once, in written order.
   Signed negative indices are normalized against the containing register width;
   UINT indices also pass checked conversion to QCX INT.
2. Reject duplicate normalized destination positions.
3. Capture source start, step, and stop once, and validate the complete source
   range and its count against `k`.
4. Snapshot all selected source bits.
5. Write the snapshot to the captured destination positions in selection order.

Thus, destination failures occur before source-bound evaluation, even if the
source would also fail, and no failed capture, check, or source read commits a
partial destination update. Destination expressions that read the destination
register use its original values, and their indices are not re-evaluated during
writes. The exact capture/check ordering is converter policy, consistent with
existing [runtime index-set assignments](#runtime-bit-register-index-set-assignments).
The specification's [evaluation-order section](https://openqasm.com/language/classical.html#evaluation-order)
describes precedence and associativity without explicitly settling destination
index evaluation versus RHS evaluation. Complete source snapshots implement
the [assignment-by-copy rule](https://openqasm.com/language/classical.html#generalities).

Repeated normalized source positions are allowed and retain their order. For
example, `C[-1:stop]` with `stop == 3` and a four-bit source reads positions
`3, 0, 1, 2, 3`, requiring a destination register or selection of five bits.
This differs from destinations, whose repeated positions are rejected.
All selected source values are snapshotted before any destination write, so
reversing a register into itself or copying overlapping slices uses original
source bits. Source evaluation or validation failures do not partially
update the destination. Copying into a separate register leaves the source unchanged.

Copies may appear in supported branches and loops, including `break` and
`continue`. Captures, runtime checks, and source reads execute only on reached
paths and refresh on every visit. Names, operand types, source visibility,
destination mutability, and iterator scopes are checked even in skipped bodies
and empty loops without evaluating runtime values. Active iterators remain
read-only and cannot be indexed as bit-register sources. General block-local
declarations remain unsupported.

<!-- runtime-bit-range-copy-example -->
```qasm
OPENQASM 3.0;
bit[8] flags = "10101010";
bit[3] result;
int start = 1;
uint stride = 2;
int stop = 6;
result = flags[start:stride:stop]; // Positions 1, 3, 5; result is "111".
int reverse = -1;
start = 7;
stop = 0;
flags = flags[start:reverse:stop]; // Original bits are reversed: "01010101".
const bit[4] C = "1010";
bit[5] repeated;
start = -1;
stop = 3;
repeated = C[start:stop];          // Positions 3, 0, 1, 2, 3; result is "10101".
bit[1] one;
one = flags[start:start];         // High bit of reversed flags: "0".
```

<!-- static-selection-range-copy-example -->
```qasm
OPENQASM 3.0;
bit[8] flags = "10101010";
bit[8] result = "11001100";
int start = 1;
uint step = 2;
int stop = 6;
result[5:-2:1] = flags[start:step:stop]; // Positions 5, 3, 1; result is "11101110".
const bit[4] C = "1010";
start = -1;
stop = 3;
result[{7, 0, 5, 2, 4}] = C[start:stop]; // Repeated source bit; result is "11111010".
start = 1;
result[0:2] = result[start:stop];        // Overlapping copy; result is "11111101".
bit[1] one = "0";
start = -1;
one[{0}] = result[start:start];         // One-bit register result is "1".
start = 0;
result[7:7] = C[start:start];            // One-bit slice; result is "01111101".
```

<!-- runtime-set-range-copy-example -->
```qasm
OPENQASM 3.0;
bit[8] flags = "10101010";
bit[8] result = "11001100";
int i = 5;
uint j = 3;
int k = -7;
int start = 1;
uint step = 2;
int stop = 6;
result[{i, j, k}] = flags[start:step:stop]; // Positions 5, 3, 1; result is "11101110".
const bit[4] C = "1010";
i = 0;
j = 7;
start = -1;
stop = 0;
result[{i, j}] = C[start:stop];           // Positions 0, 7; result is "01101111".
i = 6;
start = 7;
stop = 6;
int reverse = -1;
result[{i, j}] = result[start:reverse:stop]; // Overlapping copy; result is "10101111".
bit[1] one = "0";
int low = 0;
one[{low}] = result[start:start];          // One-bit register result is "1".
start = -1;
stop = 3;
result[{i, 0, 3, j, 4}] = C[start:stop];   // Repeated source positions; result is "01111110".
```

Runtime range destinations use the same destination-before-source policy, but
capture both selected counts at execution time:

1. Capture destination start, step, and stop once, in that order. Validate its
   selected positions and reject zero steps, empty selections, capacity overflow,
   and duplicate normalized destinations before source-bound evaluation.
2. Capture source start, step, and stop once, in that order. Validate its selected
   positions and nonempty count within capacity, then require its count to equal
   the captured destination count.
3. Snapshot every selected source bit, retaining repeated source positions.
4. Write to the captured destination positions in selection order.

Both ranges use inclusive raw integer progressions before negative-index
normalization, and both require explicit start/stop bounds. The default step is
one. Each operand may independently use a supported INT/UINT expression;
descending/nonunit ranges and full-width native operands retain the existing
range rules. An unselected stop bound need not itself be a valid bit index.
Bounds that read the destination observe its original values and are never
re-evaluated during writing. Unselected bits and completed earlier copies remain
unchanged if a later capture, range check, count check, or source read fails.
Source and destination registers may overlap, including a complete self-reversal.
Captures refresh on every reached visit; skipped paths and empty enclosing loops
still validate names, types, mutability, and scopes without evaluating runtime
values or inventing iterator values.

<!-- runtime-range-to-range-copy-example -->
```qasm
OPENQASM 3.0;
bit[8] flags = "10101010";
bit[8] result = "11001100";
int dstart = 5;
int dstep = -2;
int dstop = 1;
int sstart = 1;
uint sstep = 2;
int sstop = 6;
result[dstart:dstep:dstop] = flags[sstart:sstep:sstop]; // Three bits; result is "11101110".
const bit[4] C = "1010";
dstart = 0;
dstep = 1;
dstop = 4;
sstart = -1;
sstep = 1;
sstop = 3;
result[dstart:dstep:dstop] = C[sstart:sstep:sstop]; // Repeated source bit; result is "11110101".
dstart = 7;
dstep = -1;
dstop = 0;
sstart = 0;
sstop = 7;
result[dstart:dstep:dstop] = result[sstart:sstep:sstop]; // Self-reversal: "10101111".
bit[1] one = "0";
dstart = -1;
dstep = 1;
dstop = -1;
sstart = -1;
sstop = -1;
one[dstart:dstop] = result[sstart:sstop]; // One-bit register result is "1".
```

Mutable sources use their existing QCX element storage, including scalar
storage for `bit[1]`. Singleton destinations retain the containing register's
storage: `one[0:0]` on `bit[1] one` uses scalar storage, whereas `result[4:4]`
on a larger register writes its selected element. Constant sources use the
private native UINT-word
lookup, preserving full and partial words beyond native UINT width. With a
known destination count `k`, private source-position and snapshot tables each
have exactly `k` entries. Reached selections
longer than `k` fail before exceeding that capacity, and shorter selections fail
the count check. Source and containing destination widths must fit QCX INT;
`k` must also
fit the runtime range-position storage limit (currently 1000000). Copies do not
require distinct source indices or a `2*N` position table. Static destination
expansion is bounded before materializing the position list. Source reads and
whole-register writes use runtime loops with code size independent of `k`.
Static selections and runtime index sets emit one write per position. Runtime
sets keep their
captured INT addresses reserved throughout the copy, with `k*(k-1)/2` pairwise
duplicate checks rather than a native-word bitmap. This supports destinations
beyond native UINT width.

When both ranges are runtime-dependent, three private INT buffers hold destination
positions, source positions, and source snapshots. Each has capacity
`min(D, L)`, where `D` is the containing destination register width and `L` is
the existing runtime range-position limit (currently 1000000). A successful
destination has distinct positions and therefore at most `D` entries, even when
the source contains negative aliases or has a different containing width.
Selections exceeding capacity fail on reached paths before any destination
write. Short copies on registers wider than `L` remain supported. Both containing
widths and counts must fit QCX INT; neither `2*D` nor `2*N` must fit INT to use
this copy path. Only the captured entry count is read or written, so shrinking
selections cannot use stale buffer entries. Duplicate destination checks reuse
the range-period/GCD helper without a quadratic scan or re-evaluating the step.
Capture, snapshot, and write loops keep generated copy code independent of the
buffer capacity. Both position tables remain reserved until writing completes.

Constant lookup initialization grows with its native
word count. Conversion checks the output budget before destination capture,
before duplicate checks and source capture, and before any destination write,
including all writes to selected destinations. Temporary
declarations are hoisted outside
loops; storage is released on success and failed lowering while preserving
caller-owned temporaries. No new bra instruction or build option is required.

Outside the supported [whole-register assignment](#whole-register-assignments-from-runtime-range-expressions),
[static-selection assignment](#assigning-runtime-range-expressions-to-static-selections),
[runtime index-set assignment](#assigning-runtime-range-expressions-to-runtime-index-sets),
[runtime-range destination assignment](#assigning-runtime-range-expressions-to-runtime-ranges), UINT-cast,
[ANGLE-cast](#angle-casts-from-runtime-bit-register-ranges),
[population-count](#population-counts-of-runtime-bit-register-ranges), and
[comparison contexts](#equality-with-runtime-bit-register-and-angle-bit-ranges),
runtime ranges remain unsupported in general bitwise/shift expressions,
comparisons, and function calls. Supported scalar conversions and counts can
be used in declaration initializers and scalar assignments.
Explicit [UINT casts](#casts-between-uint-values-and-bit-registers) provide a
separate scalar conversion path for ranges in bitwise/complement/shift
expressions. Computed sources, aliases, and further indexing of selection
results remain unsupported.
Bitwise compound assignments with runtime range RHS values have the separate
destination subsets below. Existing static range and fixed-width runtime set
expressions retain their separate supported forms.

### Runtime range RHS bitwise assignments

Whole mutable `bit[k]` registers and static or runtime range/discrete-set
selections on named mutable bit registers support `&=`, `|=`, and `^=` with a
complete runtime range RHS on a named mutable or constant bit register:

```qasm
result &= flags[start:stop];
result |= C[start:step:stop];
result[7:-2:1] ^= flags[start:step:stop];
result[{5, 1, 3}] &= C[start:stop];
result[{i, j}] ^= flags[start:stop];
result[{7, index, 0}] |= C[start:step:stop];
result[dstart:dstop] &= flags[sstart:sstop];
result[dstart:dstep:dstop] ^= C[sstart:sstep:sstop];
```

Whole registers and static selections have a destination entry count `k` known
during conversion. Runtime index sets also have a known entry count, although
their positions are captured during execution. For runtime range destinations,
`k` is captured during execution. The RHS must select exactly `k` entries when
reached, following the range rules in
[Copies from runtime bit-register ranges](#copies-from-runtime-bit-register-ranges):
explicit endpoints, an optional step defaulting to one, supported INT/UINT
expressions, inclusive raw progression, and normalization of negative indices.
Descending and nonunit strides and repeated normalized source positions are
supported. The first source entry operates on the first selected destination;
discrete sets retain their written order. Empty or invalid static destinations
and duplicate normalized positions are rejected during conversion. Runtime
destinations are captured and validated only when reached: every selected
position must be in range, the selection must be nonempty, and normalized
positions must be distinct. Even constant entries of a mixed runtime set are
checked at execution. Repeated normalized source positions remain allowed.
Bits outside a selected destination are preserved.

For a direct runtime range RHS, the converter uses this execution order:

1. For runtime destinations, capture and bounds-check every set entry once in
   written order, or capture range start, step, and stop once in that order.
   Normalize negative indices and reject duplicate destination positions.
2. Snapshot the original destination bits in selection order, using the captured
   positions without reevaluating destination expressions.
3. Capture source start, step, and stop once, in that order. Check the complete
   range and require its count to equal `k` before reading selected source bits.
4. Snapshot all selected source bits, preserving repeated entries.
5. Compute every result bit privately as original destination bit `&`, `|`, or
   `^` corresponding source bit.
6. Write the complete result to the destination in selection order.

This preserves original values for overlapping self-references. Zero steps,
empty selections, invalid positions, count mismatches, and bound-evaluation
failures leave the destination unchanged. Destination failures precede any
original-destination read or RHS evaluation. RHS validation failures may follow
the original-value snapshot, but cannot commit any write. Captured destination
addresses stay reserved throughout RHS evaluation and are reused for writing.
This capture/check/snapshot ordering and rejection of repeated destinations are
converter policies, consistent with the existing compound-assignment paths.
The OpenQASM
[classical bit-register operators](https://openqasm.com/language/classical.html#classical-bits-and-registers)
require matching widths; they do not explicitly prescribe this snapshot order.

A one-entry selection remains a `bit[1]` register: whole `bit[1]` variables,
`result[4:4]`, `result[{4}]`, `result[index:index]`, and `result[{index}]` are
valid destinations, while scalar `bit` and `result[4]` are not. Both scalar
singleton storage and selections in larger registers retain their containing
storage layout.

Supported branches and loops, including `break` and `continue`, may contain
these assignments. Each reached execution takes fresh snapshots and captures;
skipped statements do not read the destination or evaluate RHS bounds. Names,
types, mutability, visibility, static selection validity, and iterator scopes
are still checked in skipped bodies and empty loops without evaluating runtime
values. Active INT/UINT iterators may supply destination indices or either
range's bounds or steps, including when an iterator is unrolled or shadows a
constant. Such selections remain runtime selections; iterator bindings remain
read-only. Live bit iteration observes completed writes to the iterated register.

<!-- runtime-range-bitwise-rhs-example -->
```qasm
OPENQASM 3.0;
const bit[8] C = "10110010";
bit[4] result = "1010";
bit[8] flags = "11001100";
int start = 1;
int step = 2;
int stop = 7;
result &= C[start:step:stop];             // Selected RHS is "1101"; result is "1000".
flags[{7, 1, 5, 3}] |= C[start:step:stop]; // flags is "11101100"; other bits are preserved.
start = 7;
step = -2;
stop = 1;
result ^= C[start:step:stop];             // Selected RHS is "1011"; result is "0011".
bit[1] one = "1";
int index = -1;
one[0:0] &= result[index:index];          // one is "0".
if (false) {
  result ^= C[start:0:stop];             // Skipped: the zero step is not evaluated.
}
```

<!-- runtime-selection-range-bitwise-example -->
```qasm
OPENQASM 3.0;
const bit[4] C = "1010";
bit[8] flags = "11001100";
int left = 7;
uint right = 1;
int source_first = 0;
uint source_last = 1;
flags[{left, right}] ^= C[source_first:source_last]; // flags is "11001110".
bool set_result = flags == "11001110";
int dstart = 7;
int dstep = -2;
int dstop = 1;
int sstart = 3;
int sstep = -1;
int sstop = 0;
flags[dstart:dstep:dstop] &= C[sstart:sstep:sstop]; // flags is "11001100".
bool range_result = flags == "11001100";
dstart = 0;
dstep = 1;
dstop = 7;
sstart = 7;
sstop = 0;
flags[dstart:dstep:dstop] |= flags[sstart:sstep:sstop]; // Original bits and their reversal: "11111111".
bit[1] one = "0";
int index = -1;
one[{index}] ^= flags[index:index];              // one is "1".
if (false) {
  flags[{left, left}] &= C[source_first:0:source_last]; // Skipped: no duplicate or zero-step check executes.
}
```

For whole/static destinations and runtime index sets, three private INT buffers
of `k` entries hold source positions, original/result bits, and source snapshots.
Runtime sets additionally retain their captured INT addresses, with
`k*(k-1)/2` duplicate comparisons bounded by the output budget. Runtime range
destinations use four private INT buffers: destination positions, source
positions, original/result bits, and source snapshots. Each has capacity
`min(D, L)`, where `D` is the containing destination width and `L` is the runtime
range-position storage limit (currently 1000000). Short selections on registers
wider than `L` remain supported. Only the captured count is read or written, so
shrinking selections cannot use stale buffer entries. Containing widths and
counts must fit QCX INT. There is no requirement that twice a containing width
fit QCX INT, and source positions need not be distinct. Reached selections exceeding capacity fail
before any destination write. The known `k` for other destination forms must
also fit the runtime range-position storage limit. Constant sources use the
shared native UINT-word lookup, including partial final words.
Whole-register snapshots and writes, source snapshots, and result computation
use runtime loops with code size independent of `k`. Runtime range destination
capture, duplicate validation, original reads, and writes also use bounded loops;
duplicate validation reuses the range-period/GCD helper. Static selections and
runtime index sets emit one original read and one final write per selected bit.
Conversion bounds expansion and checks the output budget before destination
capture and before any destination write. Temporary declarations are hoisted
outside loops, and failed lowering releases private
storage while preserving caller-owned temporaries and expression state.
No new bra instruction or build option is required.

Whole-register `=` assignments and declaration initializers also accept the
[runtime-range expression subset](#whole-register-assignments-from-runtime-range-expressions).
Static range and set destinations support
[ordinary expression assignments](#assigning-runtime-range-expressions-to-static-selections).
Runtime index-set destinations also support
[ordinary expression assignments](#assigning-runtime-range-expressions-to-runtime-index-sets).
Runtime-range destinations support
[ordinary expression assignments](#assigning-runtime-range-expressions-to-runtime-ranges) as well.
Shift and arithmetic compound assignments with a runtime range RHS, scalar
destinations, computed sources, aliases, and further indexing remain unsupported.
Nested runtime-range RHS expressions have the separate bitwise compound subset below.

#### Bitwise compound assignments with runtime-range expressions

Whole mutable bit registers and static or runtime range/discrete-set selections
on named mutable bit registers accept `&=`, `|=`, and `^=` with the RHS expression
subset described for
[whole-register assignments](#whole-register-assignments-from-runtime-range-expressions).
This includes nested `&`, `|`, `^`, `~`, `<<`, `>>`, `rotl`, and `rotr` operations
mixing ranges on mutable or constant bit registers with supported known-width
register values. Known-width destinations have a required entry count; runtime
range destinations capture their count during execution.

Whole mutable bit registers and their static selections, runtime index sets, and
runtime ranges also accept direct or nested runtime ranges on
explicitly sized angle variables, constants, array elements, and supported
angle iterators, including mixed angle and bit-register expressions.

Execution follows this order:

1. Resolve and validate destination positions. Runtime index-set entries are
   captured once in written order; runtime range start, step, and stop are
   captured once in that order. Apply existing bounds, INT/UINT conversion,
   normalization, nonempty-selection, and duplicate-destination rules before
   any read of the original LHS or RHS evaluation.
2. Snapshot the original selected LHS bits in selection order, without
   recapturing destination expressions. Keep both the addresses and original
   values reserved while evaluating the RHS.
3. Evaluate the complete RHS left to right into private snapshots. Each nested
   binary register operation checks equal lengths after both children finish.
4. Require the final RHS length to equal the destination count, using QCX
   `ASSERT`. The destination count does not determine a dynamic source leaf's
   length, shorten evaluation, or permit padding, truncation, or broadcasting.
5. Compute every result bit privately as the original LHS bit `&`, `|`, or `^`
   the corresponding RHS bit, then write the complete result in selection order.

The implicit LHS read uses its original-value snapshot. Explicit RHS references
to the same register still perform their own reads and checks. Overlap and
self-references therefore use original values, and unselected bits remain
unchanged. Destination failures precede original reads and RHS evaluation;
RHS errors and length mismatches can follow original reads but prevent all
result writes. A singleton selection remains a register, including on `bit[1]`,
rather than a scalar bit.

<!-- angle-range-register-compound-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int first = 1;
int last = 4;
bit[4] direct = "1101";
direct &= phase[first:last];          // "1000"
bit[4] nested = "1101";
nested ^= ~phase[first:last];         // "1000"
bit[4] rotated = "1000";
rotated |= rotl(phase[first:last], 1); // "1101"
```

<!-- angle-range-static-compound-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int first = 1;
int last = 4;
bit[8] bits = "11010000";
bits[1:4] ^= ~phase[first:last];            // "11011010"
bit[8] direct = "11010000";
direct[{7, 1, 5, 0}] |= phase[first:last]; // "11010011"
```

<!-- angle-range-runtime-set-compound-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int first = 1;
int last = 4;
int high = 7;
uint low = 1;
bit[8] bits = "11010000";
bits[{high, low, 5, 0}] ^= ~phase[first:last]; // "01110000"
bit[8] direct = "11010000";
direct[{high, low, 5, 0}] |= phase[first:last]; // "11010011"
```

<!-- angle-range-runtime-range-compound-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int first = 1;
int last = 4;
int dest_first = 1;
int dest_last = 4;
bit[8] bits = "11010000";
bits[dest_first:dest_last] ^= ~phase[first:last]; // "11011010"
bit[8] direct = "11010000";
direct[dest_first:dest_last] |= phase[first:last]; // "11010100"
```

<!-- runtime-expression-bitwise-compound-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[4] B = "1101";
bit[4] result = "1111";
int first = 0;
uint last = 3;
result &= ~(a[first:last] & B);                 // "0110"
result |= rotl(a[first:last], 1);               // "0111"
result ^= a[first:last] & B;                    // "1110"
bit[8] flags = "01010101";
flags[7:-2:1] |= rotl(a[first:last], 1);         // "11111101"
flags[{7, 1, 5, 0}] ^= (a[first:last] & B) ^ "0011"; // "11111110"
bool static_ok = flags == "11111110";
int high = -1;
uint low = 1;
flags[{high, low, 5, 0}] &= ~a[first:last];      // "01111100"
bool set_ok = flags == "01111100";
int start = 7;
int step = -2;
int stop = 1;
flags[start:step:stop] ^= rotl(a[first:last], 1); // "11010100"
bool range_ok = flags == "11010100";
a[last:-1:first] ^= rotl(a[first:last], int(a[first:last] != B)); // "0101"
if (false) { flags[{high, -1}] |= ~a[first:0:last]; } // No duplicate or step check.
bit[1] one = "1";
const bit[1] ZERO = "0";
int single = -1;
one[single:single] ^= ~ZERO[first:first];       // "0"
const bit[2] ZEROS = "00";
bit[4] captured = "0000";
captured[{0, int(captured[0]) + 1}] |= ~ZEROS[first:1]; // "0011"; addresses saved first
int visits = 0;
while (visits < 2) {
    result ^= ~a[first:last];
    visits += 1;
}                                             // "1110" restored
bool untouched = flags[{6, 4, 2}] == "111";    // true
```

Assignments work in supported branches and loops, including `break` and
`continue`. Every reached execution captures fresh positions, original LHS
bits, RHS bounds, scalar counts, and source bits. Scalar logical short-circuiting
is preserved. Skipped paths and empty loop bodies validate names, visibility,
types, known widths, and syntax without evaluating runtime-dependent domains.
Entirely static destinations retain conversion-time bounds and duplicate checks.
Constants and active loop iterators remain read-only. Nested array and bit
accesses within indices retain their own validation rules.

Storage for the original LHS is bounded by the known entry count, or by the smaller of
the parent width and the runtime-position limit for runtime-range destinations.
A known entry count exceeding that limit is rejected during conversion;
runtime capacity failures occur only when reached. RHS snapshots retain
independent capacities and limits. Whole-register and
runtime-range original reads and final writes use loops; static selections and
runtime index sets emit one original read and final write per selected bit.
Result computation uses a runtime loop over the snapshots. Output limits are
checked before emitting original reads, combination code, and the complete
final write sequence. Temporary declarations are hoisted outside branches and
loops, and cleanup preserves caller-owned state and storage on success and
failed lowering. Registers may exceed native UINT width; QCX INT indexing
limits still apply. No new bra instruction or build option is required.

Existing compound assignments with a direct runtime range or fixed-width RHS
retain their own capture/check timing. Scalar-bit and angle-bit destinations, shift and
arithmetic compound extensions, aliases, concatenation, computed-result indexing,
and general block-local declarations remain outside this subset. Exact capture,
snapshot, check ordering, duplicate rejection, and resource limits are documented
converter policies; the specification does not explicitly prescribe that ordering.

### Runtime bit-register index-set assignments

Named mutable bit registers support runtime-dependent discrete-set destinations
for `=`, `&=`, `|=`, `^=`, `<<=`, and `>>=`:

```qasm
flags[{i, j}] = "10";
flags[{i, j}] ^= other;
flags[{i, j}] <<= count;
```

The selected width is the number of entries. Its first entry is the low bit,
and `flags[{i}]` remains a `bit[1]` destination, distinct from scalar `flags[i]`.
Plain and bitwise compound assignments require a matching-width register RHS;
there is no scalar broadcasting, padding, or truncation. RHS values may use the
supported whole-register, static/runtime selection, bitwise, shift, rotation,
and angle-to-bit-register expression forms. Plain `=` also accepts the complete
runtime range RHS described in [Copies from runtime bit-register ranges](#copies-from-runtime-bit-register-ranges),
or [nested runtime-range expressions](#assigning-runtime-range-expressions-to-runtime-index-sets).
Bitwise compounds accept both the [direct runtime range RHS subset](#runtime-range-rhs-bitwise-assignments)
and [nested runtime-range expressions](#bitwise-compound-assignments-with-runtime-range-expressions),
checking the final count against the destination entry count at runtime.
Runtime range RHS values for shift compounds remain unsupported. Shift counts follow the existing
register policy: nonnegative INT/UINT values are accepted, negative counts fail
when reached, and counts at least the selection width produce zero. Shifts act
within the selection, preserving every unselected source bit.

Each destination entry must have supported INT/UINT type. The converter captures
every occurrence once, in source order, before evaluating the RHS. All positions
are normalized and checked against `[-N, N - 1]`, with UINT indices additionally
passing checked conversion to QCX INT. Duplicate normalized destinations are
rejected before the RHS: for `bit[4] flags`, `{i, 3}` fails if `i == -1`.
Repeated source positions remain valid for reads and loops. Empty selections,
read-only constants and iterators, and unsupported index types are rejected.

For `=`, the complete RHS is captured before any destination write. For compound
assignments, the original selected bits are first copied after destination
validation and before RHS evaluation. The implicit LHS read uses that copy and
does not recapture indices; explicit RHS accesses retain their own evaluation
and checks. Every result bit is computed and copied before the selected
destinations are updated. Consequently, overlapping swaps and self-references
use original values, and failed destination checks, RHS accesses, or arithmetic
do not partially update the target.

These rules implement register slices as references and assignment by value;
see [register slicing](https://openqasm.com/language/types.html#register-concatenation-and-slicing)
and [classical assignment and register operations](https://openqasm.com/language/classical.html#generalities).
The exact capture/check ordering and rejection of repeated destinations are
documented converter policies, consistent with static-register and runtime-angle
selection assignments. The specification does not explicitly resolve
conflicting writes to repeated destination indices.

When any destination entry is runtime- or iterator-dependent, all position,
duplicate, and arithmetic checks stay on reached paths, including for constant
entries in that selection. Skipped bodies and empty enclosing loops still
validate names, types, widths, and syntax without evaluating indices or inventing
iterator values. Entirely constant destinations retain static validation,
including conversion-time rejection of duplicate indices. Assignments in loops
preserve scoped shadowing and the existing live-read behavior of bit iterators.

<!-- runtime-bit-set-assignment-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit[4] flags = "1010";
int i = 0;
uint j = 3;
flags[{i, j}] = flags[{j, i}]; // Swap original bits: "0011".
flags[{i, j}] ^= "11";        // "1010".
flags[{j, i}] <<= 1;          // Shift the two selected bits: "0011".
uint ones = popcount(flags);  // 2.
bit[65] wide = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000";
int high = -1;
uint low = 0;
wide[{high, low}] = wide[{low, high}]; // High bit cleared; low bit set.
wide[{low, high}] <<= 1;              // Original high-only pattern restored.
h q;
p(angle[4](flags)) q;         // Phase 3*pi/8, from stored pattern "0011".
flags[{i}] = "0";            // Final flags is "0010"; queued gates keep their parameters.
```

The existing QCX element storage supports sources wider than native UINT;
`bit[1]` keeps scalar storage while checking its destination index. For a
selection of `k` entries, duplicate detection emits `k*(k-1)/2` pairwise INT
comparisons. This avoids a native-word bitmap limit. Conversion checks the
comparison count against the QCX output budget before allocating capture storage
and checks the full generated assignment before emitting destination writes.
Source and selection widths must fit QCX INT. All temporary snapshots, captured
indices, and implicit-read bindings are released on success and failed lowering
without releasing caller-owned storage. No new bra instruction or build option
is required.

#### Assigning runtime-range expressions to runtime index sets

An ordinary `=` assignment to a runtime-dependent discrete index set on a named
mutable `bit[N]` register accepts the RHS subset described for
[whole-register assignments](#whole-register-assignments-from-runtime-range-expressions).
This includes ranges on mutable or constant registers, nested `&`, `|`, `^`,
`~`, `<<`, `>>`, `rotl`, and `rotr`, and supported known-width register operands.
Direct and nested runtime ranges on explicitly sized angle variables,
constants, array elements, and supported angle iterators are also accepted,
including mixed angle and bit-register range expressions.
The destination entry count `k` is known during conversion even when its
positions depend on runtime values or active loop iterators.
Nested array and bit accesses within an index retain their own validation rules.

Execution follows these steps:

1. Evaluate each destination entry exactly once, in written order. Each entry
   must have supported INT/UINT type; UINT values pass checked conversion to
   QCX INT. Check each position against `[-N, N - 1]` and normalize negatives.
2. After all entries have been captured, reject duplicate normalized positions.
   Even constant entries in a mixed runtime set retain reached-path checks.
3. Keep the captured addresses reserved while evaluating the complete RHS left
   to right into private snapshots. Nested binary operations check equal
   operand lengths after both children finish.
4. Require the final RHS length to equal `k`, using QCX `ASSERT`, then write
   snapshot bit zero to the first captured destination, bit one to the next,
   and so on. The entry count does not determine a dynamic source leaf's length.

There is no padding, truncation, or scalar broadcasting. Source/destination
overlap and index expressions reading the destination use the original values.
Unselected bits are preserved. Any destination error, RHS error, or final
length mismatch prevents all selected writes for that execution. A singleton
set remains a register, including when its containing register is `bit[1]`.

<!-- angle-range-runtime-set-assignment-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int first = 1;
int last = 4;
int high = 7;
uint low = 1;
bit[8] result = "00000000";
result[{high, low, 5, 0}] = ~phase[first:last]; // "10100000"
bit[8] copied = "00000000";
copied[{high, low, 5, 0}] = phase[first:last];  // "00000011"
```

<!-- runtime-set-expression-assignment-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[4] B = "1101";
bit[8] result = "01010101";
int first = 0;
int last = 3;
int high = -1;
uint low = 1;
result[{high, low, 5, 0}] = rotl(a[first:last], 1); // "11110110"
bool rotated = result == "11110110";
result[{high, low, 5, 0}] = (a[first:last] & B) ^ "0011"; // "01010111"
bool combined = result == "01010111";
a[{last, 2, 1, first}] = ~a[first:last];           // "0010"; original a is read first
if (false) { result[{high, -1}] = ~a[first:0:last]; } // No duplicate or step check.
bit[1] one = "1";
one[{high}] = ~a[first:first];                    // "1"; negative singleton index
const bit[2] ZEROS = "00";
bit[4] captured = "0000";
captured[{0, int(captured[0]) + 1}] = ~ZEROS[first:1]; // "0011"; both addresses saved first
bool untouched = result[{6, 4, 2}] == "111";       // true
```

Assignments work in supported branches and loops. Each reached execution
captures fresh indices, bounds, counts, and source bits. Logical short-circuiting
in scalar indices, bounds, and distances is preserved. Skipped assignments and
empty loop bodies still validate names, visibility, types, known widths, and
syntax without evaluating runtime-dependent selection domains. Entirely static
destinations retain their conversion-time bounds and duplicate checks.

The parent register width must fit QCX INT indexing. The entry count and RHS
snapshots obey existing storage limits. Pairwise duplicate checks produce
`k * (k - 1) / 2` assertions, with generated output bounded before destination
capture, duplicate validation, and the entire final write sequence. Captured
addresses remain reserved through RHS evaluation and writing. Temporary
declarations are hoisted outside branches and loops; cleanup preserves
caller-owned expression state and storage on success and failed lowering.
Widths may exceed native UINT; no new bra instruction or build option is required.

Existing [direct runtime-range copies](#copies-from-runtime-bit-register-ranges)
and fixed-width selected assignments retain their own capture/check timing.
[Runtime-range destinations](#assigning-runtime-range-expressions-to-runtime-ranges)
also accept this subset. For bit-register range RHS expressions, these support
[bitwise compound assignments](#bitwise-compound-assignments-with-runtime-range-expressions).
Scalar elements, angle-bit destinations,
aliases, concatenation, computed-result indexing, and general block-local
declarations remain outside this subset.

Runtime ranges support the assignment subset described next. Aliases,
computed register sources, further indexing of selections, arithmetic compound
operators, and general block-local declarations remain unsupported.

### Runtime bit-register range assignments

Plain `=` and bitwise compound `&=`, `|=`, and `^=` assignments accept
`flags[start:stop]` and `flags[start:step:stop]` destinations on named mutable
`bit[n]` registers when
any bound or step depends on runtime storage or an enclosing iterator.
Each operand may independently use a supported INT/UINT expression. Both bounds
are required; the default step is one. Positive and negative steps, negative
indices, and full-width UINT operands are supported.

Ranges use inclusive bounds and generate their raw integer progression before
normalizing negative indices. The stop bound need not itself be selected:
`1:2:6` selects `1, 3, 5`. Every selected position must be in `[-N, N - 1]` for a
destination register of width `N`. Only selected positions are bounds-checked;
an unselected stop bound is not an array access.

The fixed-width RHS subset described here requires a register width known during
conversion. It may use existing
whole-register, static/runtime index-set, bitwise, shift, rotation, and supported
angle-to-bit-register expressions. The selected destination count must equal
that width at runtime, without padding, truncation, or scalar broadcasting.
A one-entry range remains a `bit[1]` destination: its RHS must be `"1"`, `"0"`,
or another one-bit register, rather than scalar `true` or `1`.

Plain `=` may instead copy a complete runtime range RHS through the
[range-to-range copy subset](#copies-from-runtime-bit-register-ranges) above.
Both selected counts are then captured and checked at runtime, using its bounded
storage policy. Bitwise compounds may also accept a complete runtime range RHS
through the [runtime range RHS bitwise subset](#runtime-range-rhs-bitwise-assignments)
above, using the captured destination count and preserving original values
before RHS evaluation. They also accept
[nested runtime-range expressions](#bitwise-compound-assignments-with-runtime-range-expressions).
The checks below describe the fixed-width RHS path.
Plain `=` additionally accepts
[nested runtime-range expressions](#assigning-runtime-range-expressions-to-runtime-ranges),
with the final length checked only after complete RHS evaluation.

The converter captures start, step, and stop once in that order, before
evaluating the RHS. It rejects zero steps, empty selections, invalid selected
positions, width mismatches, and duplicate normalized destinations before any
RHS evaluation. A nonzero raw step does not prevent negative aliases from
repeating a destination: for `bit[4] flags`, `-1:3` selects bit 3 twice.
All result bits are copied before the first destination write, so overlapping
assignments use original values. Unselected bits are preserved, and failed
destination checks or RHS accesses/arithmetic do not partially update the target.

For bitwise compound assignments, the original selected LHS bits are copied
after destination validation and before RHS evaluation. The implicit LHS read
uses that snapshot without recapturing range bounds or reading the selected bits
again. Explicit RHS accesses retain their own evaluation, captures, and checks,
including accesses to the same destination register. Each operation pairs bits
by selection order and computes all results before writing any destination.

These rules implement the specification's reference slices and assignment by
value; see [register slicing](https://openqasm.com/language/types.html#register-concatenation-and-slicing)
and [classical assignment](https://openqasm.com/language/classical.html#generalities).
Bitwise compounds follow the [equal-sized register operation rules](https://openqasm.com/language/classical.html#classical-bits-and-registers).
Exact capture/check timing and duplicate rejection are documented converter
policies; the specification does not explicitly settle those details.

Runtime checks stay on reached paths. Skipped branches, transfers, and empty
enclosing loops do not evaluate bounds, steps, or RHS arithmetic, but still
validate names, integer operand types, RHS widths, and supported syntax without
inventing iterator values. Bounds are recaptured on each reentry; scoped
shadowing and live bit-register iteration retain their existing behavior.
Constant registers and active iterators remain read-only. Entirely static
destinations retain the existing static-assignment path and validation.

<!-- runtime-bit-range-assignment-example -->
```qasm
OPENQASM 3.0;
bit[8] flags = "00000000";
bit[3] payload = "101";
int start = 1;
uint stride = 2;
int stop = 6;
flags[start:stride:stop] = payload; // Positions 1, 3, 5; flags is "00100010".
start = 5;
stop = 1;
int reverse = -2;
flags[start:reverse:stop] = flags[0:2]; // Original low bits copied; flags is "00001000".
bit[1] one = "0";
int single = -1;
one[single:single] = "1";          // One-bit register with a checked negative index.
```

<!-- runtime-bit-range-bitwise-example -->
```qasm
OPENQASM 3.0;
bit[4] flags = "1010";
int start = 3;
int step = -2;
uint stop = 0;
int index = 0;
flags[start:step:stop] &= "01";             // Positions 3, 1; flags is "1000".
flags[start:step:stop] |= flags[{index, 3}]; // flags is "1010".
flags[start:step:stop] ^= flags[{1, 3}];     // Both sides use original bits; flags is "0000".
bit[1] one = "1";
one[index:index] ^= "1";                   // One-bit register result is "0".
```

Registers wider than native UINT use existing QCX element storage; `bit[1]`
keeps scalar storage while still checking its range. For an RHS width `k`, a
private position table has exactly `k` entries. Reached ranges longer than `k`
fail before exceeding that table, and shorter ranges fail the count check.
Pairwise duplicate detection emits `k*(k-1)/2` INT comparisons. Destination and
RHS widths must fit QCX INT; table storage is also bounded by the existing
runtime range-position limit. Conversion checks the output budget before range
capture, before duplicate checks, and before destination writes. All snapshots,
captured positions, and range temporaries are released on success and failed
lowering while preserving caller-owned storage. No new bra instruction or build
option is required.

Shift compounds are described in
[Runtime bit-register range shift assignments](#runtime-bit-register-range-shift-assignments).
Range measurements are described in
[Runtime range measurement destinations](#runtime-range-measurement-destinations).
Runtime ranges can also be [copied into registers or selections](#copies-from-runtime-bit-register-ranges).
Arithmetic compound assignments into runtime ranges, runtime range expression reads
outside the supported consumers,
aliases, computed destination sources, further
selection indexing, and general block-local declarations remain unsupported.

#### Assigning runtime-range expressions to runtime ranges

An ordinary `=` assignment to a runtime-dependent range on a named mutable
`bit[N]` register accepts the RHS expression subset described for
[whole-register assignments](#whole-register-assignments-from-runtime-range-expressions).
Both destination and source lengths may depend on runtime values. The RHS may
mix ranges on mutable or constant registers with supported known-width register
values, using nested `&`, `|`, `^`, `~`, `<<`, `>>`, `rotl`, and `rotr` operations.
Direct and nested runtime ranges on explicitly sized angle variables,
constants, array elements, and supported angle iterators are also accepted,
including mixed angle and bit-register range expressions.

The converter captures destination start, step, and stop once, in that order,
before evaluating the RHS. Both bounds are required and an omitted step means
one. Existing checked INT/UINT range rules apply, including inclusive bounds,
negative positions, descending/nonunit strides, and unselected stop bounds.
Zero steps, empty selections, invalid selected positions, and duplicate
normalized destinations fail before RHS evaluation. Nested array and bit
accesses within bounds retain their own validation rules.

The captured destination positions and count remain reserved throughout complete
RHS snapshot evaluation. Each nested binary operation checks equal lengths
after both children finish. Only after the complete RHS and its checks finish
does QCX `ASSERT` require its length to equal the captured destination count.
That count never determines a dynamic source leaf's length, shortens its
evaluation, or permits padding, truncation, or scalar broadcasting.

A runtime loop then writes snapshot bit zero to the first captured destination,
bit one to the next, and so on. Overlapping assignments and bounds reading the
destination use original values; unselected bits remain unchanged. A reached
destination error, RHS error, or length mismatch prevents all selected writes
for that execution. A one-entry range remains a register, including on a
containing `bit[1]`, rather than a scalar bit.

<!-- angle-range-runtime-range-assignment-example -->
```qasm
OPENQASM 3.0;
angle[8] phase = angle[8]("10110101");
int first = 1;
int last = 4;
int dest_first = 1;
int dest_last = 4;
bit[8] result = "00000000";
result[dest_first:dest_last] = ~phase[first:last]; // "00001010"
bit[8] copied = "00000000";
copied[dest_first:dest_last] = phase[first:last];  // "00010100"
```

<!-- runtime-range-destination-expression-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[4] B = "1101";
bit[8] result = "01010101";
int first = 0;
uint last = 3;
int start = 7;
int step = -2;
int stop = 1;
result[start:step:stop] = rotl(a[first:last], 1); // "11111101"
bool rotated = result == "11111101";
start = 0;
step = 2;
stop = 6;
result[start:step:stop] = (a[first:last] & B) ^ "0011"; // "11101100"
bool combined = result == "11101100";
a[last:-1:first] = ~a[first:last];              // "0010"; original a is read first
if (false) { result[first:0:last] = ~a[first:0:last]; } // No runtime step check.
bit[1] one = "1";
int single = -1;
one[single:single] = ~a[first:first];           // "1"; checked negative singleton
const bit[2] ZEROS = "00";
bit[4] captured = "0000";
captured[int(captured[0]):int(captured[1]) + 1] = ~ZEROS[first:1]; // "0011"
int visits = 0;
while (visits < 2) {
    result[start:step:stop] = ~result[start:step:stop];
    visits += 1;
}                                             // "11101100" restored
bool untouched = result[{7, 5, 3}] == "111";   // true
```

Assignments work in supported branches and loops. Every reached execution
captures fresh destination bounds, RHS bounds, scalar counts, and source bits.
Logical short-circuiting within scalar expressions is preserved. Unreached
assignments and empty loop bodies still validate names, visibility, types,
known widths, and syntax without evaluating runtime-dependent range domains.
Constant registers and active iterators remain read-only, and wholly static
destinations retain their existing validation.

Destination position storage is bounded by the smaller of the containing
register width and the existing runtime-position limit: distinct destinations
cannot outnumber the register's bits. This capacity does not constrain RHS
leaf storage, which retains its own limits. Reached capacity failures occur
before exceeding private buffers. Destination duplicate validation reuses the
range-period/GCD check. Capture, validation, and final writes use bounded
runtime loops; generated output is checked before destination capture, before
RHS evaluation, and before the complete write sequence. Temporary declarations
are hoisted outside branches and loops, and cleanup preserves caller-owned
state and storage on success and failed lowering. Registers may exceed native
UINT width; QCX INT indexing limits still apply. No new bra instruction or
build option is required.

Existing direct runtime-range copies and fixed-width RHS assignments retain
their own capture/check timing.
[Bitwise compound assignments](#bitwise-compound-assignments-with-runtime-range-expressions)
also accept nested runtime bit-register and angle-range RHS expressions.
Angle-bit destinations, aliases,
concatenation, computed-result indexing, and general block-local declarations
remain outside this subset. Exact capture/check order, duplicate rejection,
and resource limits are documented converter policies.

### Runtime bit-register range shift assignments

Runtime-dependent ranges on named mutable bit registers also accept `<<=` and
`>>=` with a supported scalar INT/UINT shift distance. Range bounds and steps
follow the rules above, including negative indices, descending and nonunit
strides, inclusive bounds, and the default step of one. Both bounds are required.
The selected register width is its actual runtime entry count; the RHS supplies
only a distance, so no compile-time selection width is required.

The first selected entry is the low bit of the selected register. Left shifts
move bits toward later entries in selection order; right shifts move them toward
earlier entries. This ordering also applies to descending ranges. Bits shifted
out are discarded, vacated entries become zero, and unselected destination bits
are preserved. A singleton range remains a one-bit register with its index checked.

The converter captures start, step, and stop once, validates all selected
positions, and rejects zero steps, empty selections, and duplicate normalized
destinations before reading selected bits or evaluating the distance. It then
snapshots all selected bits and evaluates the distance once. Count expressions
may read the original destination, including through integer casts of scalar
bits, integer-array elements, or `popcount` of supported register expressions.
Scalar-bit, Boolean, and floating-point values require a supported explicit
integer conversion. Register-valued counts are rejected; supported functions
such as `popcount` can instead produce a scalar integer count.

Nonnegative INT/UINT counts are accepted; representable negative counts fail
when reached. Counts at least the runtime selection length produce an all-zero
result. Full-width UINT counts reach that zero-result path before any conversion
to INT or shift-index arithmetic. Very large nonnegative literal distances also
produce zeros without emitting unparseable QCX literals. Existing native integer
expression rules still apply: an out-of-range operand of unary signed negation
is rejected during conversion.

The complete result is prepared in private storage before any destination write.
Failed range checks, capacity checks, count accesses, or count arithmetic do not
partially update the destination. Checks and snapshots execute only on reached
paths; skipped branches, transfers, and empty loops still validate names, types,
and syntax without evaluating runtime values. Each reentry captures a fresh range
and resets its active result entries, so shorter and longer selections may share
storage without stale results. Iterators remain scoped and read-only.

For a destination width `N`, the position, original-bit, and result buffers each
have capacity `C = min(N, MAX_RUNTIME_BIT_RANGE_POSITIONS)`, with the existing
position limit normally 1,000,000. A successful destination contains distinct
positions and therefore never needs more than `N` entries. A reached selection
exceeding `C` fails before the snapshots, distance evaluation, or destination
writes. Larger registers may still use selections that fit the capacity.
Destination widths and capacities must fit QCX INT indexing.

Duplicate checking uses the arithmetic progression's modular repetition period,
derived from its first two captured positions without recapturing the step. A
generated integer remainder/division loop computes the period; selections longer
than that period repeat a destination and fail. This avoids quadratic comparisons
and native bitmap limits. Snapshot, shift, and write loops are emitted once and
use bounded indices. Output limits are checked before capture and before writes;
temporary declarations are hoisted where required, and cleanup preserves
caller-owned storage on success and failed lowering. Existing QCX instructions
suffice; no new bra instruction or build option is required.

The register-shift operations follow the
[OpenQASM classical register rules](https://openqasm.com/language/classical.html#classical-bits-and-registers).
Signed nonnegative counts, precise capture/check timing, duplicate rejection,
and resource limits are documented converter policies.

<!-- runtime-bit-range-shift-example -->
```qasm
OPENQASM 3.0;
bit[8] flags = "10110110";
int start = 7;
int step = -2;
int stop = 0;
uint distance = 1;
flags[start:step:stop] <<= distance; // Positions 7, 5, 3, 1; flags is "00111100".
start = 0;
step = 1;
stop = 3;
distance = 2;
flags[start:step:stop] >>= distance; // flags is "00110011".
distance = 4;
flags[start:stop] <<= distance;     // Length is four; flags is "00110000".
bit[1] one = "1";
int last = -1;
one[last:last] <<= int(one[0]);     // Original bit supplies count one; result is "0".
```

Runtime ranges can be [copied into registers or selections](#copies-from-runtime-bit-register-ranges).
Other runtime range expression reads, arithmetic
compound assignments, aliases, computed destination sources, further indexing
of selection results, and general block-local declarations remain unsupported.
Measurements have the separate [range destination subset](#runtime-range-measurement-destinations) below.

### Runtime index-set measurement destinations

Both measurement syntaxes accept a runtime-dependent index set on a named,
writable `bit[n]` register. The source must be a whole qubit register or a static
qubit-register selection of the same width. The first source qubit maps to the
first destination entry, and subsequent entries follow selection order.
As with other measurements, both operands must be registers or both scalars:
`flags[{i}]` is a `bit[1]` destination and accepts `q[{0}]`, `q[0:0]`, or a whole
`qubit[1]` register, but does not accept scalar `q[0]` or a scalar qubit.

Destination entries may independently use supported INT/UINT expressions,
including nested accesses and negative indices. Each occurrence is captured
once in source order, normalized against the destination register width, and
checked before any measurement. UINT values pass checked conversion to QCX INT.
Duplicate normalized destinations are rejected before measuring: for
`bit[4] flags`, `{i, 3}` fails when `i == -1`. Repeated source qubits are allowed
and measured in their listed order.

All destination indices see the original classical values. Every QCX `M` is
immediately followed by a copy of `:OUTCOME` into private storage. Only after
all measurements complete are those results written to the selected destinations;
unselected bits are preserved. Destination bounds, duplicate, nested-access,
and arithmetic failures therefore happen before quantum collapse or destination
writes. This does not promise rollback for backend failures after measurement
has begun.

The equal-sized register mapping and two equivalent syntaxes follow the
[OpenQASM measurement rules](https://openqasm.com/language/insts.html#measurement).
The exact capture/check timing, delayed writes, and rejection of duplicate
destinations are converter policies; the specification does not settle
conflicting writes to repeated destination indices.

Source qubit selections in this subset accept signed integer literals and
supported constant INT/UINT expressions, including named constants. Ranges
require both bounds and a nonzero constant step; empty and out-of-bounds
selections are rejected during conversion. Runtime- or iterator-dependent
qubit selectors remain unsupported, including unrolled iterator dependencies.
Destination indices may depend on active iterators, with normal scoped shadowing;
constant registers and active iterators remain read-only.

Runtime destination checks execute only on reached paths. Skipped branches,
loop transfers, and empty loops do not evaluate those indices or measure qubits,
but names, types, syntax, source qubit selections, and matching register widths
are still validated. Entirely static destinations retain the existing static
measurement path and conversion-time checks.

<!-- runtime-bit-set-measurement-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[4] flags = "0100";
int high = -1;
uint low = 0;
x q[0];
measure q[{1, 0}] -> flags[{high, low}]; // flags is "0101".
reset q;
if (flags[low]) { x q[1]; }
flags[{low, high}] = measure q;         // flags is "1100".
bit[1] one = "0";
one[{low}] = measure q[{1}];            // A register destination; one is "1".
```

Destination registers wider than native UINT use existing QCX element storage,
while `bit[1]` keeps scalar storage with its index still checked. A selection of `k` entries
uses `k*(k-1)/2` pairwise comparisons for duplicate detection, avoiding a native
bitmap limit. The converter checks the comparison count against the QCX output
budget before allocating capture storage, and checks the full generated output
before emitting any measurement. Destination register and selection widths must
fit QCX INT. Captured indices and outcomes are released on success and failed
lowering while preserving caller-owned temporary storage. No new bra instruction
or build option is required.

Runtime ranges support the measurement subset described next. Runtime qubit
indexing/selections, aliases, computed sources, and further indexing of selections
remain unsupported.

### Runtime range measurement destinations

Both measurement syntaxes accept `flags[start:stop]` and
`flags[start:step:stop]` destinations on named mutable `bit[n]` registers when
any range operand depends on runtime storage or an enclosing iterator. The
source must be a whole qubit register or a static qubit-register selection.
Its known width supplies the required destination count, which is checked at
runtime before measurement. The first source qubit maps to the first destination
entry, followed by the remaining qubits in their selection order.

Both operands must be register-valued. A singleton destination range accepts
`q[{0}]`, `q[0:0]`, or a whole `qubit[1]` register; scalar `q[0]` and scalar
qubits require scalar destinations instead. Static source selections accept
integer literals and supported constant INT/UINT expressions, including named
constants. Repeated source qubits are allowed and measured in their listed order.
Runtime- or iterator-dependent qubit selectors remain unsupported, even for
unrolled iterators. Static source selection size is bounded before expanding
large whole registers or ranges.

Destination bounds and steps use the existing INT/UINT range rules, including
negative indices, descending and nonunit strides, inclusive bounds, and a default
step of one. Both bounds are required. The raw progression is formed before
normalizing negative positions; a stop bound need not itself be selected.
Start, step, and stop are captured once, using the original classical values.
Zero steps, empty selections, invalid selected positions, width mismatches,
and repeated normalized destinations fail before the first QCX `M`.

Each `M` is immediately followed by a copy of `:OUTCOME` into a private outcome
array. Only after all measurements finish does a runtime loop write the selected
destinations using the captured position table. This preserves unselected bits
and avoids reevaluating index expressions that depend on destination bits.
Destination and arithmetic failures therefore precede quantum collapse and
destination writes. Backend failures after measurements begin do not imply
rollback.

Runtime range capture, checks, measurements, and writes stay on reached paths.
Skipped branches, transfers, and empty loops still validate names, types, source
qubit selections, and supported syntax without evaluating destination operands.
Reentry captures a fresh range and replaces every outcome before writing it.
Scoped shadowing and live bit-register iteration retain their existing behavior;
constant registers and active iterators remain read-only. Static destinations
and supported index-set measurement destinations retain their existing paths.

For a static source width `k`, the private position and outcome arrays each
contain exactly `k` INT entries. Longer reached ranges fail before overflowing
the position table; shorter ranges fail the count check. The source width must
fit QCX INT and generated buffers must fit the existing runtime range-position
limit. Duplicate detection reuses the modular repetition-period check from
runtime range shifts, avoiding quadratic comparisons and native bitmap limits.
The source emits `k` fixed-qubit measurements and immediate outcome copies;
the destination write loop is emitted once. Conversion checks the output budget
before capture and again before emitting any measurement, including all buffer
declarations and generated checks. Cleanup releases private storage and restores
expression state while preserving caller-owned temporaries, including after
failed lowering. No new bra instruction or build option is required.

The equal-sized register mapping and equivalent syntaxes follow
[OpenQASM measurement](https://openqasm.com/language/insts.html#measurement) and
[register slicing](https://openqasm.com/language/types.html#register-concatenation-and-slicing).
Precise capture/check timing, duplicate rejection, delayed writes, and resource
limits are documented converter policies.

<!-- runtime-bit-range-measurement-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
reset q;
x q[0];
x q[2];
bit[8] flags = "10000000";
int start = 1;
int step = 2;
int stop = 6;
measure q[{2, 0, 1}] -> flags[start:step:stop]; // Positions 1, 3, 5; flags is "10001010".
reset q;
if (flags[1]) { x q[1]; }
start = -1;
step = -2;
stop = -5;
flags[start:step:stop] = measure q;            // Positions 7, 5, 3; flags is "00100010".
bit[1] one = "0";
int low = 0;
one[low:low] = measure q[{1}];                // One-bit register result is "1".
```

Runtime ranges also support [copies into registers or selections](#copies-from-runtime-bit-register-ranges).
Runtime qubit indexing/selections, other runtime range expression reads, aliases,
computed source/destination registers, further selection indexing, and general
block-local declarations remain unsupported.

### Bit-register equality

`==` and `!=` accept two register-valued operands of equal width and produce a
scalar Boolean result. Operands may be whole `bit[n]` variables, bit-string
literals, supported static/runtime ranges or index sets, or nested register
bitwise, shift, and rotation expressions. Selection order defines the
corresponding bits to compare; repeated source indices are allowed. A one-bit register or selection remains
distinct from a scalar bit: `a[0:0] == "1"` is supported, but `a[0] == "1"`
is rejected. Scalar bit comparisons such as `a[0] == true` remain supported.

```qasm
bit[4] a = "1010";
bit[4] b = "0101";
bool same = a[3:-1:0] == b;             // true
bool different = (~a ^ b) != "0000";   // false
const bool FOLDED = ("0011" << 1) == "0110";
if (same && !(a == b)) { a = b; }
while (a != "0000") { a >>= 1; }
```

Results can initialize or be assigned to Boolean variables or scalar bits, be
compared with Boolean/scalar-bit values, and be used in `if`/`else`, `while`,
and logical `!`, `&&`, and `||` expressions. Supported explicit casts can convert
the Boolean result; this does not introduce casts of register values themselves.
Constant Boolean declarations accept comparisons of literal and named constant
register expressions with constant shift counts.

Equal-width register expressions also support `<`, `<=`, `>`, and `>=`.
They are compared as unsigned bit patterns: the highest register position is
the most significant. For selections, position zero is the first selected bit;
the last selected bit is the most significant. This is the converter's
interpretation of register ordering; the OpenQASM comparison section does not
explicitly specify ordering or width promotion for registers. Comparisons of
unequal widths remain unsupported.

<!-- bit-register-ordering-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
bit[4] b = "1101";
int first = 0;
int last = 3;
bool lower = a < b;                       // true
bool reversed = a[last:-1:first] >= b;    // true
bool equal = a[first:last] <= a;           // true
```

Both complete operands are evaluated left to right before any mismatching bit
can end the comparison. Outer logical operators retain short-circuit evaluation,
so runtime checks in an unreached comparison are skipped. Types, names, widths,
and static selection bounds are still checked on skipped paths and in empty
loop bodies, without evaluating runtime shift counts or reading bits.

Runtime comparisons use element-wise QCX `JUMPIF`/`JUMP` instructions, not packing
into a native integer, so registers wider than native `UINT` are supported.
Private intermediate storage stays alive until both operands are evaluated and
is released afterward. No new bra instruction is required. Unequal operand
widths, scalar/register mixing, bare register conditions, aliases, and indexing of computed register
results remain unsupported.

#### Equality with runtime bit-register and angle-bit ranges

A range on a mutable or constant bit register may have runtime-dependent
start, step, or stop values in either or both comparison operands. Both bounds
are required; an omitted step means `1`. Signed and unsigned integer bounds
and steps use the same checked selection rules as runtime-range iteration
and `popcount`: ranges must be nonempty, steps must be nonzero, and every
selected position must be valid. Negative positions and repeated physical
source bits are supported, with comparisons following selection order.

Each operand may contain nested `&`, `|`, `^`, `~`, `<<`, `>>`, `rotl`, and
`rotr` expressions, mixing runtime ranges with supported known-width register
expressions. Shift and rotation distances retain their existing type and
domain rules. A scalar `bit` remains distinct from a one-bit register.

Runtime bit ranges of sized angles also work in either operand, including
constant angles, angle array elements, and angle loop iterators. They can be
combined with bit-register ranges and fixed-width register expressions. An
unsized angle does not provide a selectable bit width. Angle selections follow
the same ordering and runtime checks as elsewhere in the converter.

The converter supports equal-width register comparisons. Statically known
width mismatches are rejected during conversion. When lengths depend on
runtime values, both complete operands are evaluated into private snapshots,
left to right, before QCX `ASSERT` checks that their lengths agree. A mismatch
is an error for all comparison operators; it does not produce a Boolean result or
pad or truncate either operand. Each nested binary register operation also
checks its operand lengths after completing its own children. Buffer capacities
are storage bounds and do not determine selected lengths.

Only after successful operand evaluation and length checks may a mismatching
bit end the comparison. A mismatch cannot suppress an error in a later bit,
range bound, or shift/rotation distance. Surrounding `!`, `&&`, and `||` retain
their Boolean behavior; an unreached comparison performs no runtime reads or
checks. Names, types, supported forms, known widths, and wholly static selection
bounds are still validated on skipped paths and in empty loop bodies. A comparison
that depends on a runtime range cannot initialize a `const bool`.

<!-- runtime-range-equality-example -->
```qasm
OPENQASM 3.0;
bit[4] a = "1011";
const bit[4] B = "1101";
int first = 0;
int last = 3;
bool different = a[first:last] != B;                     // true
bool reversed = a[last:-1:first] == B;                   // true
bool nested = ((a[first:last] & B) ^ "0011") == "1010"; // true
bool rotated = rotl(a[first:last], 1) == "0111";          // true
bool skipped = false && (a[first:0:last] == B);          // false; no step check
int visits = 0;
while (a[first:last] != B) {
    a = B;
    visits += 1;
}
bool fresh = a[first:last] == B;                         // true; visits == 1
```

<!-- runtime-angle-range-equality-example -->
```qasm
OPENQASM 3.0;
angle[4] phase = angle[4]("1011");
bit[4] bits = "1011";
int first = 0;
int last = 3;
bool same = phase[first:last] == bits;              // true
bool different = ~phase[first:last] != bits;         // true
bool reversed = phase[last:-1:first] == "1101";      // true
bool skipped = false && (phase[first:0:last] == bits); // false
```

Comparisons use bit snapshots and an INT-indexed QCX loop, so register widths
may exceed native UINT width. Snapshots remain live through both operands,
including nested comparisons in bounds or distances, and are released after
use. Repeated execution reads the current source bits and bounds. Existing
runtime selection storage and QCX output limits still apply. This support
does not add general runtime-range register expressions in other contexts,
aliases, or indexing of computed register results.

### Checked native INT negation

Unary `-` supports numeric scalars. Native INT negation checks operand and result
bounds in evaluated constant expressions and guards runtime arithmetic against
overflow. It integrates with supported casts, comparisons, indexed expressions,
assignments, and control flow. The checked behavior is:

- INT operands must have native width. Non-native signed-width emulation remains
  separate work; existing UINT modular, floating-point, and complex behavior is
  preserved.
- Direct unary minus applied to a single integer literal whose value is one
  greater than `INT_MAX` represents `INT_MIN`. Parentheses around that literal
  and hexadecimal spellings are equivalent. This is distinct from negating an
  existing `INT_MIN` value, including through a constant name or nested negation.
- Other evaluated constant INT negations must have representable operands and
  results. Negating `INT_MIN` is an overflow error, not wrapping arithmetic.
- Runtime negation checks the captured operand before multiplication by
  `-1`. Such checks, including for overflowing literal results in runtime
  expressions, stay on reached paths so logical short-circuiting and
  skipped branches/loops can bypass them.
- An out-of-range operand that cannot be captured in native INT remains a
  conversion error, even on skipped runtime paths. The minimum-value literal
  spelling above is the sole exception for an otherwise oversized literal.
- Metadata validation checks numeric types and native signed widths without
  evaluating negation, operand arithmetic, or element indices in empty loops.
  It does not fabricate iterator values. Declared widths remain tied to
  declaration scope when iterators shadow width constants.

For example, on a target with 32-bit native INT:

```qasm
const int MINIMUM = -2147483648;       // valid minimum-value literal
int a = -3;
int x = -a;                           // 3
int y = MINIMUM;
bool off = false && bool(-y);         // false: negation is not reached
float widened = -float(y);            // 2147483648.0: negate after widening
// const int BAD = -MINIMUM;          // constant negation overflow
// y = -y;                           // runtime assertion failure
```

The minimum-literal exception is syntactic: `-(2147483648)` and `-0x80000000`
are valid here, but `-(2147483647 + 1)` has an unrepresentable INT operand and
is rejected. More generally, `float(-y)` and `uint(-y)` still perform checked
INT negation before the cast; widening or converting must occur before `-`
when different operand semantics are intended.

Runtime lowering evaluates the operand once, copies it to private INT storage,
and emits `ASSERT temporary \= INT_MIN` before multiplication by `-1`. A reached
failure is reported by the bra CLI with the failed assertion and a nonzero exit
status. The destination is written only after the expression succeeds. No new
bra instruction is required. Private temporaries are released after use and on
failed operand/condition lowering, without releasing caller-owned storage.

Only unary negation is checked here. Overflow in other operand arithmetic is
not made safe by an enclosing negation; general signed-arithmetic checking and
non-native signed-width emulation remain separate work.

### Integer exponentiation

`**` supports constant folding and runtime lowering, and `**=` supports scalar
and indexed integer-array targets. Both integrate with supported expressions,
casts, comparisons, conditions, and loops. An integer result needs a comparison
or `bool(...)` cast to become a condition. The integer subset follows these rules:

- Operands must be scalar `int` or `uint`. Signed operands must have native QCX
  width; non-native signed widths, Boolean/bit values, registers, floating-point,
  and complex operands require a different feature or an explicit supported cast.
- Result types use the converter's existing integer promotion rules. Two UINT
  operands use the wider width. Native INT mixed with a narrower UINT produces
  INT; mixing with a native-width UINT produces native UINT.
- Exponents must be nonnegative. A negative signed exponent is rejected before
  promotion to UINT; it is not interpreted as a large unsigned exponent.
- An exponent of zero produces one, including `0 ** 0`. A positive exponent of
  a zero base produces zero. Negative signed bases retain ordinary odd/even
  power behavior when the promoted result type is INT.
- UINT powers are modular at the promoted result width. INT powers must fit
  native QCX INT, with checks before overflowing multiplication. General signed
  overflow checking for other arithmetic remains separate work.
- Domain/range errors in evaluated constant expressions are conversion errors.
  Runtime checks execute only on reached paths, including for literal operands
  in runtime expressions. Metadata checks validate types without evaluating
  powers, exponent arithmetic, or element indices in empty loops.
- Signed literal operands must fit native INT, even when an exponent of zero
  would make the result small. Use an explicit supported cast when UINT input
  semantics are intended; general overflow in operand arithmetic is not checked.
- Operands are evaluated once, left to right. Parser precedence is retained:
  `2 ** 3 ** 2` means `2 ** (3 ** 2)`, and `-2 ** 2` means `-(2 ** 2)`.

```qasm
const int SIZE = 2 ** 3;               // 8
int a = -2;
int n = 3;
int signed_power = a ** n;             // -8
uint[4] u = 3;
uint[4] e = 4;
uint[4] wrapped = u ** e;              // 81 modulo 16: 1
int promoted = u ** n;                // INT result: 27
array[int, 2] values = {2, 3};
int i = 0;
values[i] **= values[i + 1];           // values[0] becomes 8
if (signed_power < 0 && bool(u ** e)) { a **= 2; }
```

Compound assignment captures its destination index once and snapshots the
destination's value before evaluating the RHS. It writes only after the power
calculation and its checks succeed. Result promotion is the same as for an
ordinary `a ** b` expression, followed by the destination's usual conversion
and UINT normalization. A native UINT result assigned to INT must fit INT;
this is a checked conversion, not a signed reinterpretation cast.

Constant UINT folding uses modular exponentiation. Runtime powers use
exponentiation by squaring with a logarithmic number of iterations and existing
QCX arithmetic, branches, and `ASSERT` instructions. Signed powers compute
unsigned magnitudes (including the magnitude of `INT_MIN`) and guard each
product with an unsigned division bound before multiplying. The final signed
value is reconstructed without negating `INT_MIN` or narrowing its magnitude
through INT. An unused final squaring is not performed, so a valid `a ** 1`
does not fail because `a ** 2` would overflow. No new bra instruction is required.

Negative exponents and signed-result overflow on reached runtime paths fail
through `ASSERT`; the bra CLI reports the failed assertion and exits nonzero.
Surrounding logical short-circuiting and skipped branches/loops skip these
runtime checks. Invalid operand types, non-native signed widths, and
iterator-dependent cast widths remain conversion errors even on skipped paths.
Already declared widths retain their declaration-scope meaning when a loop
iterator shadows a constant used in the declaration. Iterators remain read-only.

OpenQASM specifies integer power and compound assignment in its
[classical instructions](https://openqasm.com/versions/3.0/language/classical.html#integers)
and general numeric promotion in
[types and casting](https://openqasm.com/versions/3.0/language/types.html#casting-specifics).
The nonnegative-exponent restriction, zero-to-zero convention, and checked
signed-overflow behavior above define this converter's subset; they are not
claimed as explicit requirements of those specification sections. The built-in
`pow(...)` function and floating-point/complex powers remain separate work.

### Scalar integer shifts

`<<` and `>>` accept scalar `int` and `uint` operands, including constants,
variables, integer-array elements, and supported integer casts. The left
operand alone determines the result type and width; the count may independently
be signed or unsigned. Thus `int_value >> uint_count` remains signed, and
`uint[4](15) << int(1)` remains four-bit unsigned. Boolean, bit, floating-point,
and complex operands require an explicit integer cast; whole registers and
arrays are not shift operands.

The supported subset uses these rules, where `k` is the count:

- Counts must satisfy `0 <= k < width`, even when the shifted value is zero.
  Unsized integers use the host-derived native width `W`; `uint[n]` uses `n`.
- Unsigned left shifts discard high bits modulo `2**width`; unsigned right
  shifts fill with zeroes. Every narrow unsigned intermediate is masked.
- Signed left shifts compute exact multiplication by `2**k`, including for
  negative values, and reject results outside the native QCX `INT` range.
- Signed right shifts sign-extend, equivalently dividing by `2**k` and rounding
  toward negative infinity. For example, `-3 >> 1` is `-2`, not `-1`.

These are the converter's supported count and signed-overflow policies, not
general fixed-width signed-integer emulation. Non-native signed widths on
shifted variables, constants, casts, and array elements are rejected rather
than silently shifted at native width. As with other supported loop operations,
`int` loop iterators use native signed semantics without enforcing their
declared width. A count's declared signed width does not change the shifted
value's width.

```qasm
OPENQASM 3.0;
const int NEGATIVE = -3 >> 1;              // -2
uint[4] a = 15;
int count = 1;
uint[4] result = (a << count) >> count;    // 7, not 15
a <<= count;                             // 14
int b = -3;
b >>= uint(1);                           // -2
array[int, 2] values = {1, 4};
int i = -1;
values[i] <<= values[0];                  // values[1] becomes 8
```

Literal expressions are folded when valid. Invalid evaluated constant counts
or signed left-shift overflow produce `InvalidShiftException`, reported by
the CLI without a Python traceback. In runtime expressions, even invalid
literal shifts emit runtime checks so branches and logical short-circuiting
can skip them. Native bra checks native-width counts and signed overflow;
the converter adds `ASSERT` guards for narrow unsigned widths. Native count
failures throw `std::out_of_range`, signed left-shift overflow throws
`std::overflow_error`, and narrow-width guard failures throw
`bra::assertion_error`. The bra CLI currently leaves these exceptions uncaught.

Compound `<<=` and `>>=` support scalar integer destinations and static or
runtime-indexed `int` and `uint` array elements. They evaluate the count and
complete a checked shift in private storage before writing the destination. Destination
indices are captured and checked once before the RHS; self-references read the
original value. Active loop iterators remain read-only. Names, types, and index
syntax are validated in empty loop bodies without evaluating counts or index
values there. General signed arithmetic and count-expression overflow checks,
scalar-bit shifts, and signed-width emulation remain future work.
Whole-register shifts follow the separate policy above.

### One-dimensional integer arrays

The converter supports top-level `array[int, N]` declarations, including sized
integer base types such as `array[int[8], N]`. The size must be a positive
compile-time integer within the host-derived QCX `INT` range described under
runtime-bound loops. Integer literals, previously declared constants, supported
constant arithmetic, and explicit integer casts can supply the size. It is
resolved in declaration scope and does not change when a later iterator shadows
a dimension constant. Declared element widths are not enforced.

```qasm
OPENQASM 3.0;
const int N = 3;
int seed = 2;
array[int, N] values = {seed, seed + 1, int(4.5)};
values[-1] += values[0];
values[1] %= 2;
// values now contains 2, 1, 6.
```

An initializer must be a flat array literal with exactly `N` integer-valued
elements. Elements may use supported runtime expressions and explicit integer
casts; their computations execute in source order at the declaration's
position. Arrays without initializers are accepted, but their OpenQASM values
are undefined until assigned. Do not rely on the backend's initial storage
contents.

Individual elements can appear wherever a supported scalar integer expression
is accepted, including arithmetic, comparisons, casts, gate parameters, and
range bounds, steps, or set elements. Element assignments support `=`, `+=`,
`-=`, `*=`, `/=`, and `%=` with the converter's existing integer assignment
conversions, and `&=`, `|=`, and `^=` with integer RHS operands.
These operations preserve the original target while evaluating the RHS.

An element index must be an integer expression. Previously declared constants,
constant-loop iterator values, runtime variables, and runtime iterators are
accepted. Indices must lie in `[-N, N - 1]`; negative indices count from the end.
Even a one-element array requires explicit indexing
when used as a scalar. Qubit indices remain static; bit-register elements also
support runtime indices as described below.

Unsigned, floating-point, Boolean, complex, and angle arrays are supported as described below.
Multidimensional arrays, arrays of other element types, slices outside supported loops and assignments,
whole-array arithmetic or assignments, array-copy initializers, aliases, and
block-local array declarations remain unsupported.

### One-dimensional unsigned integer arrays

The converter supports top-level `array[uint, N]` and `array[uint[n], N]`
declarations. The array size follows the integer-array rules: a positive
compile-time integer no larger than QCX `INT_MAX`. Element widths follow the
scalar UINT rules: unsized `uint` uses native width `W`, and specified widths
must satisfy `1 <= n <= W`. Sizes and element widths are resolved in declaration
scope; a later iterator shadowing a dimension constant does not change them.

```qasm
OPENQASM 3.0;
array[uint[8], 3] values = {1, 2, 255};
int index = -1;
values[index] += uint[8](1);
uint wrapped = values[0] + uint[8](255); // 0: the intermediate result is 8-bit.
```

Elements use native QCX `UINT` storage. Narrow element assignments are masked
modulo `2**n`; unsized and explicitly native-width values use native unsigned
wrapping. A flat initializer, when present, must contain exactly `N` expressions
and is evaluated in element order. It uses scalar UINT assignment conversions,
including integer wrapping and Boolean promotion. Floating-point and complex
values require explicit UINT casts. An uninitialized array has undefined
OpenQASM values until assigned; do not rely on backend storage initialization.

Static and runtime integer indices, negative-index normalization, bounds checks,
and explicit singleton indexing follow the shared array-indexing rules below.
Reads retain their unsigned width and specified/unsized metadata, including
runtime reads copied into temporaries. Thus intermediate arithmetic, complement,
negation, shifts, powers, comparisons, and casts behave like their scalar UINT
equivalents; assigning into a wider destination does not widen an earlier
intermediate result retroactively. Elements can also supply supported runtime
indices, `for int` bounds/steps/set elements, and gate parameters.

Element assignments support `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`,
`<<=`, `>>=`, and `**=`. Compound assignments use the scalar operand-promotion
rules, compute a completed result in private storage, and then convert and mask
it to the destination width. Shift counts are checked against the shifted
element's width. Exponentiation follows the existing signed/unsigned result
rules; a signed intermediate can require overflow checks even when assigned
back to UINT.

The destination index is captured and checked once before RHS evaluation. Its
original value is copied for compound computations; explicit RHS reads perform
their own accesses. Failed runtime checks precede the final destination write.
Skipped control-flow paths skip runtime operations and checks, while static
indices retain conversion-time bounds validation. These rules do not add general
signed-arithmetic overflow checking. Direct iteration is described under
[Unsigned-integer-array for loops](#unsigned-integer-array-for-loops).

### One-dimensional floating-point arrays

The converter supports top-level `array[float, N]` declarations, including sized
base types such as `float[32]` and `float[64]`. Size expressions follow the same
rules as integer arrays: `N` must be a positive compile-time integer no larger
than QCX `INT_MAX`, resolved in declaration scope.

Elements use bra's native QCX `REAL` storage, including expression temporaries.
As with scalar floats, declared floating-point widths are not enforced:
`float[32]` does not introduce single-precision rounding, and `float[64]` does
not guarantee double precision independently of the bra build. This feature
does not add width emulation or change existing scalar-float arithmetic.

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
array[float, 3] angles = {0.1, 0.2, 0.3};
int index = -1;
angles[index] += 0.05;
rx(angles[index]) q;
```

An optional initializer must be a flat array literal with exactly `N` elements.
Expressions execute in element order at the declaration's position and use the
existing scalar-float assignment conversions: integers and Booleans can promote
to floats, but complex values require an explicit conversion. An initializer
can read elements of a previously declared array. Without an initializer,
OpenQASM element values remain undefined until assigned; do not rely on the
backend's zero-initialized storage.

Element assignments support `=`, `+=`, `-=`, `*=`, and `/=`. Reads can be used in
supported scalar arithmetic, unary negation, comparisons, casts, gate parameters,
and `gphase` arguments. Comparisons promote compatible integer and floating-point
operands without truncating the float. A floating-point element is not a direct
Boolean condition; use a comparison or `bool(angles[index])`. Similarly, integer
loop bounds, steps, set elements, and array indices require an explicit integer
cast when supplied by a floating-point expression.

Static and runtime integer indices, including negative indices, follow the
shared indexing rules below. Access remains explicitly indexed even for
one-element arrays. Position-based integer loops can read and update float
arrays, and direct `for float` iteration is supported as described under
[Floating-point-array for loops](#floating-point-array-for-loops).

Floating-point `%`, `**`, bitwise operators, and shifts are not supported,
including their compound assignments. These type restrictions also apply to
integer-only operators in empty loops; validation does not execute the skipped
arithmetic. Unsupported array element types, multidimensional arrays, slices outside supported loops and assignments,
whole-array expressions or assignments, array-copy initializers, aliases, and
block-local declarations remain separate work.

### One-dimensional Boolean arrays

The converter supports top-level `array[bool, N]` declarations. `N` must be a
positive compile-time integer no larger than QCX `INT_MAX`, following the shared
array-size rules. Elements use QCX `INT` storage with values zero and one, but
retain Boolean type metadata; they are not integer expressions.

```qasm
OPENQASM 3.0;
bit b = 1;
array[bool, 3] flags = {true, b, false};
int index = -1;
flags[index] = flags[0] && !flags[1];
bool ready = flags[0] || flags[index];
```

An optional initializer must be a flat array literal with exactly `N` elements,
evaluated in element order at the declaration's position. Elements follow the
scalar Boolean assignment rules: Boolean and scalar-bit values, comparisons,
logical expressions, and integer constant values zero and one are accepted.
General numeric values require an explicit `bool(...)` cast, including runtime
integer variables whose values happen to be zero or one. Complex-to-Boolean
casts remain unsupported. Initializers can read previously declared array
elements. Without initialization, element values are undefined until assigned;
do not rely on the backend's zero-initialized storage.

Static and runtime integer indices, negative indices, bounds checks, and explicit
singleton indexing follow the shared indexing rules below. Only plain `=`
element assignment is supported. A destination index is captured once before
evaluating the RHS, including self-referential logical assignments.

Reads support direct conditions, `!`, short-circuit `&&` and `||`, Boolean/scalar-bit
`==` and `!=`, the supported Boolean casts, and scalar Boolean/bit assignments.
Numeric assignment conversions follow the scalar Boolean rules. Arithmetic,
ordered comparisons, bitwise operators, shifts, and compound assignments on
Boolean elements are unsupported; use an explicit numeric cast where applicable.
Integer indices, range bounds, steps, set elements, and gate parameters also
require an explicit numeric cast of a Boolean value. Direct iteration is described
under [Boolean-array for loops](#boolean-array-for-loops).

Multidimensional arrays, slices outside supported loops and assignments, whole-array expressions or assignments,
array-copy initializers, aliases, and block-local declarations remain unsupported.

### One-dimensional complex arrays

The converter supports top-level `array[complex, N]` declarations, including
explicit base types such as `complex[float[32]]` and `complex[float[64]]`. As for
scalar complex values, elements use native QCX `COMPLEX` storage; declared
component widths do not introduce rounding or precision conversion. The array
size follows the shared rules: a positive compile-time integer no larger than
QCX `INT_MAX`, resolved in declaration scope.

```qasm
OPENQASM 3.0;
complex seed = 1 + 2.0im;
array[complex[float[64]], 2] values = {seed, -seed};
int index = -1;
values[index] += values[0] * 2;
complex result = values[index];
// result is 1 + 2im; seed is unchanged.
```

An optional initializer must be a flat array literal with exactly `N` elements,
evaluated in element order at the declaration's position. Initializers use
scalar-complex assignment conversions, including numeric, Boolean, and scalar-bit
promotion. They can read elements of previously declared arrays. Without an
initializer, elements have undefined OpenQASM values until assigned; do not rely
on backend zero initialization.

Static and runtime integer indexing, negative-index normalization, bounds checks,
and explicit singleton indexing follow the shared rules below. Element assignments
support `=`, `+=`, `-=`, `*=`, and `/=`. Destination indices are captured and
checked once before RHS evaluation and stay reserved through the write, including
self-referential arithmetic. Reads support complex `+`, `-`, `*`, `/`, unary
negation, and existing scalar-complex casts. Plain implicit assignment to integer,
unsigned-integer, floating-point, Boolean, or bit destinations is rejected.

Complex values cannot be compared, used as direct conditions, or passed directly
as gate parameters or `gphase` arguments. Boolean/bit casts, `%`, `**`, bitwise
operators, shifts, and their compound assignments remain unsupported. An explicit
`float(...)` cast extracts the real component for supported arithmetic or gate
parameters. Complex-to-`uint` casts extract the real component, with native range
checks and the specified UINT-width normalization. Existing loop-expression
analysis still rejects runtime complex-to-`int` casts, including conversion through
`float`; runtime complex-to-`float` casts are also rejected in conditions analyzed
by that path. Use a supported `uint(...)` cast when a complex value supplies an
integer index, range bound, step, or set element.

Position-based loops can update complex elements; direct iteration is described
under [Complex-array for loops](#complex-array-for-loops). Multidimensional arrays,
slices outside supported loops and assignments, whole-array expressions or assignments, array-copy initializers, aliases,
and block-local declarations remain unsupported.

### One-dimensional angle arrays

The converter supports top-level `array[angle, N]` and `array[angle[n], N]`
declarations. `N` must be a positive compile-time integer no larger than QCX
`INT_MAX`. Element widths follow the scalar angle rules: `1 <= n <= W`, where
`W` is native QCX UINT width; unsized `angle` uses `W`. Lengths and widths are
resolved once in declaration scope and remain unchanged when later loop
iterators shadow the constants used in them. Each element uses native UINT
storage with distinct angle typing and its declared width.

An optional initializer must be a flat list of exactly `N` expressions,
evaluated in source order. Elements accept float and angle inputs using the
scalar angle conversion rules, including float quantization and angle-width
conversion. Initializers can read previously declared variables and array
elements. Integer, Boolean, bit-register, and complex inputs require an
appropriate explicit conversion; matching-width bit-register reinterpretation
uses `angle[n](bits)`. Without an initializer, elements have undefined OpenQASM
values until assigned; do not rely on backend zero initialization.

Static and runtime integer indices, negative indices, bounds checks, and
explicit singleton indexing follow the shared rules below. Reads retain the
array's element angle width and sizedness, including through nested expressions.
Elements support existing scalar angle arithmetic, comparisons, bitwise
operations, shifts, `rotl`/`rotr`/`popcount`, trigonometric functions, and supported
casts. They can supply parameters to the supported angle-valued phase gates;
trigonometric results can supply ordinary floating-point gate parameters.
Elements cannot receive measurements or implicitly become numeric values.

Assignments support `=`, `+=`, `-=`, `*=`, `/=`, `&=`, `|=`, `^=`, `<<=`, and `>>=`
with the scalar angle operand restrictions. In particular, scaling requires a
UINT of matching effective width, and `angle /= angle` remains unsupported.
Compound assignments compute at the binary operation's promoted width before
converting back to the element width. A destination index is evaluated and
checked once, and its old value is captured before evaluating a compound RHS.
The complete result is computed before writing the destination. Explicit RHS
accesses, including self-references, evaluate and check their own indices.

Position-based loops can read and update elements. Direct iteration is supported
as described under [Angle-array for loops](#angle-array-for-loops). Skipped
branches, loop bodies, and short-circuit operands suppress runtime index and
operand domain checks; names, types, widths, and index syntax remain validated
in empty loops.
Sources and declaration metadata are preserved, and temporary storage is
released after use. Gate parameters retain their captured values during fusion,
even when array elements are changed before the gates execute.

<!-- angle-array-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
const int W = 4;
array[angle[W], 3] phases = {0.0, pi / 2, pi};
int i = 1;
phases[i] += angle[4](pi / 4);       // Stored bits: 0110, or 3*pi/4.
bit[4] bits = bit[4](phases[i]);
phases[-1] = angle[4](bits);         // Copies the exact pattern.
h q;
p(phases[i]) q;
```

Angle-array elements support [static bit selections](#static-bit-selections-on-sized-angles)
using `phases[index][selector]`, including writes that preserve unselected bits.
[Runtime single-bit indexing](#runtime-single-bit-indexing-on-sized-angles) and
[runtime-valued discrete sets](#runtime-valued-discrete-set-bit-selections) also
support reads and writes, including `phases[element][{i, j}]`.
Array slices outside supported loops and assignments, whole-array operations and copying, aliases, multidimensional
arrays, and local array declarations remain unsupported.
The positive-length restriction is an implementation limitation:
[OpenQASM permits zero-length arrays](https://openqasm.com/language/types.html#arrays),
which cannot be indexed. Zero-length support remains separate work for all
classical array types. No new bra instruction or build option is required.

### Assignment between static array slices

Previously declared one-dimensional `int`, `uint`, `float`, `bool`, `complex`,
and `angle` arrays support `=` between statically bounded range slices. The
slices must have equal lengths and matching declared element types, including
specified widths and sized versus unsized forms. Width expressions are resolved
in declaration scope. Element widths the converter cannot resolve are rejected
for slice copies rather than assumed compatible.

The entire source slice is copied to temporary storage before any destination
element is written. This preserves copy semantics for self-assignment and
overlapping slices. Source positions may repeat, including through negative
aliases. Destination positions must be distinct after normalizing negative
indices; repeated destinations are currently rejected. Two empty slices are a
no-op, consistent with the converter's empty-slice loop interpretation.

<!-- static-array-slice-assignment-example -->
```qasm
OPENQASM 3.0;
array[int, 4] values = {1, 2, 3, 4};
values[1:3] = values[0:2];
// values is {1, 1, 2, 3}; the original source is copied first.
```

Both bounds and steps must be compile-time integers. Steps must be nonzero,
each selected position must be valid, and source and destination counts must
match. The same validation applies in skipped or empty enclosing loops. The
converter emits two QCX `LET` instructions per selected element and rejects a
copy that exceeds `MAX_LOOP_OUTPUT_LINES`. Runtime source or destination slices
with a static slice on the other side are described below. Slice initializers,
whole-array copies, array concatenation, and multidimensional array assignments
remain unsupported.

### Runtime array-slice sources in static slice assignments

The source of a slice assignment may have runtime-dependent `int` or `uint`
start, step, and stop expressions when the destination slice is static. Both
source bounds are required; an omitted step is one. Source bounds are captured
once in written order. The converter checks the complete selected position list
and then asserts that its length matches the destination before reading the
source values. All source values are snapshotted before any destination write,
including for overlapping copies. The element-type, static destination, and
repeated-destination restrictions above continue to apply.

<!-- runtime-array-slice-source-example -->
```qasm
OPENQASM 3.0;
array[int, 4] values = {1, 2, 3, 4};
int first = 0;
int last = 2;
values[1:3] = values[first:last];
// values is {1, 1, 2, 3}.
```

Reached source selections check the step, each selected index, and the
selection length before any destination write. Negative indices are normalized
individually, and repeated source positions are allowed. Empty-to-empty copies
still evaluate and check their source bounds and step before doing no writes.
Skipped copies do not execute runtime checks, but their types, names, static
destination, and source expression types are validated.

Captured source positions use private INT-indexed storage bounded by the
smallest of twice the source array length, `MAX_RUNTIME_BIT_RANGE_POSITIONS`
(currently 1,000,000), and QCX `INT_MAX`. A reached selection exceeding that
capacity fails before the destination is written. The typed value snapshot uses
the static destination length, and generated output is also checked against
`MAX_LOOP_OUTPUT_LINES`.

### Runtime array-slice destinations in static slice assignments

The destination of a slice assignment may have runtime-dependent `int` or
`uint` start, step, and stop expressions when the source slice is static. Both
destination bounds are required; an omitted step is one. The bounds and step
are captured once, in written order. The converter checks the full destination
selection, its length, and the distinctness of its normalized indices before
writing. It snapshots all source values before writing, so overlapping copies
use the original values. The element-type rules above continue to apply.

<!-- runtime-array-slice-destination-example -->
```qasm
OPENQASM 3.0;
array[int, 4] values = {1, 2, 3, 4};
int first = 1;
int last = 3;
values[first:last] = values[0:2];
// values is {1, 1, 2, 3}.
```

Reached selections check for a nonzero step, valid indices, a matching length,
and distinct normalized destination indices. Empty-to-empty copies still
evaluate and check the destination range. Skipped copies do not execute runtime
checks; their names, types, and static source are still validated. Captured
positions use private INT-indexed storage bounded by the smallest of twice the
destination array length, `MAX_RUNTIME_BIT_RANGE_POSITIONS` (currently
1,000,000), and QCX `INT_MAX`. An oversized selection fails before any write.
The typed snapshot uses the static source length; generated output is limited
by `MAX_LOOP_OUTPUT_LINES`. Assignments with runtime-dependent slices on both
sides remain unsupported.

### Runtime array indexing

Signed/unsigned integer, floating-point, Boolean, complex, and angle arrays accept runtime
indices using scalar `int`, `uint`, supported integer arithmetic, enclosing runtime
iterators, and indexed integer-array elements.
Nested accesses such as `values[indices[i]]` are accepted. Floating-point,
Boolean, and scalar bit indices require an explicit integer cast, for example
`values[int(flag)]`. Complex indices require a supported explicit integer cast,
such as `values[uint(z)]`. Whole-register casts retain their existing restrictions.

```qasm
OPENQASM 3.0;
array[int, 3] values = {2, 4, 6};
int index = -1;
int total = values[index];
values[index] %= 4;
values[index] += values[index + 1];
// total is 6; values contains 2, 4, 4; index is still -1.
```

Each runtime index expression is evaluated once per access and copied into
private QCX `INT` storage. The converter emits `ASSERT index >= -N` and
`ASSERT index < N` before accessing storage, then adds `N` to negative indices.
Checks apply to the original signed value; normalization neither changes the
source variable nor negates the index. For accepted sizes and indices, this
normalization does not overflow QCX `INT`, including at native integer endpoints.
UINT indices undergo checked conversion to QCX `INT` before these assertions;
a value above `INT_MAX` throws `std::out_of_range` before the element access.
Unsigned expression widths are enforced before capture. Signed index arithmetic
is not protected by general overflow checks, and declared signed widths remain
unenforced.

Runtime reads copy the selected element into an expression temporary. For
assignments, the destination index is captured and checked before evaluating
the RHS and remains reserved until the write completes. Supported compound
assignments use the same captured destination. In particular, `%=` does not
reevaluate the destination index when reading its original value. Separate
accesses in an expression, including an explicit access to the same element on
the RHS, each evaluate and check their own index.

An executed out-of-bounds index throws `bra::assertion_error` before the element
is read or written; its diagnostic includes the failed comparison and evaluated
values. The bra CLI currently leaves this exception uncaught. Generated runtime
accesses require a bra version supporting `ASSERT`, documented in
[bra.md](../../docs/bra.md). No new bra instruction is introduced.

Index computations, checks, and element accesses remain on the execution paths
that reach them. Skipped branches, loop bodies, and short-circuit operands skip
those runtime operations. For example, `false && values[bad] > 0` does not
execute the access or its bounds checks. Names, types, and supported syntax are
still validated during conversion. Static indices retain their existing
conversion-time bounds checks and generated operands without runtime assertions.
Runtime qubit indexing, dynamic slices, and multidimensional accesses remain
unsupported. Runtime bit-register element indexing is described next.

### Runtime bit-register indexing

A declared `bit[N]` register supports single-element reads, `=` assignments,
and scalar-qubit measurement destinations with runtime integer indices.
Scalar `int`, `uint`, supported integer arithmetic, enclosing integer
iterators, and integer-array elements can supply an index. Floating-point,
Boolean, and scalar bit expressions require an explicit integer cast.
Nested accesses such as `flags[positions[i]]` and `flags[int(flags[i])]` are
accepted. A scalar `bit` is not a runtime-indexable register; `bit[1]` is, even
though QCX stores it as a scalar. A runtime-indexed register's positive size
must fit the host-derived QCX `INT` range described under runtime-bound loops.

```qasm
OPENQASM 3.0;
bit[3] flags = "101";
int index = -1;
bit selected = flags[index];
flags[index - 1] = selected;
bool ready = flags[index] && flags[index - 1];
// selected is 1; flags is "111"; ready is true; index is still -1.
```

An indexed read retains scalar bit semantics, not integer semantics. It can
appear directly in conditions and logical expressions, or be assigned to a
scalar bit or Boolean. Use an explicit numeric cast for arithmetic or gate
parameters. Plain assignments retain the existing scalar bit rules: bit and Boolean
values are accepted, as are integer literals `0` and `1`; general numeric
values, whole registers, and bit strings are rejected. Compound `&=`, `|=`, and
`^=` assignments require a scalar bit RHS as described above; other compound
bit assignments remain unsupported.

Each executed access evaluates its index once, copies it into private QCX
`INT` storage, and checks the original value against `[-N, N - 1]` using
`ASSERT`. Negative indices are then normalized by adding `N` to the private
copy without changing the source variable. Checks precede normalization and
storage access, including for `bit[1]`, whose valid indices are `-1` and `0`.
UINT indices first undergo checked conversion to QCX `INT`; values above
`INT_MAX` throw `std::out_of_range` before index assertions or storage access.
Accepted normalization is representable. General signed index arithmetic
overflow checks and signed-width emulation remain separate work.

Reads copy the selected bit into an expression temporary. Assignment destination
indices are captured and checked before the RHS, and their private storage
remains reserved until the write completes. Separate RHS accesses evaluate and
check their own indices; overlapping reads see the pre-store values.

Both measurement syntaxes accept runtime-indexed destinations:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit[2] outcomes = "00";
int index = -1;
x q;
outcomes[index] = measure q;
measure q -> outcomes[index + 1];
// outcomes is "11"; index is still -1.
```

The destination index is captured, checked, and normalized before QCX `M`.
An invalid destination therefore fails before measurement. `M` is immediately
followed by assignment from `:OUTCOME` using the captured destination.
The source must be a scalar qubit or a statically indexed single qubit, not a
register selection (even a one-element range). Runtime qubit indices remain
unsupported.

Executed bounds failures throw `bra::assertion_error`, currently uncaught by
the bra CLI, with the failed comparison and evaluated values in the diagnostic.
Generated accesses require a bra version supporting `ASSERT`; no new backend
instruction is introduced. Index arithmetic, checks, accesses, and measurements
stay on the paths that reach them: skipped branches, loop bodies, transfers,
and short-circuit operands skip these runtime operations. Names, types, and
syntax are still validated during conversion, including on unreachable paths.
Static indices retain their existing conversion-time checks and QCX output.
Multidimensional indexing and aliases remain unsupported. Plain and bitwise
compound range assignments are described in
[Runtime bit-register range assignments](#runtime-bit-register-range-assignments).
Register-valued measurement destinations are described in
[Runtime index-set measurement destinations](#runtime-index-set-measurement-destinations).
Range measurement destinations have their own [supported subset](#runtime-range-measurement-destinations).
Supported register expressions and assignments are described above.

### Boolean values and conversions

OpenQASM scalar `bool` variables also use QCX `INT` storage: `false` is zero
and `true` is one. The converter tracks Boolean types separately from integers
and bits. Boolean literals, constants, and variables can initialize or be
assigned to Boolean variables and can be used directly in branching conditions.
Comparison and logical expressions can also initialize or be assigned to Boolean
variables. Boolean constant expressions are evaluated during conversion.
Boolean and scalar bit values can be assigned to one another, including single
indexed elements of bit registers. Integer constant values zero and one are also
accepted as Boolean initializers and assignment values. Boolean values can be
assigned to `int`, `uint`, `float`, and `complex` variables as zero or one.
Variables declared without initializers have no defined OpenQASM
value; initialize them before use, as specified in the
[OpenQASM variable rules](https://openqasm.com/language/types.html#variables).

The supported Boolean-related explicit casts are:

- `bool(...)` from `bool`, scalar `bit`, `int`, `uint`, or `float`;
- `bit(...)` from a Boolean or scalar bit; and
- `int(...)`, `uint(...)`, `float(...)`, and `complex(...)` from a Boolean.

Numeric-to-Boolean casts test whether the value is nonzero; they do not first
truncate floating-point values to integers. For example, `bool(-0.25)` is true.
General numeric-to-Boolean assignments require an explicit `bool(...)` cast.
Boolean casts can also be used directly in conditions and inside logical
expressions, preserving short-circuit evaluation. These rules follow the
supported subset of the
[OpenQASM casting rules](https://openqasm.com/language/types.html#casting-specifics).

Whole-register Boolean casts, including `bool(register)` when `register` is a
`bit[1]` register, remain unsupported. The supported `bit[n](angle)` and
`angle[n](register)` casts are described under
[casts between angles and bit registers](#casts-between-angles-and-bit-registers).
Complex-to-Boolean casts,
Boolean arithmetic, and mixed Boolean/numeric comparisons remain unsupported;
explicitly cast Boolean values
to a numeric type before using them in these operations. Boolean array elements
and iterators follow these same rules. Block-local declarations remain unsupported.

### Quantum instructions and gate parameters

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
With a non-angle radian parameter, OpenQASM `gphase(angle)` is emitted as
`PHASE angle`. Fixed-point angle parameters use the binary contributions
described under [angle-valued global phase](#angle-valued-global-phase).

OpenQASM barriers accept scalar qubits and qubit-register selections. A
barrier without operands is also accepted. Because QCX execution preserves
instruction order, barriers emit no QCX instruction; their explicit operands
are nevertheless validated. Gate-power simplification does not move operations
across barriers.

### U-family phase conventions

QCX `U3` and the converter's `u3` alias implement the conventional matrix

$$
W(\theta,\phi,\lambda)=
\begin{pmatrix}
\cos(\theta/2)&-e^{i\lambda}\sin(\theta/2)\\
e^{i\phi}\sin(\theta/2)&e^{i(\phi+\lambda)}\cos(\theta/2)
\end{pmatrix}.
$$

The `u2(phi, lambda)` alias uses this matrix with `theta = pi / 2`.
These conventional aliases and the native QCX gates are unchanged.

The converter follows the published OpenQASM 3 built-in matrix for `U`:

$$
U(\theta,\phi,\lambda)=e^{i\theta/2}W(\theta,\phi,\lambda).
$$

It also follows the published mathematical definition of `cu`: the inactive
control block is identity, and the active block is
`exp(i * gamma) * U(theta, phi, lambda)`. Thus, lowering adds a global phase
of `theta / 2` for `U`, or separate phases of `gamma` and `theta / 2` on the
original control for ordinary/integer-powered `cu`. Their phase factors are
multiplied without forming the potentially inaccurate or overflowing angle
sum. See the [built-in matrix definition](https://openqasm.com/versions/3.0/language/gates.html#built-in-gates)
and the [standard-library mathematical definition of `cu`](https://openqasm.com/language/standard_library.html#gate-cu).

Upstream sources are inconsistent here. The example
[`stdgates.inc`](https://github.com/openqasm/openqasm/blob/main/examples/stdgates.inc)
defines `cu` using `p(gamma - theta / 2)` before controlled `U`, which instead
gives active block `exp(i * gamma) * W`. Its `u2`/`u3` phase corrections also
conflict with the stated OpenQASM 2 compatibility when combined with the
published built-in `U` matrix. The converter uses the explicit mathematical
`U` and `cu` definitions above and preserves conventional `u2`/`u3`; it does
not expand these inconsistent example-library definitions.

This corrects the previous converter behavior, which added
`(theta + phi + lambda) / 2` for `U` and
`gamma + (theta + phi + lambda) / 2` for `cu`. Regenerate affected QCX programs
from their OpenQASM sources. The correction matters even without controls for
absolute amplitudes; under controls it becomes an observable relative phase.

For example, `U(0, 0, pi)` is exactly `Z`, not `i * Z`. Both controlled
operations below leave the target in `|0>` and preserve the control's phase:

<!-- u-family-phase-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
ctrl @ U(0, 0, pi) q[0], q[1];
cu(0, 0, pi, 0) q[0], q[1];
// Final state: (|00> + |01>) / sqrt(2), with wire 0 on the right.
```

For ordinary, inverse, and constant/runtime integer-powered `cu`, lowering
uses `U1 control gamma`, `U1 control theta_half`, and the conventional `CU3`.
The half-angle uses private storage when needed; it does not overwrite theta
or gamma. Parameter expressions are evaluated once in source order and their
storage is retained across all sequence components and broadcast copies.

Keeping these corrections separate prevents two avoidable numerical failures:
`gamma = 1e20` can swallow `theta/2 = 0.35` in a combined sum, while
`gamma = theta = 1.5e308` gives an overflowing sum despite finite inputs.
Both phase factors can instead be evaluated separately by bra. Ordinary native
rounding and half-angle underflow remain possible; this is not arbitrary-
precision arithmetic. Regenerate previously converted QCX to obtain the fix.

<!-- cu-phase-factor-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
float theta = 0.7;
float gamma = 1e20;
h q[1];
cu(theta, 0, 0, gamma) q[1], q[0];
// theta and gamma are unchanged.
// The active control branch retains exp(i*gamma) * exp(i*theta/2),
// rather than losing theta/2 by first adding it to gamma.
```

The distinction is a relative phase between active and inactive control
subspaces, so checking measurement probabilities alone is insufficient.
The [floating-point power lowering](#controlled-u-powers) remains different:
principal powers must act on the full active eigenvalues, not on gamma and
the remaining factors separately.

Run the independent phase-factor references alone, or also execute the
default-double, non-MPI audit with bra:

```bash
python3 bra/test/qasm2qcx_cu_phase_numerical.py
python3 bra/test/qasm2qcx_cu_phase_numerical.py --bra /path/to/bra
```

### Inverse gate modifiers

`inv @` is supported on every currently supported built-in and standard-library
gate, and on `gphase`. It inverts the complete operation, not just its main QCX
instruction. An odd number of `inv` modifiers selects the inverse; an even
number cancels and preserves ordinary forward lowering. Each modifier is still
validated, so cancelling inversions cannot hide an unsupported modifier.

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
float theta = 0.7;
rx(theta) q;
inv @ rx(theta) q;
s q;
inv @ s q;
gphase(theta);
inv @ gphase(theta);
// The circuit returns exactly to |0>, including its global phase.
```

The converter uses explicit mappings to QCX adjoint mnemonics: `S` and `S+`,
`T` and `T+`, `SX` and `SX+`, `U1`/`U2`/`U3` and their `+` forms, and the
supported rotation and controlled-gate equivalents. For example, `inv @ sdg`
emits `S`, and `inv @ rx(theta)` emits `EX+` with the same converted half-angle
operand used by forward `rx`. Self-adjoint operations keep their accepted QCX
mnemonics (`H`, `X`, `Y`, `Z`, `I`, `SWAP`, and supported controlled forms);
the converter does not emit unsupported aliases such as `H+` or `CX+`.

Lowering reverses each multi-instruction sequence and adjoints its components.
The forward phase conventions above apply to the full sequence:

- Built-in `U(theta, phi, lambda)` lowers to `PHASE phase` followed by `U3`,
  where `phase = theta / 2`. Its inverse emits `U3+` followed
  by `PHASE -phase`, with the same three gate parameters.
- `cu(theta, phi, lambda, gamma)` lowers to control-qubit `U1 gamma`,
  control-qubit `U1 theta_half`, then `CU3`, where `theta_half = theta / 2`.
  Its inverse emits `CU3+`, control-qubit `U1+ theta_half`, then
  control-qubit `U1+ gamma`, keeping both phase operands unchanged.
- For non-angle parameters, `inv @ gphase(angle)` emits `PHASE` with the negated angle. Runtime
  negation uses private `REAL` storage, preserving the source variable and
  avoiding signed-integer negation overflow.

Parameters retain the existing expression, type, angle-conversion, and include
rules. Runtime parameter expressions are evaluated once per gate invocation,
with their temporaries kept reserved through all emitted components. Broadcasting
repeats the complete inverse sequence for each selected operand tuple, including
its phase correction. For example:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit control;
qubit[2] targets;
float theta = 0.3;
h control;
cu(theta, 0.4, 0.5, 0.6) control, targets;
inv @ cu(theta, 0.4, 0.5, 0.6) control, targets;
inv @ inv @ h control;
// The circuit returns exactly to |000>.
```

Inverse-modified gates work inside branches and all supported loops, including
`break` and `continue`. Runtime computations and access checks stay on their
execution paths; skipped bodies do not evaluate their runtime parameters.
Modifier syntax is validated even inside empty enclosing loops, without
evaluating unsupported modifier arguments. Successful and failed parameter
lowering release gate-owned temporaries without releasing caller-owned storage.

`inv` can be combined with `ctrl` and `negctrl` as described below. User-defined
gates, gate durations, arguments to `inv`, and qubit operands on uncontrolled
`gphase` remain unsupported. Modifier support does not relax gate arity, operand
selection, broadcasting, or parameter-type restrictions.

### Control gate modifiers

`ctrl @`, `ctrl(n) @`, `negctrl @`, and `negctrl(n) @` are supported on every
currently supported built-in and standard-library gate, and on `gphase`.
Added controls are prepended to the original quantum operands. A `ctrl` group
requires all its wires to be one; a `negctrl` group requires all its wires to
be zero. All groups must match for the original gate to act. For example:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit control;
qubit target;
h control;
ctrl @ x control, target;
// The state is (|00> + |11>)/sqrt(2).
```

The default count is one. An explicit count must be a positive compile-time
`int` or `uint` expression, including previously declared integer constants and
supported explicit integer casts. Boolean or floating-point counts require an
explicit integer cast; runtime variables, array reads, forward references, and
loop-iterator dependencies are rejected. Even an expanded constant-loop iterator
cannot supply a control count. Repeated control modifiers add their counts:
`ctrl @ ctrl(2) @ x` adds three positive controls, and
`negctrl(2) @ ctrl @ x` adds two negative controls followed by one positive
control. Individual and summed counts must not exceed QCX `INT_MAX`; a gate's
existing controls also count toward the native
mnemonic's limit. The resulting quantum operand count must match exactly.

The converter uses native controlled and multi-controlled QCX gate families.
For example, `ctrl @ h` emits `CH`, `ctrl(2) @ x` emits `C2X`, and
`ctrl @ cx` also emits `C2X`, preserving added controls before the original
control and target. Standard-library adjoint gates such as `sdg` use native
forms such as `CS+`. Controlled identity uses `CI1` or `C<n>I1`, explicitly
identifying its one identity target. Compact count notation avoids constructing
a string with one `C` per control.

Negative controls use QCX `wire#0` operands; positive controls keep their bare
wire spelling. No X gates are inserted to emulate a negative control. Modifier
order determines control operand order, even when `inv` is interleaved:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
negctrl @ h q[3], q[1];                    // CH 3#0 1
ctrl @ negctrl @ x q[3], q[0], q[2];       // C2X 3 0#0 2
negctrl @ ctrl @ x q[3], q[0], q[2];       // C2X 3#0 0 2
negctrl @ inv @ sx q[3], q[1];            // CSX+ 3#0 1
```

`inv` commutes with control modifiers: inversion preserves their required
values and reverses/adjoints the complete underlying sequence. Repeated `inv`
modifiers cancel in pairs without removing any controls. The `#` suffix belongs
to QCX, not OpenQASM syntax; runtime qubit indices remain unsupported.

Controlling an operation conditions its entire lowered sequence, including
every phase correction. The forward phase conventions described above apply:

- `ctrl @ U(theta, phi, lambda) c, t` emits `U1 c phase` and
  `CU3 c t theta phi lambda`, with `phase = theta / 2`.
  The correction is a phase on the added control, not a global `PHASE`.
- `ctrl @ cu(theta, phi, lambda, gamma) extra, c, t` emits
  `CU1 extra c gamma`, `CU1 extra c theta_half`, and
  `C2U3 extra c t theta phi lambda`, where `theta_half = theta / 2`.
  Both corrections require the added and original controls to be one.
- For non-angle parameters, `ctrl @ gphase(angle) c` emits `U1 c angle`; it needs no extra target.
  With `n` controls, the converter emits a `U1` conditioned on the other
  `n - 1` control operands, so only the all-one basis component acquires the phase.

The same phase conditioning applies to negative and mixed controls. For
example, `negctrl @ U(1, 2, 3) c, t` emits `U1 c#0 0.5` and
`CU3 c#0 t 1 2 3`. Its inverse emits `CU3+ c#0 t 1 2 3` followed by
`U1 c#0 -0.5`, negating the phase correction exactly once. For `cu`, added
negative controls do not change the original positive control's polarity.
For a non-angle parameter, `negctrl @ ctrl @ gphase(angle) a, b` emits `CU1 a#0 b angle`: only the
component with `a=0` and `b=1` acquires the phase, with no additional target.

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] controls;
const uint N = 2;
float theta = 0.4;
h controls;
ctrl @ ctrl(N) @ gphase(theta) controls[0], controls[1], controls[2];
// Only the |111> amplitude acquires exp(i * theta).
```

Scalar qubits, static selections, and complete registers follow the existing
broadcasting rules: all register operands must have equal lengths, and scalar
operands repeat across those selections. Broadcasting repeats the complete
controlled sequence for each tuple. Qubits must be distinct within each emitted
controlled operation; added controls cannot alias original controls or targets.
Execution remains subject to backend resources, including MPI local-qubit limits.

Runtime parameter expressions are evaluated once per invocation and remain
reserved through every component and broadcast copy. An inactive quantum
control (positive in `|0>`, or negative in `|1>`) does **not** skip classical
parameter evaluation or bounds/divisor checks. In contrast, an unexecuted
classical branch or loop body skips its runtime
parameter computations. Controlled gates work in all supported loops, including
`break` and `continue`. Modifier kinds, constant counts, and modified gate arity
are validated even inside empty enclosing loops. Successful and failed lowering
release gate-owned temporaries while preserving caller-owned storage.

User-defined gates, gate durations, and runtime-dependent control counts
remain unsupported. No new QCX mnemonic is introduced: the native control
families use the `#0` operand suffix to select zero-controls.

### Integer-power gate modifiers

`pow(k) @` is supported on every currently supported gate and on `gphase`
when `k` is a supported constant or runtime integer expression. Previously
declared `const int` and `const uint` values, scalar `int`, `uint`, and supported
`uint[n]` variables, integer array elements and loop iterators, supported integer
arithmetic, and explicit integer casts may supply the exponent. Boolean and
complex exponents require an explicit supported numeric cast. Floating-point
exponents follow the separate, narrower subset described below.

Positive powers apply the complete gate repeatedly. Negative powers apply its
inverse repeatedly. Exponent zero applies the identity, but does not
bypass validation of modifiers, gate availability, arity, parameter types,
qubit operands, or broadcasting. Parameters retain their usual evaluation and
checks, including for zero powers and inactive quantum controls. An unexecuted
classical branch or loop body skips runtime parameter evaluation as before.

Nested powers multiply their integer exponents; each `inv` changes the sign.
Integer powers commute with adding controls. Control groups retain their
source order and polarities regardless of interleaved `pow` and `inv`
modifiers. Every exponent and control count is checked, including when an
outer zero power would otherwise discard the operation.

These product/sign simplifications apply only to integer-only modifier chains.
If any exponent has floating-point type (even `2.0`), modifier order is preserved
instead, as described under [constant floating-point powers](#constant-floating-point-gate-powers).
Fixed-point angle parameters use the
[whole-angle evaluator](#floating-powers-of-angle-valued-phase-gates).

For angle-valued phase gates, including `gphase`, integer powers instead
multiply the stored bits modulo `2^n`, with no exponent-sized repetition loop;
see [angle-valued phase gates](#angle-valued-phase-gates). The scaling and
repetition details below concern other, non-angle parameters.

The converter simplifies exact unitary identities before bounding expansion:

| Gate family | Reduction |
| --- | --- |
| `id` | No quantum instruction |
| `x`, `y`, `z`, `h`, `swap`, and their supported controlled forms | Exponent modulo 2 |
| `s`, `sdg` | Exponent modulo 4, using `S`, `Z`, or `S+` |
| `t`, `tdg` | Exponent modulo 8, using existing gates or a two-gate sequence |
| `sx` | Exponent modulo 4, using `SX`, `X`, or `SX+` |

These identities retain global phase, so they also hold with added positive
or negative controls. Distinct-wire validation still runs when a gate reduces
to identity. Periodic reductions do not rely on a floating-point approximation
of the exponent, and large nested powers need not generate large circuits.

For non-angle radian parameters, constant angles are scaled for
`p`/`phase`/`u1`, `rx`/`ry`/`rz`, their supported
controlled forms, and `gphase`, when the multiplier is exactly representable
as a Python float and multiplication produces an exact, finite host-float
result. Inexact integer-to-float conversion is not introduced by this
optimization. A rounded product of a large angle could otherwise change its
phase arbitrarily modulo `2*pi`, despite remaining finite.
Runtime-valued non-angle parameters retain repetition: scaling an unknown angle
could overflow even when every original application has a finite angle.
Unrepresentable multipliers and rounded or overflowing literal products also
fall back to repetition. General `U`, `u2`, `u3`, and `cu` parameters are not scaled.

<!-- integer-power-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
const int N = 3;
qubit[3] q;
pow(2) @ x q[0];                             // identity: no instruction
pow(-2) @ s q[0];                            // Z 0
pow(N) @ t q[0];                             // S 0; T 0
negctrl @ pow(2) @ sx q[1], q[2];            // CX 1#0 2
pow(-3) @ rz(0.5) q[2];                      // EZ+ 2 -0.75
pow(2) @ gphase(0.25);                       // PHASE 0.5
```

The fallback repeats the entire lowered sequence for each broadcast operand
tuple. Built-in `U` includes its global-phase correction on every repetition;
`cu` includes its control-phase correction. Inversion reverses and adjoints
the complete sequence before repetition. Runtime parameter expressions are
evaluated once per source application, with private temporaries reserved
through every component, repetition, and broadcast copy.

#### Runtime exponents

Each runtime exponent expression is evaluated once per source application,
in modifier order, before gate parameter evaluation. Its value is copied to
private storage and preserved across repetitions and broadcasting. All factors
are evaluated even if another factor is zero. Parameter checks also execute for
zero powers and inactive quantum controls. Skipped classical branches and loop
bodies skip runtime expression evaluation, but do not bypass conversion-time
type and operand validation.

The converter uses unsigned magnitudes and separate sign parity, so signed
`INT_MIN` and the full native `UINT` range are supported without signed
negation overflow. For gates outside the fixed-point angle path, nested factors
use separate QCX loops instead of multiplying full-width magnitudes.
All zero magnitudes are checked before entering any
repetition loop. Source variables are not modified.

For the exact periodic gate families above, runtime factors are reduced modulo
the effective period before multiplication. Only small residues are multiplied,
avoiding overflow and bounding repetitions independently of exponent magnitude.
Other gates repeat the complete lowered operation, preserving the global and
control-phase corrections of `U` and `cu`. Runtime exponents are never used to
scale non-angle radian parameters; constant factors may still use the constant-angle optimization
described above.

In mixed constant/runtime nonperiodic powers that require repetition counters,
each individual constant factor's magnitude must fit native `UINT`. Their
product need not fit: separate loops preserve it without overflow. Larger
individual constants can still be accepted when periodic simplification or
safe constant-angle scaling eliminates the need for such a counter.

The two powered operations below cancel, including their controlled phase:

<!-- runtime-integer-power-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
int n = -2;
uint[3] m = 3;
float theta = 0.2;
h q;
negctrl @ pow(n) @ pow(m) @ U(theta, 0.4, 0.6) q[2], q[0];
inv @ negctrl @ pow(n) @ pow(m) @ U(theta, 0.4, 0.6) q[2], q[0];
```

Generated code size does not grow with runtime exponent values, but execution
time can: nonperiodic powers have no arbitrary runtime iteration limit.

After simplification, `MAX_GATE_POWER_OUTPUT_LINES` limits the generated
repetition instructions from each source statement containing `pow` to 100,000
QCX lines, including all broadcast elements, phase corrections, and generated
repetition control flow. On repetition paths, exponent capture and parameter
preparation precede this budget. Fixed-point angle paths use the same limit
for parameter preparation and binary phase-contribution code, including bit
tests. The converter checks the budget before allocating a repeated
instruction list.
Enclosing loop-output budgets also apply. Exceeding a limit reports a
conversion error; the CLI emits no partial QCX program. This limit does not
change the behavior of statements without `pow`.

Scalar arithmetic exponentiation (`**`) is a separate feature described above.
No new QCX instruction is required.

### Constant floating-point gate powers

Finite compile-time floating-point exponents are supported on the following
fixed gates and on the parameterized phase, rotation, and U-family gates
described below.
The fixed gates require `include "stdgates.inc";`.

Scalar-angle parameters of phase gates use
[floating powers of angle-valued phase gates](#floating-powers-of-angle-valued-phase-gates),
with exact turn arithmetic and native final rounding.

| Gate family | Nonzero principal eigenphase before modifiers |
| --- | --- |
| `id` | None: identity |
| `x`, `y`, `z`, `h`, `swap` | `pi` |
| `s`, `sdg` | `pi / 2`, `-pi / 2`, respectively |
| `t`, `tdg` | `pi / 4`, `-pi / 4`, respectively |
| `sx` | `pi / 2` |

Intrinsic controlled forms `cx`/`CX`, `cy`, `cz`, `ch`, `ccx`, and `cswap`
are supported too. Additional `ctrl` and `negctrl` modifiers, including mixed
polarities and constant counts, act on the full powered operation. The other
eigenphases are zero, including those of inactive control subspaces.

Exponents may use literals, previously declared `const float` values, supported
constant arithmetic, and explicit numeric casts. For example, `pow(1.0 / 2)`
and `pow(float(2))` are supported. A `float` variable initialized to a literal
is still runtime-valued and is not accepted. NaN and infinities, including
overflowing constant expressions, are rejected. Positive and negative zero
are accepted.

#### Spectral convention and modifier order

For a gate matrix

$$
G=V\mathrm{diag}(e^{i\theta_j})V^\dagger,
$$

choose each eigenphase in `(-pi, pi]`, assigning `+pi` to eigenvalue `-1`.
The floating-point power is

$$
G^r=V\mathrm{diag}(e^{ir\theta_j})V^\dagger.
$$

This branch convention is an explicit converter policy. The OpenQASM
[power-modifier specification](https://openqasm.com/versions/3.0/language/gates.html#power-modifier)
permits floating-point exponents but does not explicitly prescribe this branch.
Global phase is retained, not discarded as an unobservable equivalence.

Modifiers act from the innermost (rightmost) outward. Each power selects the
principal eigenphases of its input operation again. Nested powers are not
generally flattened, and inversion is not generally absorbed into an exponent:

| Modifier chain on `x` | Result |
| --- | --- |
| `pow(0.5) @` | `sx` |
| `pow(0.5) @ pow(2) @` | Identity: `(x^2)^0.5` |
| `pow(2) @ pow(0.5) @` | `x`: `(x^0.5)^2` |
| `pow(0.5) @ inv @` | `sx`: inverse `x` is still `x` |
| `inv @ pow(0.5) @` | Inverse `sx` |

An integral-valued float such as `2.0` still selects ordered spectral handling;
an explicit `int(2.0)` cast supplies an integer exponent instead. Integer-only
chains retain their existing product/sign rules.

#### Lowering, validation, and limits

For the fixed families, the converter uses existing QCX gates for a basis
change, a diagonal phase, and the inverse basis change. Added and intrinsic controls condition only the
diagonal phase; the unconditional basis changes cancel in inactive subspaces.
For `swap`, a `CX`/`H` basis change isolates the antisymmetric subspace, on which
the diagonal controlled phase acts. No arbitrary-matrix instruction or new bra
feature is required. Result names such as `sx` in the table describe matrix
equivalence, not a promise to emit that particular mnemonic.

Fixed-family eigenphase calculations use exact rational multiples of `pi` and
the exact binary value of each evaluated host float. Only the final QCX angle
is rounded using the host float representation of `pi`; basis-change angles and backend
gate execution also have ordinary floating-point rounding. Declared floating
widths are not newly enforced. Generated circuit size is independent of
exponent magnitude. The existing gate-power and enclosing loop-output budgets
apply, including all broadcast copies.

Zero and identity powers still validate modifier arguments, includes, gate
arity, qubit operands, distinct wires, and broadcasting. Static literal operand
checks also run in empty loops; named or compound selections in constant
unrolled loops are checked when an iteration is expanded. Supported powers
work in branches, loops, and register broadcasting. Fixed-family lowering
requires no runtime exponent evaluation or additional classical scratch storage;
parameterized phase, rotation, and U-family powers use the runtime preparation
described below.

The following remain unsupported, including when an outer power is zero:

- runtime floating-point exponents;
- runtime integer factors or loop-iterator factors in a chain containing a
  floating-point power, even if the loop is unrolled.

These are explicit conversion errors, not approximate angle-scaling fallbacks.
The CLI emits no partial QCX program on failure.

<!-- floating-power-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
const float HALF = 0.5;
qubit[3] q;
pow(HALF) @ pow(2) @ x q[0];               // identity
pow(2) @ pow(HALF) @ x q[1];               // X on wire 1
negctrl @ pow(HALF) @ x q[0], q[2];
inv @ negctrl @ pow(HALF) @ x q[0], q[2];   // cancels preceding operation
// Final state: |010>, including its phase, with wire 0 on the right.
```

#### Parameterized phase gates

Floating-point powers also support `p(angle)`, `phase(angle)`, `u1(angle)`,
`cp(angle)`, `cphase(angle)`, and `gphase(angle)`. The named standard-library
gates require `include "stdgates.inc";`; built-in `gphase` does not. Additional
`ctrl` and `negctrl` modifiers may be interleaved with powers and inversions.
Intrinsic controls on `cp`/`cphase` remain positive controls.

Scalar `angle` parameters use the
[whole-angle evaluator](#floating-powers-of-angle-valued-phase-gates).
The following description concerns radian-valued `float`, `int`, and `uint`
parameters.

The angle may be a supported scalar `float`, `int`, or `uint` expression,
including runtime variables, indexed array elements, and supported loop
iterators. Bare Boolean, bit, complex, and whole-array parameters are rejected;
explicit casts must follow the existing scalar-cast rules. The angle is
evaluated once per reached source application and copied to private storage
before processing the modifiers. Broadcasting reuses that snapshot, and source
variables and array elements are not modified.

The converter forms `exp(i * angle)` with QCX `APPLY ... <- EXP ...`, then
uses `APPLY ... <- ARG ...` to select its principal argument before every
power. It does not simply scale the original angle or reduce it modulo a
machine approximation of `2*pi`. For example, the principal square root of
`p(3*pi/2)` has nontrivial phase `-pi/4`, not `3*pi/4`.
Nested powers select the current coefficient's principal argument again;
inversion negates the current angle without commuting past a power.

An exactly negative-real coefficient uses `+PI` for either sign of imaginary
zero. Nearby coefficients are not tolerance-snapped to the cut, even when
their argument rounds to machine `-PI`. For example, the usual binary64
approximation of the angle `-pi` produces a coefficient with a tiny negative
imaginary component. Its principal root can differ from that of the exact
fixed `z` gate. This is a numerical distinction in the evaluated coefficient,
not a different branch policy for the two families.

All exponent expressions must be compile-time constants. In a chain containing
a floating-point power, each integer factor is promoted separately to a host
Python float and must fit its finite range; an oversized integer factor is a
conversion error, including in an empty loop or behind an outer zero power.
Factors are never multiplied together during conversion. Folded numeric angle
literals are also serialized using host-float precision; runtime arithmetic and
generated phase preparation use the backend's native `REAL` precision.
Declared OpenQASM floating-point widths are not newly enforced.

`EXP` checks the initial angle and each intermediate modified phase. A nonfinite
angle or overflowing intermediate product is a runtime error, including under
an inactive quantum control or when an outer power is zero. This deliberately
rejects some mathematically finite powers with enormous intermediate products;
extended-range multiplication is separate work. Underflow follows the backend's
floating-point environment. Reached failures occur before the powered quantum
operation is emitted to the state; the CLI currently leaves native `APPLY`
exceptions uncaught.

Skipped classical branches and loop bodies do not execute angle preparation or
its runtime checks. Conversion-time validation of modifier arguments, parameter
types, names, gate arity, and supported qubit operands still applies. Only
temporary declarations may be hoisted out of a skipped body. The existing
integer-only and fixed-family power paths are unchanged.

The generated program uses `REAL` and `COMPLEX` temporaries, `APPLY`, component
extraction, and ordinary QCX branches; no arbitrary-matrix instruction is
required. This gate-power lowering is separate from the scalar OpenQASM
function calls described above. Use a bra version with `APPLY`,
`EXP`, and `ARG` support; see
[classical functions in bra](../../docs/bra.md#classical-function-invocation).
There is no repetition proportional to the exponent's magnitude. For this
parameterized floating-power path, `MAX_GATE_POWER_OUTPUT_LINES` also counts
the generated phase preparation, including all broadcast gate instructions.
Enclosing loop-output budgets still apply.

<!-- parameterized-phase-power-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
const float HALF = 0.5;
qubit[3] q;
float theta = 3 * pi / 2;
h q;
pow(HALF) @ p(theta) q[0];                 // nontrivial phase is -pi/4
inv @ pow(HALF) @ p(theta) q[0];
negctrl @ pow(HALF) @ cp(theta) q[2], q[1], q[0];
inv @ negctrl @ pow(HALF) @ cp(theta) q[2], q[1], q[0];
pow(HALF) @ gphase(theta);
inv @ pow(HALF) @ gphase(theta);
h q;
// Final state: |000>, including its global phase; theta is unchanged.
```

The independent phase-power policy and numerical audit can be run with:

```bash
python3 bra/test/qasm2qcx_phase_power_numerical.py
python3 bra/test/qasm2qcx_phase_power_numerical.py --bra /path/to/bra
```

The second command uses a default-double, non-MPI executable. It checks actual
complex amplitudes without aligning away global phase, including branch-cut
boundaries, very large angles, nested modifiers, mixed controls, broadcasting,
arrays, classical control flow, fusion, and runtime failures. The reference-only
command does not require the OpenQASM parser or a backend executable.

#### Parameterized Pauli rotations

Floating-point powers support `rx(theta)`, `ry(theta)`, `rz(theta)`, and their
intrinsic controlled forms `crx(theta)`, `cry(theta)`, and `crz(theta)`.
These gates require `include "stdgates.inc";`. Additional `ctrl` and `negctrl`
modifiers, including mixed polarities and constant counts, may be interleaved
with powers and inversions. Intrinsic controls remain positive.

For Pauli matrix `P`, the rotation has the spectral decomposition

$$
R_P(\theta)=e^{-i\theta/2}\frac{I+P}{2}
             +e^{i\theta/2}\frac{I-P}{2}.
$$

The converter tracks both eigenphases independently and selects each
coefficient's principal argument before every power. It does not infer one
principal phase by negating the other: if both coefficients are exactly `-1`,
both select `+PI`, so their principal square roots are both `i`. Nor does it
simply replace `theta` with `r * theta`. For example, the principal square root
of `rz(3*pi)` differs from `rz(3*pi/2)` by a global factor of `-1`. Under quantum
controls this becomes a relative phase and cannot be discarded. Numerical
coefficients at machine approximations of `+/-2*pi` need not be exactly `-1`;
the same exact-cut and no-tolerance-snapping rules as for phase gates apply.

The angle expressions, compile-time exponent restrictions, once-only snapshot,
source preservation, broadcasting, precision, runtime finite checks, and
classical-path rules are the same as for parameterized phase gates above.
In particular, the original `theta` is checked before halving, and each modified
eigenphase is checked before the quantum operation, even with an outer zero
power or inactive quantum controls. Skipped classical paths do not execute
these runtime checks. Integer-only rotation powers retain their existing
lowering.

Lowering uses existing QCX basis changes and two separate `U1`-family phases:

| Rotation | Basis change before phases | Basis change after phases |
| --- | --- | --- |
| `rx` / `crx` | `H` | `H` |
| `ry` / `cry` | `S+`, then `H` | `H`, then `S` |
| `rz` / `crz` | None | None |

In that basis, the `P=+1` phase acts on target value zero (`wire#0`) and the
`P=-1` phase acts on target value one. All intrinsic and added controls condition
both projector phases; the basis changes are unconditional and cancel in
inactive control subspaces. Both phases are retained instead of subtracting
them: two finite eigenphases can have an overflowing difference, which is not
a reason to reject the operation. No new bra instruction is required.
The existing output budgets include phase preparation and all broadcast copies.
Fusion is supported with the same lowering.

<!-- rotation-power-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
const float HALF = 0.5;
qubit[3] q;
float theta = 3 * pi;
h q;
pow(HALF) @ rz(theta) q[0];
rz(-theta / 2) q[0];                       // together: -I, not I
negctrl @ pow(HALF) @ cry(theta) q[2], q[1], q[0];
inv @ negctrl @ pow(HALF) @ cry(theta) q[2], q[1], q[0];
h q;
// Final state: -|000>, including its global phase; theta is unchanged.
```

The independent rotation references and numerical audit can be run with:

```bash
python3 bra/test/qasm2qcx_rotation_power_numerical.py
python3 bra/test/qasm2qcx_rotation_power_numerical.py --bra /path/to/bra
```

The reference-only command uses Python's standard library and needs neither
the parser nor bra. The second command uses a default-double, non-MPI bra
executable with `APPLY`, `EXP`, `ARG`, negative controls, and fusion support.
It compares actual complex amplitudes without aligning away global phase.

#### Parameterized U-family gates

Floating-point powers support `u2(phi, lambda)`, `u3(theta, phi, lambda)`, and
built-in `U(theta, phi, lambda)`. Controlled `cu` powers are described
[below](#controlled-u-powers). The aliases require `include "stdgates.inc";`;
built-in `U` does not. Powers may be interleaved with `inv`, `ctrl`, and
`negctrl`, including mixed control polarities and constant counts.

These gates retain the [existing U-family phase conventions](#u-family-phase-conventions):
`u3` is conventional `W`, `u2` fixes its `theta` to `pi/2`, and built-in `U`
is `exp(i * theta/2) * W`. Its extra phase is part of both eigenvalues; it is
not factored out and powered separately. Principal powers do not generally
distribute over such a factorization.

The shared analytic decomposition uses

$$
t=\frac{\theta}{2},\qquad
a=\frac{\phi+\lambda}{2},\qquad b=\frac{\phi-\lambda}{2},
$$

$$
h=\cos t\cos a,\qquad
\boldsymbol v=(-\sin t\sin b,\ \sin t\cos b,\ \cos t\sin a),
\qquad q=\|\boldsymbol v\|.
$$

Writing `W = exp(i*a) * (h*I - i*v.Pauli)`, the full eigenvalues are

$$
\mu_\pm=e^{i\delta}(h\mp iq),\qquad
\delta=\begin{cases}a&\text{for u2 and u3},\\a+t&\text{for built-in U}.\end{cases}
$$

These sums describe the mathematics, not the runtime arithmetic. Preparation
multiplies separate half-angle unit factors instead of adding or subtracting
large source angles. `ABS` computes the vector magnitudes without directly
squaring tiny or large components. `ARG` supplies the eigenbasis angles and
selects each full eigenvalue's principal argument before every power. Inversions
conjugate the current eigenvalue coefficients directly, avoiding a reconstruction
from rounded arguments that could change a subsequent branch-cut decision.

For nonzero `q`, a conventional `u3(beta, eta, 0)` supplies the eigenbasis,
with `eta = atan2(v_y, v_x)` and
`beta = atan2(hypot(v_x, v_y), v_z)`. These formulas are implemented through
`ABS` and `ARG`, not new `ATAN2` or `HYPOT` instructions. An exactly scalar
gate uses the identity basis and two equal full eigenvalues. A Z-axis vector
uses `eta = 0`. Both cases avoid `ARG(0)` and division by a small eigenvalue
gap. There is no epsilon-based merging or branch-cut snapping.

Chronologically, lowering emits `U3+`, separate `U1`-family phases on target
values zero and one, then `U3`. Added controls condition both phases; the basis
changes remain unconditional and cancel in inactive control subspaces. No
global phase is discarded, and no difference between the two eigenphases is
formed. Fusion uses the same lowering, without new quantum instructions.

Each original numeric scalar parameter is evaluated and checked once, from
left to right, before any halving or spectral construction. Supported `float`,
`int`, and `uint` expressions include indexed array elements and loop iterators;
bare Boolean, bit, complex, and whole-array parameters are rejected. Private
snapshots preserve source variables and are shared across broadcast copies.
The constant-factor restrictions, native precision and underflow behavior,
runtime failure rules, skipped-classical-path behavior, and output budgets
are the same as for parameterized phase powers. Checks still run under inactive
quantum controls or an outer zero power; skipped classical paths do not run
them. Budgets include preparation, its private declarations, and all broadcast
gate copies. Integer-only powers remain unchanged.

Near a degeneracy at the negative-real branch cut, arbitrarily close gates can
have very different principal roots. Floating-point preparation determines the
evaluated eigenvalues and hence their branches; it does not promise exact-real
arithmetic at such boundaries. Very large exponents can also magnify ordinary
phase roundoff. Do not use an epsilon to turn these cases into scalar gates or
merge their eigenphases. Use a bra version supporting `APPLY`, `EXP`, `ARG`, and
complex `ABS`.

<!-- u-family-power-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
const float HALF = 0.5;
qubit[3] q;
float theta = 0.7;
float phi = 0.4;
float lam = -0.2;
h q;
pow(HALF) @ u2(0, pi) q[0];
pow(HALF) @ u2(0, pi) q[0];               // two roots give conventional H
h q[0];                                 // cancels that H
negctrl @ pow(HALF) @ ctrl @ u3(theta, phi, lam) q[2], q[1], q[0];
inv @ negctrl @ pow(HALF) @ ctrl @ u3(theta, phi, lam) q[2], q[1], q[0];
pow(2.0) @ U(pi / 2, 0, pi) q[0];        // i*I, not I
h q;
// Final state: i|000>; theta, phi, and lam are unchanged.
```

The independent references and default-double, non-MPI audit can be run with:

```bash
python3 bra/test/qasm2qcx_u_power_numerical.py
python3 bra/test/qasm2qcx_u_power_numerical.py --bra /path/to/bra
```

The reference-only command needs only Python's standard library. The backend
audit checks actual complex amplitudes, retaining their absolute phase.

#### Controlled U powers

Floating-point powers also support `cu(theta, phi, lambda, gamma)` with
`include "stdgates.inc";`. Its active block is `exp(i*gamma) * U`, where `U`
is the built-in gate, not conventional `u3`. The intrinsic control is positive;
added `ctrl` and `negctrl` operands precede it in source order. For example,
`negctrl @ ctrl @ pow(0.5) @ cu(...) a, b, c, t` phases the active block only
when `a=0`, `b=1`, and `c=1`.

Let `mu+` and `mu-` be the full built-in `U` eigenvalues above. The active
eigenvalues and a single principal power are

$$
\nu_\pm=e^{i\gamma}\mu_\pm,\qquad
\mathrm{cu}^{\,r}=\mathrm{diag}\!\left(I_2,
 V\,\mathrm{diag}\!\left(e^{ir\mathrm{Arg}(\nu_+)},
 e^{ir\mathrm{Arg}(\nu_-)}\right)V^\dagger\right).
$$

Gamma rotates both full eigenvalues before their principal arguments are
selected. Powering gamma separately, or scaling all four source parameters,
is not equivalent. The eigenbasis `V` is unchanged from built-in `U`.
Ordered nested powers reselect the current eigenvalues' principal arguments;
inversions and interleaved controls follow the same rules as other U-family
powers. Inactive intrinsic and added control subspaces remain identity,
including when the active block is scalar: its common phase is a relative
controlled phase and cannot be dropped.

Preparation captures all four numeric parameters once, left to right, and
validates their original values before any halving. Gamma's full-angle unit
factor multiplies both built-in `U` eigenvalues before any power is applied.
It is never halved or added to `theta/2`, so large finite source angles do not
require a finite or accurately representable combined phase sum. Both full
eigenphases use `U1`-family gates conditioned on the intrinsic control and
every added control; the target's unconditional `U3+`/`U3` basis changes cancel
on inactive subspaces. The same sequence supports fusion.

Numeric parameter types, indexed arrays, broadcast snapshots, loop support,
constant-exponent restrictions, runtime checks, output budgets, and native
precision/branch-boundary limitations are the same as for the other U-family
powers. Nonfinite parameters or intermediate phase products cannot be hidden
by an outer zero power or inactive quantum controls; skipped classical paths
do not evaluate them. Ordinary and integer-only `cu` use separate control-phase
corrections for gamma and `theta/2`, as described under
[U-family phase conventions](#u-family-phase-conventions). That sequence is
not used to distribute floating-point powers over those factors; full active
eigenvalues must be formed before selecting their principal arguments.

<!-- cu-power-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
const float HALF = 0.5;
qubit[3] q;
float theta = 0.7;
float gamma = -0.75 * pi;
h q;
pow(HALF) @ cu(0, 0, 1.5 * pi, gamma) q[1], q[0];
negctrl @ pow(HALF) @ cu(theta, 0.4, -0.2, gamma) q[2], q[1], q[0];
inv @ negctrl @ pow(HALF) @ cu(theta, 0.4, -0.2, gamma) q[2], q[1], q[0];
// The last two gates cancel; theta and gamma are unchanged.
// Final amplitudes are 1/sqrt(8) times:
//   1                         if q[1] = 0;
//   exp(-i * 3*pi/8)           if q[1] = 1 and q[0] = 0;
//   exp(+i * 3*pi/8)           if q[1] = 1 and q[0] = 1.
```

In this example, gamma moves the active eigenvalues across the principal
branch before the square root is taken. Separately powering gamma and `U`
would give the wrong sign for one active eigenvalue. The independent
references and default-double, non-MPI audit can be run with:

```bash
python3 bra/test/qasm2qcx_cu_power_numerical.py
python3 bra/test/qasm2qcx_cu_power_numerical.py --bra /path/to/bra
```

The reference-only command uses Python's standard library, without the parser
or bra. The backend audit needs the same native features as U-family powers
and compares complex amplitudes without removing global phase.

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

Qubit indices, range bounds, range steps, and discrete indices must be signed
integer literals outside constant loop scopes, except for the constant-expression
qubit selectors supported by runtime index-set and range measurement destinations above.
Bit selections also accept
supported constant `int` and `uint` expressions, including named constants and
casts. Both range bounds must be present. A zero step, an empty
range, or an out-of-bounds index is rejected during conversion.

## Classical control flow

The converter supports `if`/`else` statements with comparisons, direct scalar
bit and Boolean conditions, and logical operators. The supported comparison
operators are `==`, `!=`, `>`, `<`, `>=`, and `<=`. OpenQASM `!=` is emitted as the QCX
not-equal operator `\=`.

Conditions can compare scalar `int`, `uint`, `float`, and `bit` expressions.
A bit-register, signed/unsigned integer-array, floating-point-array, or
Boolean-array element with a supported static or runtime index is also accepted.
Compatible integer and floating-point operands are promoted when necessary. Complex
values cannot be compared. Equal-width bit-register comparisons
are supported as described in [Bit-register equality](#bit-register-equality).
Boolean and scalar bit operands can be compared with one another using `==`
or `!=`.

A scalar `bit`, a statically or runtime-indexed element of a bit register, or a
Boolean literal, constant, variable, array element, or iterator can also be used
directly as a condition: zero is false and one is true. Logical negation
`!` and logical combinations `&&` and `||` can be applied recursively to these
conditions and supported comparisons. Parentheses can group conditions.

Logical AND and OR use short-circuit evaluation from left to right: `&&` skips
its right operand when the left operand is false, and `||` skips its right
operand when the left operand is true. For example:

```qasm
if (ready && count > 0) {
    x q;
}
if (!(flags[0] || flags[1])) {
    reset q;
}
```

Here `ready` is a scalar bit or Boolean, `flags` is a bit register, and `count`
is an integer variable, all declared before these statements. A whole bit
register,
including `bit[1]`, is distinct from a scalar bit and cannot be used directly
as a condition; select its element explicitly, as in `flags[0]`. Direct scalar
bit conditions follow the rules in the
[OpenQASM live specification](https://openqasm.com/language/types.html#classical-bits-and-registers).

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
measurements, resets, assignments, barriers, nested branches, and supported
`for` and `while` loops.

Comparison and logical expressions also produce Boolean values, following the
[OpenQASM classical instruction rules](https://openqasm.com/language/classical.html#comparison-boolean-instructions).
They can initialize Boolean variables and constants or be assigned to Boolean
variables. For example:

```qasm
const bool enabled = 2 > 1;
int count = 1;
bool ready = enabled && count > 0;
ready = !ready;
```

Runtime expressions are lowered to branches assigning zero or one to a
temporary, followed by assignment to the destination. The destination is not
overwritten until the complete expression has been evaluated. Logical AND and
OR retain short-circuit evaluation, including in constant expressions. Skipped
operands are still checked for valid names and supported types.

Here is a complete example combining measurement, Boolean expression values,
an explicit numeric-to-Boolean cast, conditional gates, and bit assignments:

```qasm
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] outcomes;
x q[0];
outcomes = measure q;

bool ready = outcomes[0] && !outcomes[1];
int count = 2;
bool enabled = bool(count);
bool proceed = ready && enabled;
if (proceed) {
    x q[1];
} else {
    reset q[1];
}
outcomes[1] = measure q[1];
ready = !ready;
outcomes[0] = ready;
```

The final values are `ready = false`, `proceed = true`, and `outcomes = "10"`.

Boolean expressions can also be assigned to scalar bits, indexed bit elements,
and numeric variables using the supported conversions described above.
Boolean gate parameters require an explicit numeric cast. Scalar bit bitwise
operators are supported as described above; Boolean bitwise operands require
an explicit cast.
Direct integer, floating-point, or complex conditions are also
rejected; use a supported explicit comparison instead for integer and
floating-point values. Variables used by a branch must be declared outside it;
general block-local declarations are not yet supported. The scoped iteration
variables described below are an exception.

## Constant for loops

### Integer ranges

The converter supports `for int name in [start:stop]` and
`for int name in [start:step:stop]`, with either a single-statement or braced
body. Bounds and steps must evaluate to integers during conversion: literals,
previously declared constants, supported constant expressions and explicit
integer casts are accepted. Nested bounds may also use outer iteration values.
Runtime variables are not constant bounds or steps, even when initialized with
a literal;
such ranges use the runtime lowering described below instead of unrolling.
Only `int` iteration variables are supported. UINT bounds and steps are accepted
when their evaluated values fit QCX `INT`; an unrepresentable constant UINT
operand is rejected during conversion, including for an empty range.
Declared signed iterator widths remain unenforced.

Following the [OpenQASM range-loop rules](https://openqasm.com/versions/3.0/language/classical.html#for-loops),
the step defaults to one and the stop is inclusive when reached. Negative steps
are supported. A range whose direction does not match its step is empty, and a
zero step is rejected. Both bounds must be present.

```qasm
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;
for int i in [0:3] {
    h q[i];
}
```

This generates `H 0`, `H 1`, `H 2`, and `H 3` after `QUBITS 4`. The converter
unrolls loops without adding QCX loop instructions or runtime iterator storage.
Within a loop, static qubit and bit indices may use constant expressions such
as `q[2 * i + j]`, including supported range and discrete-set selections.
Runtime-dependent qubit indices remain unsupported. Single bit-register elements
may use runtime indices, including enclosing runtime integer iterators.

### Constant integer sets

The converter also supports `for int name in {value, ...}`, with a
single-statement or braced body. Each element must evaluate to an integer during
conversion. Integer literals, previously declared constants, supported constant
expressions, explicit integer casts, and outer iteration values are accepted.
As with ranges, only `int` iterators are supported. Constant UINT elements must
fit QCX `INT`; declared signed iterator widths remain unenforced.

Elements are visited in the listed order, including duplicates. They are not
sorted or deduplicated:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
for int i in {0, 2, 5, 2} {
    x q[i];
}
```

This generates `X 0`, `X 2`, `X 5`, and `X 2` after `QUBITS 6`. Both occurrences
of `2` are separate iterations, including for `continue` targets and expansion
accounting. No new QCX instruction or runtime iterator storage is required.

Nested sets may depend on an outer iterator:

```qasm
OPENQASM 3.0;
const int first = 2;
int total = 0;
for int i in {first, first + 1} {
    for int j in {i, i + 1} {
        total += j;
    }
}
// total is 12 after execution: 2 + 3 + 3 + 4.
```

Runtime variables are not constant elements, even when initialized with a
literal; sets containing them use the runtime-valued lowering described below.
Non-integer elements require a supported explicit integer cast;
implicit conversion of floating-point, Boolean, or complex elements is not
supported. Signed/unsigned integer-array, floating-point-array, Boolean-array,
complex-array, angle-array, and bit-register iteration use the runtime lowering
described below; alias iteration remains unsupported.
The installed parser requires a nonempty set in
source code: the spelling `{}` is rejected. An empty discrete-set AST supplied
directly to the converter emits no body instructions but still validates its
body structurally.

### Bodies and iterator scope

Loop bodies may contain supported gates, global phase, measurement, reset,
barriers, assignments to existing variables, conditionals, nested loops,
`break`, and `continue`.
Iterator reads are integer literals in supported expression contexts, including
gate parameters and comparisons. Operations on runtime variables remain runtime
QCX operations; unrolling does not evaluate measurement outcomes or choose
runtime branches. Computations remain inside their original branches, with
distinct generated labels and safe temporary reuse.

```qasm
int total = 0;
for int i in [1:3] {
    for int j in [0:i] {
        total += i + j;
    }
}
// total is 30 after execution.
```

The iterator is visible only within its body. It may shadow an outer variable,
constant, or iterator, and the outer binding is restored afterward. Bounds and
set elements are evaluated before binding the new iterator, so a shadowing
inner loop can use the outer value in its bounds or elements. These rules follow
[OpenQASM scoping](https://openqasm.com/versions/3.0/language/scope.html).
As an explicit subset restriction, assignments or measurements into any active
iterator are rejected, although OpenQASM itself permits modifying iterators.
General local declarations remain unsupported. Supported array and bit-register
iteration are described below.
Runtime `while` loops may also appear inside a `for` body, as described below.

### Break and continue

Within a supported `for` loop, `break;` exits the nearest enclosing loop, and
`continue;` skips the rest of the current iteration and proceeds to the next
one. Both statements may appear inside runtime conditionals, including
conditions based on measurement results. A transfer in an inner loop does not
exit or continue its outer loop. Transfers outside a supported loop are rejected.

```qasm
int total = 0;
int skip = 2;
int stop = 4;
for int i in [0:5] {
    if (i == skip) { continue; }
    if (i == stop) { break; }
    total += i;
}
// total is 4 after execution: 0 + 1 + 3.
```

The converter lowers transfers to existing QCX `JUMP` instructions: `break`
targets a label after all expanded iterations of its loop, while `continue`
targets a label at the end of the current expanded iteration. Conditions use
the existing `JUMP`/`JUMPIF` lowering. No new QCX instruction or runtime iterator
is required, and loops without their own transfers emit no extra loop labels.
Temporary declarations are moved before control flow when needed so that
skipping their first use cannot leave later uses undeclared; computations stay
in their original positions and are skipped at runtime as appropriate.

Transfers do not prune conversion. Later statements and iterations are still
validated and generated, even after an unconditional `break` or `continue`.
Unsupported syntax in such statements remains an error, and all expanded
iterations and statements still count toward the limits below. Runtime-skipped
operations are not executed by `bra`, but conversion-time constant evaluation
and validation still apply.

### Validation and expansion limits

Empty loops emit no body instructions. Structural checks still reject
unsupported body statements, operators, gate syntax, and iterator writes.
Independent nested bounds and set elements are validated; checks that require
an outer iterator's value are deferred until an actual iteration. No fictitious
iteration value is used to execute body arithmetic or check an iterator-dependent
index.

To bound expansion, the converter checks these limits:

- 10,000 total expanded `for` iterations, including outer and inner iterations and
  sequential loops;
- 100,000 statements visited inside expanded bodies, including conditionals
  and nested loop headers; and
- 1,000,000 accumulated QCX lines, checked while converting loop bodies and
  emitting loop-control labels.

The budgets apply across a conversion, not separately to each loop. Initialization
and emission check independent copies of the budgets, so the two passes do not
charge each iteration twice. Runtime-skipped branches still consume expansion
budgets because their instructions must be generated. Exceeding a limit produces
a converter error rather than silently truncating the loop.
The same budgets apply to constant range and set loops and to the generated
body copies of runtime-valued set loops and constant bit-register loops; repeated
elements each count as an expanded iteration. A `break` does not reduce the
generated iteration count.
Runtime-bound range, signed/unsigned integer-array, floating-point-array,
Boolean-array, complex-array, angle-array, and bit-register loops do not expand
their runtime iteration count.

## Runtime-bound for loops

The converter supports `for int name in [start:stop]` and
`for int name in [start:step:stop]` when either bound or the step depends on a
runtime variable or an enclosing runtime iterator. Both bounds must be present and have
integer type. Scalar `int` and `uint` variables, supported integer
arithmetic, and explicit integer casts are accepted. Floating-point, Boolean,
and scalar bit operands require an explicit integer cast; statically or
runtime-indexed bit-register elements are also accepted through a cast. Whole bit registers
are not integer range operands. These typing rules apply to start, step, and stop.

An omitted step is `1`. Positive and negative integer steps are supported,
including nonunit steps. Named constants, supported arithmetic and integer
casts, scalar runtime variables, and enclosing constant or runtime iterators
may supply the step. A runtime-valued step selects runtime lowering even when
both bounds are constant. Fully constant ranges continue to use unrolling.

Steps used in runtime lowering must fit QCX `INT`, whose backend representation
is C++ `int`. The converter derives its signed bounds from the host's native C `int`
through Python's `ctypes`; on a 32-bit `int` host, the accepted range is
`-2147483648` through `2147483647`, excluding zero. This assumes the generated
program runs on a backend with a compatible integer representation; different
target integer widths are not automatically detected. Out-of-range constant steps
produce converter-specific errors rather than unrepresentable QCX literals;
runtime expressions retain the backend's existing integer representation and
arithmetic limitations.
Declared signed widths do not change these bounds. Fully constant signed ranges
still use unrolling and retain their conversion-time integer-step behavior,
including signed steps outside the runtime QCX representation. UINT operands
are normalized to their unsigned widths first, and values used in INT loop
storage must fit QCX `INT`. Runtime UINT bounds and steps use checked `:INT:`
captures; out-of-range values throw `std::out_of_range` when loop entry executes.
Unsigned negation wraps and cannot supply a negative step; use `int` for
descending loops.

```qasm
OPENQASM 3.0;
int first = 1;
int last = 3;
int total = 0;
for int i in [first:last] {
    total += i;
    last = 0;
}
// total is 6: the captured stop remains 3 despite assignments to last.
```

For a nonunit step, the stop is included only if reached exactly:

```qasm
OPENQASM 3.0;
int first = 1;
int last = 6;
int total = 0;
for int i in [first:2:last] {
    total += i;
}
// total is 9: 1 + 3 + 5. The unaligned stop 6 is not visited.
```

Runtime steps are captured as well as bounds:

```qasm
OPENQASM 3.0;
int first = 1;
int last = 6;
int stride = 2;
int total = 0;
for int i in [first:stride:last] {
    total += i;
    stride = 0;
    last = -100;
}
// total is 9: the captured step remains 2 and stop remains 6.
```

The start, runtime-valued step, and stop are evaluated once in source order on
each entry to the loop, before binding the new iterator. A constant step is
evaluated during conversion. Mutating source variables in the body does not
change the captured range or step. On reentry, including from an enclosing
`while` loop, all runtime operands are evaluated and captured again. The stop is
inclusive when reached, and a range whose direction does not match its step
executes zero iterations. A nested loop can use its outer iterator in either
bound or the step, including when the inner iterator shadows the outer name.
The outer binding is restored after the inner loop.

The body is generated once using QCX labels, `LET`, `JUMP`, and `JUMPIF`.
Private integer storage holds the iterator, captured stop, and runtime-valued
step for the whole loop, independently of expression temporaries and nested
loops. For a runtime-valued step, generated `ASSERT captured_step \= 0`
checks the captured value after evaluating the range operands but before
testing whether the range is empty. A failed check throws `bra::assertion_error`
and identifies the comparison and evaluated values; the CLI currently leaves
this exception uncaught. Zero is therefore an error even for an otherwise empty
range. Literal and conversion-time constant zero steps remain converter errors.
Generated programs with runtime-valued steps require a bra version supporting
`ASSERT`, documented in [bra.md](../../docs/bra.md).

An entry check skips empty ranges. After each iteration, an advance guard exits if the
next value would lie beyond the stop, including when the stop is unaligned.
Conversion-time constant unit steps retain the existing endpoint check and
emitted instructions. Runtime-valued steps select entry and advance comparisons
from the captured step's sign; the body is not duplicated for the two directions.

For nonunit constant steps and runtime-valued steps, the guard selects a safe
calculation according to the stop's sign. It computes `stop - step` only where
that subtraction is representable;
otherwise it computes and checks a representable candidate `iterator + step`.
The iterator is advanced only when another in-range value exists. Negative
steps are used directly without negating them, including the minimum signed
integer. Thus both guard arithmetic and iterator advancement avoid overflow
for representable bounds and steps. This is not a general arithmetic overflow
check: signed calculations producing bounds or steps, signed body arithmetic,
and declared signed widths retain their existing limitations. Unsigned
calculations follow the width and promotion rules described above.

Bodies support the same statements as constant loops. The iterator is a scoped,
read-only runtime integer usable in arithmetic, gate parameters, and conditions.
Assignments and measurements into active iterators remain rejected. Runtime
iterators cannot supply static qubit indices. Using one in a range step
selects runtime-step lowering; using one in a set element selects runtime-valued
set lowering.
Runtime integer-array and single-element bit-register indices are supported;
runtime qubit indices remain unsupported. Constant range and set loops,
runtime-valued set loops, and runtime `while` loops may be nested in either direction.

`break` exits the nearest loop. In a runtime-bound `for`, `continue` jumps to
the advance guard and iterator advancement, rather than reevaluating the
bounds or step. For example:

```qasm
int last = 5;
int total = 0;
for int i in [0:last] {
    if (i == 2) { continue; }
    if (i == 4) { break; }
    total += i;
}
// total is 4 after execution: 0 + 1 + 3.
```

Bodies are converted and validated even if the runtime range is empty or a
transfer makes later statements unreachable. Temporary declarations may move
before control flow, but computations remain at their original positions.
Runtime computations in bound and step expressions, including integer division
and remainder, remain at loop entry; body computations remain on the paths that
reach them. A skipped branch or outer runtime loop therefore skips generated
nested range computations and zero-step assertions too. Integer zero divisors
are checked by `bra` when the division executes. Conversion-time constant
validation still applies to fully constant ranges and independent constant
bounds or steps of nested loops; those checks are not suppressed by a
runtime-skipped path.

Runtime iteration counts do not consume the constant-loop expansion budgets,
and there is no runtime iteration limit. Constant loops nested in a runtime
loop are expanded once per generated instance and still consume the budgets.
A runtime loop nested in an expanded constant loop contributes its generated
statements and instructions to the enclosing expansion budgets.

## Runtime-valued integer-set for loops

The converter supports `for int name in {value, ...}` when one or more elements
depend on a runtime variable or an enclosing runtime iterator. Elements must
have integer type: scalar `int`, `uint`, supported integer arithmetic,
and supported explicit integer casts are accepted. Floating-point, Boolean,
and scalar bit values require integer casts; statically or runtime-indexed
bit-register elements may also be cast. Whole-register numeric casts and runtime complex-to-`int`
casts remain unsupported here; explicit complex-to-`uint` casts use the real
component with native range checks. Existing constant numeric casts retain
their behavior. Captured UINT elements use checked conversion to QCX `INT`;
values above `INT_MAX` throw `std::out_of_range` before the first body executes.

```qasm
OPENQASM 3.0;
int first = 1;
int last = 3;
int total = 0;
for int i in {first, last, first + 1, first} {
    total += i;
    first = 9;
    last = 9;
}
// total is 7: the captured elements are 1, 3, 2, 1, in that order.
```

All elements are evaluated and captured once, in source order, on each entry
to the loop, before binding its iterator or executing any body instruction.
The body cannot change the captured values by modifying their source variables.
Duplicate elements are distinct iterations and are neither sorted nor removed.
Even an unconditional `break` in the first body does not skip evaluation of
later elements. For example, an executed runtime zero divisor in a later
element still produces a `bra` error before the first body runs.

The set's element count is fixed in the source. The converter emits one body
copy per element using existing QCX `LET`, labels, and `JUMP` instructions.
Private storage holds every captured element and the runtime iterator throughout
loop execution. The converter reserves this storage while generating the body
copies, so expression temporaries and nested loops cannot reuse live captures.
The set-loop lowering itself needs no new bra instruction or runtime array
indexing; element expressions may use the supported indexed accesses.
Fully constant sets continue to use the existing constant-iterator unrolling.

The iterator is scoped to the body and may shadow a source variable, constant,
or outer iterator. Element expressions use the outer binding, and that binding
is restored after the loop. The iterator is a read-only runtime integer usable
in arithmetic, conditions, gate parameters, and runtime range bounds or steps,
and integer-array or single-element bit-register indices, but not in static
qubit indices.
General block-local declarations remain unsupported.

`break` exits the nearest loop; `continue` skips the rest of the current element's
body and proceeds to the next captured element. Each body copy has its own
continuation label, including when values are duplicated. Nesting with constant
range/set loops, runtime-bound ranges, other runtime-valued sets, and `while`
loops is supported. Nested element expressions may use outer runtime iterators.

Temporary declarations may move before control flow, but element computations
remain at loop entry and body computations remain in their original positions.
Skipped branches, empty outer runtime ranges, and earlier transfers can skip
generated capture computations. Logical element expressions preserve
short-circuit evaluation. Conversion-time validation still applies, including
independent constant elements of nested loops; an invalid constant expression
is not silently treated as a runtime value. Bodies and later elements remain
validated even after unconditional transfers.

Every generated body copy consumes the expansion budgets described above,
whether or not it executes. Nested generated statements and instructions also
count, even when their iterator is runtime-valued or shadows an outer iterator.
Runtime reentry into an already generated loop does not charge additional
conversion-time iterations. Signed widths remain unenforced; unsigned expressions
follow the normalization and promotion rules above. Signed/unsigned integer-array,
floating-point-array, Boolean-array, complex-array, angle-array, and bit-register
iteration are described below; alias iteration remains unsupported.
The parser still rejects
an empty source set `{}`.

## Integer-array for loops

The converter supports `for int name in values` when `values` is a previously
declared one-dimensional integer array. Elements are visited in increasing
index order, including repeated values. Only `int` iterators are supported.

```qasm
OPENQASM 3.0;
array[int, 3] values = {1, 3, 5};
int total = 0;
for int value in values {
    total += value;
    values[1] = 10;
}
// total is 16: the visited values are 1, 10, 5.
```

The converter uses live element reads: at the start of each iteration,
the current element is copied into private iterator storage. A body write to a
future element affects its later visit; a write to the current element does not
change the iterator's copied value. This is the converter's explicit mutation
policy. The [OpenQASM array-loop rules](https://openqasm.com/versions/3.0/language/classical.html#for-loops)
specify index order and a non-reference iterator but do not explicitly settle
whether the iterable's values are snapshotted at entry.

In contrast, `for int value in {values[0], values[1]}` captures both element
values before executing its first body, using runtime-valued set lowering.
Reentering an array loop starts again at element zero and reads the array's
then-current contents. The array's size remains fixed.

The iterator is scoped to the body and may shadow a source variable, constant,
array, or outer iterator. The outer binding is restored afterward. The source array is
resolved before binding the iterator, so `for int values in values` is accepted.
As with other supported loops, iterator writes and measurements into active
iterators are rejected. Runtime integer iterators can index integer-array and
bit-register elements, but cannot supply static qubit selections. Explicitly indexing
with a constant outer iterator remains supported.

Bodies support the same statements as other loops, including nested range,
set, array, and `while` loops. `break` exits the nearest loop; `continue` proceeds
to its next element. Runtime computations remain in their original branches
and bodies, while temporary declarations may move before control flow.
Bodies remain validated even when skipped at runtime or after an unconditional
transfer.

The body is generated once using existing QCX `VAR`, `LET`, labels, `JUMP`, and
`JUMPIF` instructions. Private integer storage holds the current index and the
copied iterator value; QCX's existing indexed operands read the source array.
No new bra instruction is required. A final-index check occurs before advancing,
so loop-control arithmetic stays within QCX `INT` for an accepted array size.
This does not add general overflow checks or enforce declared integer widths.

Runtime array iteration counts do not consume expansion budgets. Nested
constant or set-loop body copies still consume their usual budgets, and an
array loop inside an expanded loop contributes its generated statements and
instructions. Unsigned-integer-array, floating-point-array, Boolean-array,
complex-array, angle-array, and bit-register iteration are described below.
Iteration over unsupported array types, aliases, or arbitrary array expressions
remains unsupported. Static and runtime slice iteration are described below.

## Static array-slice for loops

Previously declared one-dimensional `int`, `uint`, `float`, `bool`, `complex`,
and `angle` arrays support `for` loops over static ranges, such as
`values[1:3]` or `values[4:-2:0]`. Bounds and steps must be compile-time
integers, both bounds are required, and an omitted step is one. Steps must be
nonzero, and each selected index must be within the source array. Negative
indices are normalized individually, so a range crossing zero can visit a
source element more than once. Selection order is preserved. Existing
array-iterator type and width rules, including UINT and angle widening, apply.

Each visit reads the current source element and copies it into the scoped,
read-only iterator. Changes to future or repeated source positions affect later
visits. This is the converter's chosen interpretation: the OpenQASM slice and
loop sections do not explicitly settle whether a slice iterable is snapshotted.
An empty static range performs no iterations, while the body is still checked;
the specification's empty-array-slice treatment is not explicit here. Static
out-of-range positions, invalid steps, types, names, and iterator writes are
rejected even inside skipped or empty enclosing loops.

<!-- static-array-slice-loop-example -->
```qasm
OPENQASM 3.0;
array[int, 5] values = {1, 2, 3, 4, 5};
int total = 0;
for int value in values[4:-2:0] {
    total += value;
}
// total is 9: the loop visits 5, 3, and 1.
```

The body is generated once and uses a private position and copied iterator;
negative positions are normalized at each visit. `break`, `continue`, nested
loops, and source shadowing follow the ordinary array-loop rules. Discrete
index sets and slice initialization,
concatenation, multidimensional arrays, and aliases remain unsupported.

## Runtime array-slice for loops

The same supported one-dimensional array types and iterator rules also accept
runtime-dependent `int` or `uint` start, step, and stop expressions. Both bounds
are required; an omitted step is one. They are captured once, in written order,
when the loop is entered. Changing a bound in the body does not change the
selected positions. The complete position list is checked before the first
body execution, then each visit reads the current source element. Repeated
negative aliases can therefore observe source mutations on later visits.

Reached selections check that the step is nonzero and every generated index
is valid. Native signed and unsigned bounds are retained without silently
narrowing them. An empty range executes zero iterations; as with static array
slices, this is the converter's documented interpretation. Skipped loops do
not evaluate bounds or perform runtime checks, but their types, names, iterator
bindings, and bodies are validated.

<!-- runtime-array-slice-loop-example -->
```qasm
OPENQASM 3.0;
array[int, 5] values = {1, 2, 3, 4, 5};
int first = 1;
int last = 3;
int total = 0;
for int value in values[first:last] {
    total += value;
    last = 0; // The selected positions remain 1, 2, and 3.
}
// total is 9.
```

The captured positions use private INT-indexed storage. Its capacity is the
smallest of twice the array length, the converter's runtime position limit
(`MAX_RUNTIME_BIT_RANGE_POSITIONS`, currently 1,000,000), and QCX `INT_MAX`.
A reached selection exceeding that capacity fails before its body. The array
element values are not snapshotted. Static slices continue to use their
separate path and do not consume this runtime position storage.

## Unsigned-integer-array for loops

The converter supports `for uint name in values` and `for uint[n] name in values`
over previously declared one-dimensional UINT arrays. The iterator's effective
width must be at least the element width. Widening preserves the element's
numeric value; it does not shift the copied bits. An unsized `uint` iterator
has native width and can iterate over any supported UINT-array width. The
iterator retains its own specified/unsized metadata, and its expressions use
that width, including for modular arithmetic. Narrower iterators are rejected.
Its width must be a supported compile-time integer, resolved before introducing
its binding; enclosing loop iterators cannot supply a width.

```qasm
OPENQASM 3.0;
array[uint[8], 3] values = {1, 2, 255};
values[-1] += 1;
uint total = 0;
for uint[8] value in values {
    total += value;
}
// total is 3; values contains 1, 2, 0.
```

<!-- uint-array-widening-example -->
```qasm
OPENQASM 3.0;
array[uint[4], 2] values = {3, 15};
array[uint[8], 2] copied;
uint total = 0;
int index = 0;
for uint[8] value in values {
    copied[index] = value;
    total += value + uint[8](1);
    index += 1;
}
// copied contains 3 and 15; total is 20.
```

Elements are visited in increasing index order with the same live-read policy
as other array loops. Each current element is copied into a reserved `UINT`
iterator. Changes to future elements affect later visits; changes to the current
element do not change the copied iterator value. Reentry starts again at index
zero. The body is generated once, with a private `INT` position and the existing
endpoint guard; no new bra instruction is required.

Bindings are scoped and read-only. They can shadow source variables, constants,
arrays, or outer iterators; widths and values are restored with their scopes.
The source is resolved first, so `for uint[8] values in values` is accepted for
an eight-bit source array. Assignments and measurements into active iterators
are rejected. Nested range/set, supported array/register, and `while` loops,
`break`, and `continue` use the existing loop-control rules.

Iterator expressions preserve width through arithmetic, complement, negation,
shifts, exponentiation, casts, comparisons, and gate parameters. Bare UINT
conditions require a comparison or `bool(...)`. UINT values used as runtime
indices or `for int` bounds, steps, or set elements undergo the existing checked
conversion to QCX `INT`; values above `INT_MAX` are not silently narrowed.
Typed validation, including width compatibility and shadowing, also applies
inside empty enclosing loops without fabricating iterator values.

Runtime iteration counts do not consume constant-loop expansion budgets;
nested or enclosing expansions retain their existing accounting. Computations
stay on their original paths, while temporary declarations may be hoisted.
UINT range/set iteration, implicit cross-base-type iteration, narrowing
UINT iterators, slices outside supported loops and assignments, multidimensional arrays, aliases, and arbitrary array
expressions remain unsupported.

## Floating-point-array for loops

The converter supports `for float name in values` when `values` is a previously
declared one-dimensional floating-point array. Elements are visited in increasing
index order, including repeated values. Sized float base and iterator types such
as `float[32]` and `float[64]` follow the existing native-`REAL` policy: their widths
are not enforced, and differing declared float widths do not introduce rounding
or precision conversion. Signed-integer arrays require `int` iterators,
unsigned-integer arrays require matching or wider `uint` iterators, and float arrays
require `float` iterators; implicit cross-base-type conversion is not supported.

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
array[float, 3] angles = {0.1, 0.2, 0.3};
for float theta in angles {
    rx(theta) q;
}
```

The same live-read policy as integer-array loops applies. Each iteration reads
the current element into a private `REAL` iterator before executing the body.
Writes to future elements affect their later visits; writes to the current
element do not change its copied iterator value. Reentering the loop starts
at index zero and reads the array's then-current contents.

The iterator is scoped and read-only. It can shadow a variable, constant, source
array, or outer iterator, whose binding is restored afterward. The source is
resolved before binding the iterator, so `for float values in values` is accepted.
Assignments and measurements into active iterators are rejected. Arithmetic,
negation, comparisons, casts, gate parameters, and `gphase` arguments use the
same rules as scalar floats. Direct floating-point conditions require a
comparison or an explicit `bool(...)` cast. Integer indices, range bounds,
steps, and set elements require an explicit integer cast of a float iterator.

Nested integer-range/set loops, supported array and bit-register loops, and
`while` loops are supported. `break` exits the nearest loop; `continue` proceeds
to that loop's next iteration. A float iterator does not enable floating-point
ranges or discrete sets. Slices outside supported loops and assignments, multidimensional arrays, aliases, arbitrary
array expressions, and arrays of other unsupported element types remain outside
this subset; general block-local declarations remain unsupported.

The body is generated once using the existing QCX loop instructions. A private
`INT` index and copied `REAL` iterator remain reserved throughout the body,
including nested computations. An endpoint guard precedes index advancement,
as for integer arrays. No new bra instruction is required. Runtime iteration
counts do not consume constant-loop expansion budgets; nested expansions and
enclosing expanded loops retain their existing accounting.

Computations stay on their original execution paths, while temporary
declarations may move before control flow. Skipped bodies and short-circuit
operands skip their runtime arithmetic and access checks. Iterator types,
conditions, read-only bindings, and supported syntax are validated even inside
empty enclosing loops. Failed loop lowering releases loop-owned temporaries without releasing
storage reserved by enclosing callers.

## Boolean-array for loops

The converter supports `for bool name in flags` when `flags` is a previously
declared one-dimensional Boolean array. Elements are visited in increasing index
order, including repeated values. The iterator must be `bool`; `bit` and numeric
iterators do not implicitly convert Boolean array elements. Conversely, `bool`
iterators cannot iterate over bit registers or numeric arrays.

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
array[bool, 3] flags = {true, false, true};
int count = 0;
for bool flag in flags {
    if (!flag) { continue; }
    count += int(flag);
    x q;
}
// count is 2; q is back in state |0>.
```

Each iteration copies the current element into a private Boolean iterator backed
by QCX `INT`. Writes to future elements affect their later visits, but writes to
the current element do not change its copied iterator value. Reentering the loop
starts at element zero and reads the array's then-current contents.

The iterator is scoped and read-only. It can shadow variables, constants, arrays,
or outer iterators, with their bindings restored afterward. The source is resolved
before binding the iterator, so `for bool flags in flags` is accepted. Assignments
and measurements into active iterators are rejected. Iterators retain Boolean
semantics: they support direct conditions, logical expressions, Boolean/scalar-bit
equality, casts, and scalar Boolean/bit assignments. Numeric operations, integer
indices, range bounds, steps, set elements, and gate parameters require explicit
numeric casts as described for Boolean array elements.

Nested supported range, set, array, bit-register, and `while` loops are supported.
`break` exits the nearest loop; `continue` proceeds to its next iteration. Boolean
iterators do not enable Boolean ranges or sets. Slices outside supported loops and assignments, multidimensional arrays,
aliases, arbitrary array expressions, and block-local declarations remain unsupported.

The body is generated once using existing QCX loop instructions, with a private
`INT` position and copied Boolean iterator reserved throughout the body. No new
bra instruction is required. Runtime iteration counts do not consume constant-loop
expansion budgets; enclosing or nested expanded loops retain their existing
accounting. Runtime checks and arithmetic stay on their original execution paths,
so skipped bodies and short-circuit operands skip their access checks. Types,
conditions, supported syntax, and read-only bindings are validated even inside
empty enclosing loops. Failed lowering releases loop-owned temporary storage
without releasing storage reserved by callers.

## Complex-array for loops

The converter supports `for complex name in values` over previously declared
one-dimensional complex arrays. Explicit iterator types such as
`complex[float[64]]` are also accepted. Elements and iterators must both have
complex base type; no implicit conversion from numeric or Boolean arrays or bit
registers is performed. Declared component widths follow the scalar native-storage
policy, so differing complex element/iterator widths do not introduce rounding.

```qasm
OPENQASM 3.0;
array[complex[float[64]], 2] values = {1 + 1.0im, 2 + 2.0im};
complex total = 0;
for complex z in values {
    total += z;
}
// total is 3 + 3im.
```

Elements are visited in increasing index order. Each current element is copied
into a private `COMPLEX` iterator before the body executes. Writes to future
elements affect later visits, while writes to the current element do not change
its copied iterator value. Reentering the loop starts at element zero and reads
the array's then-current contents.

The iterator is scoped and read-only. It may shadow a variable, constant, array,
or outer iterator, whose binding is restored afterward. The source is resolved
before binding the iterator, so `for complex values in values` is accepted.
Assignments and measurements into active iterators are rejected. Arithmetic,
negation, casts, and type restrictions follow the complex-element rules above.
In particular, iterators are not direct conditions or comparison operands;
`uint(z)` can supply supported integer indices and nested range/set values, and
`float(z)` can supply gate parameters. Complex ranges and sets remain unsupported.

Nested supported range, set, array, bit-register, and `while` loops are supported.
`break` exits the nearest loop; `continue` proceeds to its next iteration. The
body is generated once, with a private `INT` position and copied `COMPLEX`
iterator reserved throughout the body. An endpoint guard precedes index
advancement. No new bra instruction is required, and runtime iteration counts
do not consume constant-loop expansion budgets. Enclosing and nested expanded
loops retain their existing accounting.

Runtime arithmetic and access checks stay on their execution paths, including
skipped bodies and short-circuit operands. Iterator types, read-only bindings,
conditions, and unsupported operators are validated even inside empty enclosing
loops, without executing skipped arithmetic. Failed lowering releases loop-owned
temporaries while preserving caller-owned storage. Slices outside supported loops and assignments, multidimensional
arrays, aliases, arbitrary array expressions, and block-local declarations remain
outside this subset.

## Angle-array for loops

The converter supports `for angle name in phases` and
`for angle[n] name in phases` over previously declared one-dimensional angle
arrays. Elements are visited in increasing index order, including repeated
values. The iterator's effective width must be at least the element width.
Widening appends low-order zero bits to the copied element, preserving the
represented angle. Unsized `angle` has the native QCX UINT width `W`; sized
and unsized forms can be combined when their effective widths permit widening.
The iterator keeps the sizedness of its own declaration. Narrowing and
implicit iteration with a different element type remain unsupported, even
where an explicit scalar conversion would be accepted.

The source array and iterator width are resolved before introducing the
iterator binding. A width must be a supported compile-time integer expression,
independent of enclosing loop iterators. Array length and element width remain
fixed when an iterator shadows a declaration constant.

The live-read policy described under [Integer-array for loops](#integer-array-for-loops)
also applies here. Each iteration copies the current element into private UINT
storage with distinct angle typing and width metadata. A write to a future
element affects its later visit; a write to the current element leaves the
copied iterator unchanged. Reentry starts at index zero and reads the array's
then-current contents. The source is resolved before binding the iterator, so
`for angle[4] phases in phases` is accepted. Outer variables, constants, arrays,
and iterators shadowed by the loop variable are restored after the body.

Iterators support the scalar angle expressions, comparisons, bit functions,
casts, and phase-gate parameters described above. Use an explicit comparison or
`bool(value)` in a condition; a bare angle is not a Boolean condition. Integer
results such as `popcount(value)` or an angle quotient can supply nested integer
range/set values. Trigonometric results can supply floating-point gate parameters.
Gate modifiers retain their existing restrictions, including constant-only
floating-point powers. Fusion preserves the captured gate parameters when the
iterator storage is reused or source elements are changed later.

<!-- angle-array-loop-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
array[angle[4], 3] phases = {0.0, pi / 2, pi};
angle[4] total = 0.0;
int i = 0;
h q;
for angle[4] phase in phases {
    if (phase == angle[4](0.0)) {
        i += 1;
        continue;
    }
    p(phase) q;
    phases[i] = 0.0;              // The copied phase is unchanged.
    total += phase;
    if (i == 1) {
        phases[2] = angle[4](3 * pi / 4); // The next visit sees this value.
    }
    i += 1;
}
// total has stored bits 1010 (5*pi/4); phases contains three zero angles.
```

<!-- angle-array-widening-example -->
```qasm
OPENQASM 3.0;
array[angle[4], 2] phases = {angle[4]("0011"), angle[4]("0110")};
array[angle[8], 2] copies;
int index = 0;
for angle[8] phase in phases {
    copies[index] = phase; // The stored patterns are "00110000" and "01100000".
    index += 1;
}
```

Nested supported range, set, array, bit-register, and `while` loops are accepted.
`break` exits the nearest loop; `continue` advances to its next iteration. As with
other converter loops, assignments and measurements into active iterators are
rejected, including in skipped bodies. Read-only iterators are a converter
restriction: OpenQASM permits modifying the local loop variable without changing
the source element. Source visibility, types, widths, index syntax, and read-only
bindings are validated even in empty enclosing loops; skipped paths do not
execute runtime arithmetic, index checks, or operand domain checks.

The body is generated once using existing QCX instructions, with a private INT
index and UINT iterator reserved throughout the body. An endpoint guard precedes
index advancement. Loop-owned temporaries are released on success and failure
while caller-owned storage is preserved. Runtime iteration counts do not consume
expansion budgets; nested and enclosing expanded loops retain their accounting.
No new bra instruction or build option is required. Angle ranges/sets, narrowing
iterator widths, array slices outside supported loops and assignments, aliases, arbitrary array expressions, multidimensional
or zero-length arrays, and block-local declarations remain unsupported.
Iteration over an element's bits uses scalar `bit` iterators, as described under
[angle bit-selection loops](#angle-bit-selection-for-loops).

## Bit-register for loops

The converter supports `for bit name in flags` when `flags` is a previously
declared mutable `bit[N]` or constant `const bit[N]` register.
Elements are visited in increasing index order,
including repeated values. Bit zero is the rightmost bit in a source bit string:

```qasm
OPENQASM 3.0;
bit[3] flags = "101";
int total = 0;
for bit value in flags {
    if (value) {
        total += 1;
    }
}
// total is 2 after visits to flags[0], flags[1], and flags[2].
```

The iterator retains scalar `bit` type even though QCX stores its copied value
in a private `INT`. It supports direct conditions, logical expressions,
supported comparisons and casts, and assignments to existing scalar bits,
statically indexed bit elements, or Boolean variables. Use an explicit numeric
cast for arithmetic, gate parameters, integer range bounds or steps, integer
set elements, and integer-array indices, for example `int(value)`.

Bit-register loops require a scalar `bit` iterator. `for int` over a bit register,
`for bit` over an integer array, and register-valued iterators such as `bit[1]`
are rejected. A `bit[1]` source is iterable and visits its single element;
a scalar `bit` source is not iterable. As elsewhere, a scalar bit cannot be
implicitly broadcast to a whole-register assignment target, including `bit[1]`.

For mutable sources, the converter uses the same live-read policy as integer-array
loops: each iteration copies the current element into the iterator before executing its
body. Writes or measurements into future source elements affect later visits,
but changing the current source element does not change its copied iterator.

```qasm
OPENQASM 3.0;
bit[3] flags = "111";
int total = 0;
for bit value in flags {
    flags[1] = 0;
    total += int(value);
}
// total is 2: the visited values are 1, 0, 1.
```

The source is resolved before binding the iterator, so `for bit flags in flags`
is accepted. The iterator is scoped to its body and may shadow a source variable,
constant, register, or outer iterator; the outer binding is restored afterward.
Its type is preserved through mixed bit/integer nesting. Writes and measurements
into active iterators remain rejected, including on unreachable paths.

Register sizes are resolved in declaration scope and remain fixed when an
iterator shadows a size constant. A mutable register used for iteration must
have a positive size that fits the host-derived QCX `INT` range described under
runtime-bound loops. This assumes a backend with compatible integer storage.

Bodies support the same statements as other loops, including measurement,
conditional gates, and nested range, set, array, bit-register, and `while` loops.
`break` exits the nearest loop; `continue` advances to the next register element.
Computations stay on their original execution paths; skipped branches and
transfers skip runtime arithmetic and accesses, while conversion-time validation
still applies.

For mutable sources, the body is generated once using existing QCX storage,
indexed operands, labels, `LET`, `JUMP`, and `JUMPIF`. Private storage holds the current index and copied
bit value throughout the body. Singleton registers use their existing scalar
QCX storage. An endpoint check precedes advancement, keeping loop-control
arithmetic representable for accepted register sizes. No new bra instruction is
required. Runtime iteration counts do not consume expansion budgets, but nested
expanded loops and enclosing expansions retain their existing accounting.

Bit iterators may index integer-array and mutable bit-register elements through an
explicit integer cast. Static register selections are also iterable, as described
[below](#static-selection-sources), as are
[runtime singleton sets](#runtime-singleton-selection-sources) and
[runtime multi-entry sets](#runtime-multi-entry-selection-sources) and
[runtime ranges](#runtime-dependent-bit-register-range-loops).
Runtime user-supplied qubit indices, other runtime-dependent register selections
as iterable sources, aliases, general
array-type iteration, and block-local declarations remain unsupported.

### Constant sources

For a named `const bit[N]` source, the converter unrolls one body copy per source
bit, assigning literal zero or one to private BIT-typed iterator storage before
each copy. The constant source itself needs no QCX storage. This supports widths
above native integer size without packing the register into a native integer.
The source is resolved before the iterator is bound, so
`for bit FLAGS in FLAGS` is also valid for a constant register.

<!-- constant-bit-register-loop-example -->
```qasm
OPENQASM 3.0;
const bit[8] FLAGS = "10001111";
uint count = 0;
int visits = 0;
for bit value in FLAGS {
    visits += 1;
    if (!value) { continue; }
    count += uint(value);
}
bool unchanged = FLAGS == "10001111";
// Visits 1, 1, 1, 1, 0, 0, 0, 1; count is 5, visits is 8.
```

Although source bits are known, the iterator remains a read-only scalar bit with
runtime expression semantics, not a constant integer. It cannot supply static
qubit indices or compile-time selections of constant registers. Numeric uses
require the same explicit casts as mutable-register iterators. Both iterator
writes and source-register writes are rejected, including in skipped bodies.
Names must be declared before use, outer bindings are restored after shadowing,
and source widths retain declaration-scope meaning.

Nested constant/mutable register, array, range, set, and `while` loops retain
their respective behavior. `break` targets the nearest loop's end; `continue`
targets the next body copy of this loop. Runtime calculations remain on the
paths that reach them. Temporary declarations are hoisted so skipped copies
cannot leave reused storage undeclared.

Each constant-register body copy consumes the conversion-wide expansion
budgets above, including copies skipped at runtime by branches or transfers.
An empty enclosing constant loop validates the nested loop without expanding
its body. Large registers can therefore hit expansion limits even though their
declarations and constant expressions are otherwise supported. No new bra/QCX
instruction is required. Computed register expressions, runtime-dependent
selections outside the loop subsets below, aliases, scalar bits,
and general block-local declarations remain unsupported for bit-register loop sources.

### Static selection sources

The converter supports scalar-bit iteration over static ranges and discrete sets
on named mutable `bit[N]` and constant `const bit[N]` registers:

```qasm
for bit b in flags[0:3] { ... }
for bit b in flags[3:-1:0] { ... }
for bit b in flags[{-1, 0, -1}] { ... }
for bit b in FLAGS[2:2] { ... }
for bit b in one[{0}] { ... } // one is bit[1].
```

Only scalar `bit` iterators are accepted. A single integer subscript such as
`flags[2]` produces scalar `bit`, so it cannot be a loop source; use
`flags[2:2]` or `flags[{2}]` for an iterable singleton selection. This differs
from angle bit slicing, whose single-index result is `bit[1]`.
The specification permits indexed expressions yielding `bit[n]` as loop sources;
see [OpenQASM for loops](https://openqasm.com/language/classical.html#for-loops)
and [classical bits and registers](https://openqasm.com/language/types.html#classical-bits-and-registers).

Selectors follow the existing static selection rules. Bounds, steps, and set
entries must be literals or supported expressions of previously declared
constants. Active iterators cannot supply them, even when unrolled or shadowing
a constant. Ranges include the stop when reached, default to step `+1`, and
require an explicit negative step for descending iteration. Negative indices
normalize after generating the raw range positions. Empty selections, omitted
bounds, zero steps, noninteger indices, and invalid positions are rejected
during conversion, including in skipped bodies and empty enclosing loops.

Visits follow selection order, preserving repeated positions and aliases after
negative-index normalization. For example, a four-bit register selected by
`{-1, 0, 3}` visits its high bit, low bit, and high bit again. Repeated-position
handling retains the converter's existing source-selection policy. For mutable
sources, each visit reads the current source bit and copies it to the iterator.
Writes or measurements into the source affect future visits, including repeated
positions, without changing the current copied value. The specification describes
register slices as references, but does not explicitly settle loop read timing;
live reads are the converter's documented interpretation.

The source and selectors are resolved before the iterator is bound, allowing
the iterator to shadow the register name or a selector constant. Outer bindings
are restored afterward. The iterator remains scoped and read-only, with scalar
`bit` type, including for constant sources. OpenQASM permits iterator assignment;
rejecting it remains a converter restriction. Mutable and constant selection
loops support nesting, reentry, conditional gates, measurement, `break`, and
`continue` with the same execution-path rules as whole-register loops.

<!-- static-bit-selection-loop-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit[4] flags = "0001";
uint checksum = 0;
int visits = 0;
bit saved = 0;
for bit b in flags[{0, 1, 0, 3}] {
    if (visits == 0) {
        flags[1] = true;
        flags[0] = false;
        saved = b; // Still 1: b is a copy.
    }
    if (visits == 1) { flags[3] = true; }
    checksum = checksum * 3 + uint(b);
    visits += 1;
}
// Visits 1, 1, 0, 1: checksum is 37, visits is 4, flags is "1010".
const bit[4] C = "1011";
uint ones = 0;
int constant_visits = 0;
for bit b in C[3:-1:0] {
    constant_visits += 1;
    if (!b) { continue; }
    ones += uint(b);
    if (ones == 2) { break; }
}
// Visits 1, 0, 1: ones is 2, constant_visits is 3, C is unchanged.
h q;
p(angle[4](flags)) q; // Phase 5*pi/4, from the final stored bits.
```

Mutable selection bodies are generated once, using a private position table
and a copied scalar-bit iterator. Singleton registers retain scalar QCX storage.
The selection length must fit QCX INT indexing; runtime iterations do not consume
unrolling budgets. Constant sources reuse constant-register unrolling, with one
body copy per selected entry. Repetitions count toward the existing expansion
budgets, while unselected source bits do not. Both paths support registers wider
than native UINT without packing them into a native integer. Temporary storage
is released on success and failed lowering, preserving caller-owned storage.
No new bra instruction or build option is required.

Runtime singleton sets, multi-entry sets, and ranges are described next.
Aliases, further indexing of a selection result,
computed register expressions, and general block-local declarations remain
outside this feature.

### Runtime singleton selection sources

A named mutable or constant bit register can also be iterated using a singleton
discrete set whose integer index depends on runtime storage or an enclosing iterator:

```qasm
for bit b in flags[{index}] { ... }
for bit b in C[{positions[index]}] { ... }
```

The loop has exactly one visit when the index is valid. `{index}` retains the
register result type `bit[1]`; `flags[index]` instead yields scalar `bit` and
is not an iterable source. The index must have supported INT/UINT type; supported
expressions include arithmetic, integer-array elements, integer-valued mathematical
functions, and enclosing iterators. Boolean, floating-point, and scalar-bit values
require a supported explicit integer cast. Purely constant singleton sets retain
their existing static-selection behavior.

The source register is resolved first. Its index is evaluated once, normalized,
and bounds-checked before reading the selected bit or binding the iterator.
For a register of width `N`, signed indices must be in `[-N, N - 1]`; negative
indices normalize by adding `N`. UINT indices must be below `N`; checked
conversion to QCX INT rejects values that cannot be represented. Both mutable and
constant source widths must fit the supported host-derived QCX INT indexing
range, including `bit[1]`. A singleton register still checks its index even
though its QCX storage is scalar.

The selected bit is copied to a private scalar-bit iterator before executing
the body. Changing the index variable or the source bit in the body cannot
change that copy or trigger another selection. Each reentry captures a fresh
index and value. The source and index may be shadowed by the iterator name;
outer bindings are restored afterward. Iterators remain scoped and read-only,
including for constant sources. Nesting, measurement, and conditional gates
follow the existing loop rules. Both `break` and `continue` exit the one-visit
loop, after which execution continues in its enclosing body.

Supported runtime index arithmetic and bounds checks stay on reached paths.
Invalid indices fail before the body, even if it is empty or starts with a
transfer. Skipped branches and empty enclosing loops still validate names,
types, and supported syntax without evaluating runtime indices or inventing
iterator values. Source resolution and index-capture timing are documented
converter semantics; the specification does not explicitly settle their timing.

<!-- runtime-singleton-bit-loop-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit[4] flags = "1010";
int index = -1;
bit mutable_saved = 0;
int visits = 0;
for bit b in flags[{index}] {
    index = 0;
    flags[3] = false;
    mutable_saved = b; // Still 1: both the index and bit were captured.
    visits += 1;
}
// flags is now "0010"; the body ran once.
const bit[65] C = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000";
int wide_index = -1;
bit constant_saved = 0;
for bit b in C[{wide_index}] {
    wide_index = 0;
    constant_saved = b; // Bit 64 was selected, not the new index 0.
    visits += 1;
    continue;
    visits += 100; // Skipped: continue exits this singleton loop.
}
// visits is 2; both saved bits are 1; C is unchanged.
h q;
if (mutable_saved && constant_saved) { p(pi / 4) q; }
```

The body is generated once for both mutable and constant sources; runtime
visits do not consume unrolling budgets. Wide constants use private native-width
UINT words so the full source is preserved. Lookup code grows with the number
of words and is bounded by the existing QCX output limit. Lookup temporaries
are released after copying the bit; the iterator remains reserved throughout
the body. Cleanup on failed lowering preserves caller-owned storage. Existing
QCX instructions suffice, without a bra instruction or build-option change.

Multi-entry sets and runtime ranges have the loop-source subsets described next.
Fixed-width runtime sets also support [expression reads](#runtime-bit-register-index-set-expressions).
Runtime ranges also support [plain and bitwise compound assignments](#runtime-bit-register-range-assignments).
They also support [copies into registers or selections](#copies-from-runtime-bit-register-ranges).
Scalar consumers include the supported UINT casts, `popcount`, and
[comparisons](#equality-with-runtime-bit-register-and-angle-bit-ranges).
Whole-register assignments and top-level initializers also accept
[runtime-range expressions](#whole-register-assignments-from-runtime-range-expressions).
Other runtime range expressions, aliases, and computed register sources remain
unsupported. Angle selection loops have their own supported forms below.

### Runtime multi-entry selection sources

Scalar-bit loops also accept multi-entry discrete sets on named mutable and
constant bit registers when any entry depends on runtime storage or an enclosing
iterator:

```qasm
for bit b in flags[{i, j, i}] { ... }
for bit b in C[{positions[index], -1, index}] { ... }
```

Each entry must have supported INT/UINT type independently; signed and unsigned
entries may be mixed. Supported index expressions and conversions follow the
singleton rules above. A set must be nonempty. Entirely constant sets keep their
static validation and lowering, including constant-source unrolling.

The source is resolved before the iterator is bound. Every occurrence of every
entry is evaluated once, in source order at loop entry, normalized, and checked
before the first body statement. This includes literal and constant entries
when another entry is runtime-dependent. Signed indices must be in `[-N, N - 1]`
for a source of width `N`; negatives normalize by adding `N`. UINT indices must
be below `N` and pass checked conversion to QCX INT. Source widths and selection
lengths must fit QCX INT indexing, including for `bit[1]` sources.

Visits preserve entry order, repetitions, and negative aliases. Captured
positions stay fixed even if the body changes index variables. Each mutable
source bit is read live when visited and copied into the scalar-bit iterator;
writes or measurements into the source affect future visits, including repeated
positions, without changing the current copied value. These capture and read
timings are converter policies; the specification describes register slices as
references but does not explicitly settle loop read timing.

An invalid later entry fails before the body, even if the body starts with
`break`. Arithmetic and bounds checks run only on reached paths. Skipped
branches and empty enclosing loops still validate all entry names, types, and
supported syntax without evaluating runtime indices. Iterators are scoped and
read-only; source and selector names may be shadowed and are restored afterward.
`continue` advances to the next captured position, and `break` exits the nearest
loop. Reentry captures a fresh selection; nested loops retain independent state.

<!-- runtime-bit-set-loop-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit[4] flags = "1001";
int i = 0;
uint j = 3;
uint checksum = 0;
int visits = 0;
for bit b in flags[{i, j, i}] {
    if (visits == 0) {
        i = 1;
        j = 2;
        flags[0] = false;
    }
    checksum = checksum * 2 + uint(b);
    visits += 1;
}
// Captured positions are 0, 3, 0; live reads are 1, 1, 0.
// checksum is 6, visits is 3, and flags is "1000".
const bit[65] C = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001";
int index = -1;
uint ones = 0;
int constant_visits = 0;
for bit b in C[{index, 1, index}] {
    index = 0; // Does not change the captured positions 64, 1, 64.
    constant_visits += 1;
    if (!b) { continue; }
    ones += uint(b);
}
// ones is 2, constant_visits is 3, and C is unchanged.
h q;
if (checksum == 6 && ones == 2) { p(pi / 4) q; }
```

Both source kinds emit the body once and use a private INT position table;
runtime visits do not consume unrolling budgets. Mutable `bit[1]` retains scalar
QCX storage while still checking every index. Constant sources use native-width
UINT words prepared once per loop entry, with a single per-visit lookup. This
preserves wide sources and partial final words without native-integer truncation.
Constant-source initialization and body code are checked against the QCX output
limit; enclosing and nested expansions retain their accounting. Temporary declarations
are hoisted, and cleanup preserves caller-owned storage on success and failure.
No new bra instruction or build option is required.

This feature is restricted to direct scalar-bit loop sources. Runtime ranges
have their own subset below, and runtime sets also support
[expression reads](#runtime-bit-register-index-set-expressions) and
[assignments](#runtime-bit-register-index-set-assignments). Runtime ranges also
support [plain and bitwise compound assignments](#runtime-bit-register-range-assignments).
They can also be [copied into registers or selections](#copies-from-runtime-bit-register-ranges).
Scalar single-index loop sources,
computed register sources, aliases, further indexing of selections, and general
block-local declarations remain unsupported.

### Runtime-dependent bit-register range loops

Named mutable and constant bit registers support scalar-bit loops over ranges
whose start, stop, or step depends on runtime storage or an enclosing iterator:

```qasm
for bit b in flags[start:stop] { ... }
for bit b in C[start:step:stop] { ... }
```

Both bounds are required. Each operand must independently have supported
INT/UINT type; mixed signed and unsigned operands are accepted. The default
step is `+1`, descending ranges require an explicit negative step, and zero is
invalid. Supported operand expressions include arithmetic, integer-array
elements, integer-valued functions, and enclosing iterators. These forms follow
the specification's [register slicing](https://openqasm.com/language/types.html#register-concatenation-and-slicing)
and [for-loop](https://openqasm.com/language/classical.html#for-loops) rules.

The range generates raw positions `start`, `start + step`, and so on, stopping
before passing the inclusive stop bound. Each negative position then normalizes
by adding the source width `N`; endpoints are not normalized first. For `bit[4]`,
`[-4:3]` visits `0,1,2,3,0,1,2,3`. A stop need not itself be selected:
`[0:3:4]` selects only `0,3`, so the unselected stop `4` is not a bounds error.
Equal bounds select one position for any nonzero step.

At loop entry, the source is resolved and start, optional step, and stop are
evaluated once in textual order, before binding the iterator. The complete
position sequence is built and checked before any body statement executes.
A zero step, empty selection, or selected position outside `[-N, N - 1]`
fails on the reached path, even if the body immediately executes `break`.
Unlike an ordinary integer range loop, an empty register selection is invalid.
Entirely constant ranges retain conversion-time validation and existing lowering.
Skipped paths and empty enclosing loops still validate names, types, and syntax
without evaluating runtime operands or fabricating iterator values.

Captured positions stay fixed throughout the body. Mutable source bits are read
live on each visit and copied into the scoped, read-only scalar-bit iterator.
Writes and measurements affect later visits, including negative aliases, while
leaving the current copied value unchanged. Source and operand names may be
shadowed; outer bindings are restored afterward. Reentry captures a fresh range,
and nested loops, `break`, and `continue` retain their usual behavior. Capture,
complete-selection validation, and live-read timing are converter policies;
the specification does not explicitly settle these timings.

<!-- runtime-bit-range-loop-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit[4] flags = "0011";
int start = -4;
int step = 1;
int stop = 3;
int visits = 0;
uint ones = 0;
bit saved = 0;
for bit b in flags[start:step:stop] {
    if (visits == 0) {
        start = 999;
        step = 0;
        stop = -999;
        flags[0] = false;
        flags[3] = true;
        saved = b; // Still 1: b is a copy.
    }
    ones += uint(b);
    visits += 1;
}
// Reads 1,1,0,1,0,1,0,1: ones is 5, visits is 8, flags is "1010".
const bit[65] C = "1_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001";
int first = -1;
int stride = -32;
int last = -65;
uint constant_ones = 0;
int constant_visits = 0;
for bit b in C[first:stride:last] {
    first = 0; // Positions remain 64,32,0.
    constant_visits += 1;
    if (!b) { continue; }
    constant_ones += uint(b);
}
// constant_ones is 2, constant_visits is 3, and C is unchanged.
h q;
if (saved && constant_ones == 2) { p(pi / 4) q; }
```

Both source kinds emit the body once. Wide constants use the native UINT-word
table prepared once per loop entry; mutable `bit[1]` retains scalar storage.
Operand capture preserves native INT/UINT values, including full-width UINT
steps. Range advancement preserves the mathematical progression without wrapping
at native integer endpoints. Source widths and worst-case counts `2*N` must fit
QCX INT. A private position table has capacity `min(2*N, 1000000)`; a reached
selection exceeding this capacity fails before the body. Runtime visits do not
consume unrolling budgets; generated code obeys the QCX output limit, and
enclosing/nested expansions retain their accounting. Temporary declarations are
hoisted and cleanup preserves caller-owned storage on success and failure.

Runtime ranges also support [plain and bitwise compound assignments](#runtime-bit-register-range-assignments).
They also support [copies into registers or selections](#copies-from-runtime-bit-register-ranges).
Other runtime range expressions, arithmetic compound range destinations,
omitted bounds, aliases,
computed register sources, further indexing of selections, and block-local
declarations remain unsupported. No new bra instruction or build option is needed.

## Angle bit-selection for loops

The converter supports `for bit name in selection` over the supported bit
selections of explicitly sized angles. Sources can be named variables or
constants, sized angle iterators, or angle-array elements. The iterator must be
scalar `bit`; `bit[1]`, `int`, and other iterator types are rejected.

Supported forms include:

```qasm
for bit b in a[0:3] { ... }
for bit b in a[3:-1:0] { ... }
for bit b in a[{3, 0, 3}] { ... }
for bit b in a[index] { ... }
for bit b in a[{i, j, i}] { ... }
for bit b in phases[element_index][{i, j}] { ... }
for bit b in phases[element_index][0:3] { ... }
for bit b in phases[element_index][bit_index] { ... }
for bit b in a[start:stop] { ... }
for bit b in phases[element_index][start:step:stop] { ... }
```

Static indices, inclusive ranges, and discrete sets retain the existing
selection rules, including negative indices and repeated source indices.
Bits are visited in selection order, rather than sorted by their original
indices. A runtime single INT/UINT index selects `bit[1]`, so its loop executes
exactly once. A runtime-valued discrete set captures each position once and
executes one iteration per entry, preserving repetitions. This follows the
combination of OpenQASM's
[classical-value bit-slicing rules](https://openqasm.com/language/types.html#classical-value-bit-slicing)
and [iteration over bit-register values](https://openqasm.com/language/classical.html#for-loops).
Runtime-dependent range sources are described [below](#runtime-dependent-angle-bit-range-loops).

The source is resolved before introducing the iterator binding. For an array
element, its element index is evaluated and checked once at loop entry, followed
by the selector entries in source order when runtime capture is needed.
These captured indices remain fixed even
if the body changes the variables used to compute them. Each reentry captures
them afresh. Same-name shadowing, such as `for bit a in a[0:3]`, is supported;
the outer binding is restored when the loop ends.

The converter uses live reads, consistent with its mutable array/register
loops: each iteration reads the current selected source bit and copies it into
the iterator. Changes to future bits affect later visits, including repeated
visits to the same bit. Changing the current source bit does not change the
copied iterator. The specification establishes the result type and iteration
order, but does not explicitly distinguish snapshot versus live reads for
angle bit selections; live reads are the converter's documented interpretation.

The iterator has the same scalar-bit behavior as other bit iterators: direct
conditions, logical expressions, comparisons, scalar-bit copies, and supported
explicit casts. Numeric uses require a cast such as `uint(b)`. It is scoped and
read-only, including when the source is constant. OpenQASM permits assignments
to loop variables; rejecting them remains a converter limitation. The source
angle retains its own mutability rules, so constant angles and outer angle
iterators cannot be modified.

<!-- angle-bit-selection-loop-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
angle[4] a = angle[4]("0001");
uint checksum = 0;
int visits = 0;
for bit b in a[{0, 1, 0, 3}] {
    if (visits == 0) { a[1] = "1"; a[0] = "0"; }
    if (visits == 1) { a[3] = "1"; }
    checksum = checksum * 3 + uint(b);
    visits += 1;
}
// Visits 1, 1, 0, 1: checksum is 37, visits is 4, a is "1010".
array[angle[4], 1] phases = {a};
int element_index = -1;
int bit_index = -1;
bit saved;
for bit b in phases[element_index][bit_index] {
    element_index = 0;
    bit_index = 0;
    phases[0][3] = "0";
    saved = b; // Still 1: b is a copy of the bit read before the body.
}
// One iteration; phases[0] is "0010", or pi/4.
h q;
p(phases[0]) q;
```

Bodies support nesting with existing range, set, array/register, selection, and
`while` loops. `break` exits the nearest loop; `continue` advances to its next
selected bit, or exits a single-iteration selection loop. Skipped branches and
empty enclosing loops do not execute runtime index checks or arithmetic, while
name, type, static-bound, and read-only validation still applies without
inventing iterator values. Invalid runtime indices fail before the body runs,
even if the body is empty.

The body is generated once, including for constant angle sources, using existing
QCX instructions. A private INT position table preserves selection order and
repeated indices; the source address, copied bit, and loop state remain reserved
throughout the body. The number of selected bits must fit QCX INT indexing.
Loop advancement stays within QCX INT's range, and temporary storage is released on
success and failed lowering without releasing caller-owned storage. Runtime
iteration counts do not consume expansion budgets; enclosing and nested
expanded loops retain their accounting. Gates preserve captured parameters
across source writes, including under fusion. No new bra instruction or build
option is required.

Unsized angles, computed angle sources or cast results, computed expressions
and runtime-dependent range expression reads on ordinary bit-register sources
outside the loop subsets above, runtime range destinations on angle sources, aliases,
multidimensional arrays, and block-local declarations remain outside this feature.

### Runtime-dependent angle bit-range loops

Explicitly sized angle sources also support `for bit b in a[start:stop]` and
`for bit b in phases[element_index][start:step:stop]` when any bound or step
depends on runtime storage or an enclosing iterator. Supported INT/UINT
expressions include arithmetic, integer-array elements, and integer-valued
mathematical functions. Names follow declaration order and iterator shadowing.
Both bounds are required. The omitted step is `+1`; descending ranges require
an explicit negative step, and zero is invalid.

The range generates `start`, `start + step`, and subsequent integer positions
up to the inclusive stop bound without passing it. Each generated negative
index then normalizes by adding the angle width. Endpoints are not normalized
before generating the progression. Consequently, for `angle[4] a`,
`a[-4:3]` visits positions `0, 1, 2, 3, 0, 1, 2, 3`. A stop need not itself
be selected: `a[0:3:4]` visits positions `0, 3`, so the unselected stop `4`
does not cause a bounds failure. These rules match static angle selections.

At each loop entry, the converter resolves the source and captures any array
element index first, then evaluates start, optional step, and stop once in
source order, before binding the bit iterator. It constructs and validates the
complete position sequence before executing the first body. A zero step,
empty selection, or selected position outside `[-n, n - 1]` fails on the
executed path, even if the body is empty or immediately executes `break`.
Unlike an ordinary integer `for` range, an empty selection is invalid because
it indexes a classical value. Entirely constant ranges retain conversion-time
validation. Names, types, and unsupported syntax are checked even in skipped
bodies and empty enclosing loops without fabricating iterator values.

Range operands retain their evaluated signed or unsigned integer values.
Full-width UINT bounds and steps are not narrowed to INT. Advancement preserves
the mathematical progression without wrapping at native integer endpoints;
it terminates when the next position passes the stop. Operands must use the
supported native INT/UINT domains and existing expression rules. For a value
above INT's maximum, use a supported UINT declaration or explicit UINT cast.

Captured positions and the source element address stay fixed throughout the
loop; body changes to bound variables or the element-index variable do not
change that sequence. Source bits are read live at each visit, and the current
bit is copied to the read-only iterator. Reentry captures a fresh sequence.
This capture timing and live-read policy are converter semantics; the
specification does not explicitly settle them for angle bit selections.
Nested loops, same-name shadowing, `break`, and `continue` follow the rules above.

<!-- runtime-angle-bit-range-loop-example -->
```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
array[angle[4], 2] phases = {angle[4]("0011"), pi};
int element = 0;
int start = -4;
int step = 1;
int stop = 3;
int visits = 0;
uint ones = 0;
bit saved = 0;
for bit b in phases[element][start:step:stop] {
    if (visits == 0) {
        element = 1;
        start = 100;
        step = 0;
        stop = -100;
        phases[0][0] = "0";
        phases[0][3] = "1";
        saved = b; // Still 1: the iterator holds a copy.
    }
    ones += uint(b);
    visits += 1;
}
// Visits 1, 1, 0, 1, 0, 1, 0, 1: ones is 5 and visits is 8.
// The captured source remains phases[0], now "1010", or 5*pi/4.
h q;
p(phases[0]) q;
```

The body is generated once. A bounded private position table holds the validated
sequence: a nonzero-step range contains at most `2*n` valid raw indices for
an `angle[n]`, including repeated positions after negative-index normalization.
The runtime iteration count does not consume unrolling budgets. Existing QCX
instructions suffice; no bra instruction or build-option change is required.
General runtime-width selection expressions and assignments and bit selections
on unsized angles remain unsupported. Ordinary bit-register ranges have their
own [loop-source subset](#runtime-dependent-bit-register-range-loops).

## Runtime while loops

The converter supports `while (condition)` with either a single-statement or
braced body. The condition uses the same supported subset as `if`: scalar
Boolean or bit values, individual bit-register elements, supported comparisons,
logical `!`, `&&`, and `||`, and supported explicit Boolean casts. Direct
integer, floating-point, complex, and whole-bit-register conditions remain
unsupported. Logical expressions retain short-circuit evaluation.

```qasm
OPENQASM 3.0;
int n = 0;
int total = 0;
while (n < 6) {
    n += 1;
    if (n == 2) { continue; }
    if (n == 4) { break; }
    total += n;
}
// n is 4 and total is 4 after execution: 1 + 3.
```

Unlike constant-range `for` loops, a `while` body is generated once and executes
at runtime. The converter emits a condition label, condition evaluation using
existing QCX `JUMP`/`JUMPIF` instructions, a body label, a jump back to the
condition, and an exit label. The condition, including any arithmetic needed to
compute it, is reevaluated before every iteration. The body executes zero times
if the initial condition is false.

Nested `while` loops and combinations with supported `for` loops are allowed.
`break` exits the nearest enclosing loop. In a `while` loop, `continue` jumps
back to its condition, rather than directly to its body. An inner transfer does
not affect an outer loop. Bodies may use supported gates, global phase,
measurement, reset, barriers, assignments to existing variables, and conditionals.
Temporary declarations are moved before control flow when needed; computations
remain in their original condition or body positions.

Measurement results can control termination:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit outcome = 0;
int visits = 0;
while (!outcome) {
    visits += 1;
    if (visits == 2) { x q; }
    outcome = measure q;
}
// The loop terminates after two iterations with outcome equal to 1.
```

General block-local declarations, runtime qubit indexing, and
writes to active `for` iterators remain
unsupported. A `while` body is still converted and validated even when its
condition is literally false or a transfer makes later statements unreachable. Conversion-time
constant evaluation still applies; skipped runtime computations are not executed.
The amplitude-output pragma continues to request output after all operations,
not at an intermediate loop position.

The expansion limits above concern generated `for` iterations and instructions,
not the number of runtime `while` iterations. A nested `for` is expanded once
per generated `while` instance, regardless of how often that instance executes.
A `while` nested in an expanded `for` contributes its generated statements and
lines to the enclosing expansion limits. There is no runtime iteration limit
or automatic infinite-loop detection; users must ensure termination. Final
amplitude output is reached only if the program terminates.

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

- delays, iteration over arrays other than the supported
  `int`/`uint`/`float`/`bool`/`complex`/`angle` subset,
  alias iteration, or `switch` statements;
- user-defined gates, gate durations, runtime floating-point `pow` exponents,
  mixed runtime-integer/floating-point power chains, floating-point powers
  outside the fixed, phase, Pauli-rotation, and `u2`/`u3`/`U` families above, or
  runtime-dependent control counts;
- runtime qubit indices, dynamic register ranges or discrete sets outside the
  supported selection subsets described above, ranges with
  omitted bounds, or multidimensional indexing;
- classical arrays beyond the one-dimensional `int`, `uint`, `float`, `bool`,
  `complex`, and `angle` subsets above, zero-length arrays,
  whole-register casts other than the same-width
  angle/bit-register and UINT/bit-register conversions above, complex-to-Boolean casts,
  Boolean arithmetic, mixed Boolean/numeric comparisons, or block-local
  declarations;
- scalar-bit shifts, runtime bit-register range expressions outside the supported
  iteration, copy, whole-register assignment/initializer, static-selection assignment,
  runtime index-set assignment, runtime-range destination assignment, UINT/ANGLE cast, population-count,
  comparison, and bitwise compound-assignment RHS expression subsets,
  runtime bit-register range arithmetic compound assignments,
  whole-array bitwise operations, direct
  Boolean bitwise operations, or classical functions other than the
  `exp`/`sqrt`/`log`/`sin`/`cos`/`tan`/`arcsin`/`arccos`/`arctan`/`ceiling`/`floor`/
  `mod`/`popcount`/`rotl`/`rotr`/`real`/`imag` subset above;
- angle-by-angle multiplication, scaling by types other than same-width UINT,
  `uint / angle`, `angle /= angle`, mixed angle/non-angle comparisons and bitwise
  operations, compound operators other than `+=`/`-=`/`*=`/`/=`/`&=`/`|=`/`^=`/`<<=`/`>>=`,
  runtime-dependent angle bit ranges outside scalar-bit iteration, ANGLE/UINT casts,
  and direct whole-register copies,
  bit selections on unsized angles, arithmetic compound
  assignments to angle bit selections, INT/UINT bit slicing, multidimensional angle arrays,
  iteration over angle-valued ranges/sets,
  narrowing angle-array iterators,
  numeric casts from angles, angle-valued gates outside
  `p`/`phase`/`u1`/`cp`/`cphase`/`gphase`; or
- arithmetic operators other than `+`, `-`, `*`, `/`, and integer `%` and `**`.

Integer limitations additionally include `uint[n]` widths above native `UINT`,
UINT range/set iterators, narrowing UINT array iterators,
fixed-width signed reinterpretation casts from UINT,
mixed signed/unsigned expressions with non-native signed widths, and general
signed-width emulation or signed-arithmetic overflow checks.

The characterization tests in `bra/test/test_qasm2qcx.py` define the working
baseline. `bra/test/qasm2qcx_if_else_numerical.py` additionally converts and
executes deterministic measurement-controlled programs with `bra`, verifies
logical truth tables for conditions and values, and checks Boolean conversions,
self-referencing assignments, short-circuit evaluation, and temporary reuse.
It also executes the complete Boolean example above. These tests cover the
converter's supported Boolean subset; the limitations above remain explicit
future work.

`bra/test/qasm2qcx_angle_numerical.py` checks exact stored angle bits for constant,
literal, and runtime inputs, width conversions, ties and adjacent values,
wraparound, huge finite values, nonfinite failures, Boolean casts, control flow,
array inputs, iterator shadowing, and skipped execution. It executes the scalar
angle example above. Unit tests additionally check declaration visibility,
read-only constants, unsupported uses, empty-loop validation, and temporary
cleanup. `bra/test/classical_function_angle_numerical.py` checks native
`ANGLEBITS` against an independent exact-rational oracle across float, double,
and long-double REAL builds in debug, release, and mixed-release modes. It covers
rounding directions, stored subnormals, invalid inputs, and unchanged
floating-point controls; `--sanitize` repeats the matrix with UBSan.

`bra/test/qasm2qcx_angle_array_numerical.py` checks exact native UINT storage
against integer-pattern and rational-quantization references, without converting
outputs to floating point. It covers every angle width, singleton and unsized
arrays, float and angle initializers, width conversion, static/runtime and
negative indices, all supported compound assignments, mixed-width narrowing,
self-references, bounds failures, and source preservation. Integration cases
include casts, functions, loops, iterator shadowing, skipped domains, and phase
gates with modifiers and fusion. The documented angle-array example is executed
normally and with fusion. Unit tests additionally cover declaration and type
errors, empty-body validation, once-only destination capture, and temporary
cleanup. Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_angle_array_loop_numerical.py` checks live reads, copied
iterator values, future/current source mutation, reentry, nested shadowing, and
nearest-loop transfers. Exact raw-storage references cover every supported
element-to-iterator widening pair, all supported widths,
arithmetic, comparisons, shifts, bit functions, casts, and cross-width promotion.
Integration cases include trigonometric functions, dependent integer loops,
mixed iterator types, reached/skipped errors, phase gates, modifiers, and fusion.
The complete angle-array-loop example above is executed normally and with fusion.
Unit tests additionally cover width/type restrictions, source visibility,
read-only iterators in skipped bodies, expansion budgets, CLI failure output,
and cleanup. Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_angle_arithmetic_numerical.py` checks exact stored bits at
every supported width, mixed-width promotion and destination narrowing,
wraparound, negation, self-referencing and compound assignments, source
preservation, executable casts, and repeated execution in branches and loops.
It covers measurement-controlled updates, iterator shadowing, reached/skipped
errors, short-circuiting, and complete phase-gate matrices with modifiers,
broadcasts, zero qubits, and fusion. It also executes both arithmetic examples
above. Unit tests cover exhaustive four-bit arithmetic, constant evaluation,
sized/unsized metadata, validation, operand capture, cleanup, and declaration
hoisting. No new native bra arithmetic is required.

`bra/test/qasm2qcx_angle_multiplicative_numerical.py` checks exact products,
unsigned quotients, wrapping, and truncation at every supported width, mixed-width
angle ratios, UINT result composition, compound assignments, operand preservation,
array factors, iterator shadowing, and repeated execution. It checks reached and
skipped zero divisions, including executable constant operands, short-circuiting,
and empty loops. Full phase matrices cover modifiers, broadcasting, positive and
negative controls, global phase, zero qubits, and fusion. Both multiplicative
examples above are executed. Unit tests additionally cover exhaustive four-bit
arithmetic, sizedness, constant evaluation, invalid pairs and widths on skipped
paths, once-only operand capture, and temporary cleanup on success and failure.
Run the numerical driver with `--bra /path/to/bra`; it can also use a UBSan build.

`bra/test/qasm2qcx_angle_comparison_numerical.py` checks all six comparisons
against independent aligned-integer results at every supported width, including
exhaustive four-bit truth tables, mixed-width equality and ordering, wraparound,
and sized/unsized operands. It covers Boolean destinations, arithmetic operands,
iterator shadowing, repeated conditions, `break`/`continue`, measurement-controlled
branches, and reached/skipped operand errors. Register comparisons with angle
quotients as shift/rotation counts retain their existing register semantics.
Full phase matrices check
comparison-controlled gates, global phase, negative controls, powers, zero
qubits, and fusion. Both comparison examples above are executed. Unit tests
additionally check constant folding, invalid mixed operands on skipped paths,
once-only capture, declaration hoisting, and cleanup with caller-owned temporaries.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_angle_bitwise_numerical.py` checks exact complements and binary
bit patterns at every supported width, mixed-width alignment, destination
narrowing, sized/unsized values, self-reference, and compound assignments. It
covers repeated execution, iterator shadowing, measurement-controlled updates,
angle arithmetic and comparisons, and reached/skipped operand errors. Full phase
matrices cover the supported phase-gate aliases, modifiers, negative controls,
global phase, zero qubits, and fusion. Both bitwise examples above are executed.
Unit tests additionally check exhaustive four-bit constants, sizedness metadata,
once-only operand capture, invalid mixed operands, read-only targets, declaration
hoisting, and temporary cleanup. Run with `--bra /path/to/bra`, including a UBSan
build.

`bra/test/qasm2qcx_angle_shift_numerical.py` checks exact shift results at every
supported angle width, count boundaries, signed and unsigned counts, UINT_MAX,
huge literals, count widths, sizedness, source preservation, and self-referencing
compound assignments. It covers repeated execution, iterator shadowing,
measurement-controlled updates, skipped/reached errors, and integration with
arithmetic, bitwise operations, comparisons, and phase gates. Complete phase
matrices check modifiers, negative controls, global phase, zero qubits, and
fusion. Both shift examples above are executed. Unit tests additionally cover
constant evaluation, invalid types and widths on skipped paths, once-only
capture, declaration hoisting, and cleanup with caller-owned temporaries.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_angle_bit_function_numerical.py` checks angle rotations and
population counts at every supported width with an independent bit-string
oracle, including exhaustive four-bit runtime rotations. It covers signed
distance boundaries, checked UINT-to-INT conversions before normalization,
sizedness, nested functions, self-assignment, arithmetic, casts, comparisons,
and repeated execution. Iterator shadowing, measurement-controlled updates,
reached/skipped operand errors, and full phase matrices are checked, including
runtime popcount exponents, negative controls, global phase, zero qubits, and
fusion. Both angle bit-function examples above are executed. Unit tests also
check once-only capture, temporary cleanup, invalid signatures on skipped paths,
and metadata when a reused expression changes between angle and UINT bindings.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_angle_bit_cast_numerical.py` verifies exact patterns and
individual little-endian bits, without floating-point reference conversions.
It covers all angle widths, exhaustive small widths, native-width boundaries,
deterministic random patterns, nested round trips, source preservation,
overlapping destinations, register expressions, static selections, loops,
iterator shadowing, measurement sources, and reached/skipped operand errors.
Phase-gate and trigonometric parameters are checked with and without fusion,
including the documented example above. Unit tests additionally check constant
evaluation, width/type rejection, empty-body validation, once-only capture,
temporary ownership and cleanup, and CLI errors without partial output.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_angle_cast_numerical.py` checks exact ANGLE
storage against independent little-endian bit vectors at every supported width.
It covers mutable and constant sources wider than native UINT, sized and unsized
targets, negative aliases, reverse and strided ranges, nested register operations,
fixed-width angle-bit leaves, indexed array destinations, overlapping round trips,
fresh loop evaluations, and scalar consumers. Reached failures check operand
order, absence of partial destination writes, and preservation of earlier scalar
and array assignments. Skipped domains, measurement feedback, phase-gate
modifiers, trigonometric arguments, and fused execution are also checked, along
with the runtime-range ANGLE example above. Run with `--bra /path/to/bra`,
including a UBSan build.

`bra/test/qasm2qcx_runtime_angle_range_cast_numerical.py` audits exact casts
from runtime angle-bit ranges using independent bit-vector references. It covers
source forms, nested expressions, capture and failure order, overlapping writes,
loops, measurement, and gate parameters with and without fusion. Run with
`--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_angle_range_uint_cast_numerical.py` checks exact UINT
results from runtime angle-bit ranges against independent bit-vector references.
It covers all supported widths, source forms, nested trees, array destinations,
overlap, loops, skipped paths, and failure order. Run with `--bra /path/to/bra`,
including a UBSan build.

`bra/test/qasm2qcx_runtime_angle_selection_uint_cast_numerical.py` checks UINT
casts from runtime angle-bit indices and sets against independent bit-vector
references. It covers all supported widths, duplicate and negative indices,
arrays, iterators, nested expressions, overlap, skipped paths, and failure
ordering. Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_angle_range_register_assignment_numerical.py` checks
direct copies against independent bit-vector references. It covers source and
destination widths, including destinations beyond native UINT, repeated
positions, arrays, iterators, skipped paths, and failure ordering. Run with
`--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_angle_range_expression_numerical.py` checks mixed
angle and bit-register expression trees against independent bit-vector
references. It covers source widths, repeated positions, wide results, aliases,
iterators, skipped paths, and failure order. Run with `--bra /path/to/bra`,
including a UBSan build.

`bra/test/qasm2qcx_angle_range_static_selection_numerical.py` checks runtime
angle-range expression assignments to static selections against independent
bit-vector references. It covers ordered selections, wide results, overlap,
iterators, skipped paths, and failure order. Run with `--bra /path/to/bra`,
including a UBSan build.

`bra/test/qasm2qcx_angle_range_runtime_set_numerical.py` checks runtime
angle-range expressions assigned to runtime index sets against independent
bit-vector references. It covers changing and overlapping destinations,
wide selections, skipped paths, and failure order. Run with `--bra /path/to/bra`,
including a UBSan build.

`bra/test/qasm2qcx_angle_range_runtime_range_numerical.py` checks runtime
angle-range expressions assigned to runtime bit ranges against independent
bit-vector references. It covers changing and overlapping ranges, wide results,
skipped paths, and failure order. Run with `--bra /path/to/bra`, including a
UBSan build.

`bra/test/qasm2qcx_angle_range_register_compound_numerical.py` checks
whole-register bitwise compounds with runtime angle ranges against independent
bit-vector references. It covers all three operators, overlap, wide results,
loops, skipped paths, and failure order. Run with `--bra /path/to/bra`,
including a UBSan build.

`bra/test/qasm2qcx_angle_range_static_compound_numerical.py` checks static
selection compounds with runtime angle ranges against independent bit-vector
references. It covers all three operators, selection order, overlap, loops,
skipped paths, and failure order. Run with `--bra /path/to/bra`, including a
UBSan build.

`bra/test/qasm2qcx_angle_range_runtime_set_compound_numerical.py` checks runtime
index-set compounds with runtime angle ranges against independent bit-vector
references. It covers all three operators, changing and overlapping destinations,
skipped paths, and failure order. Run with `--bra /path/to/bra`, including a
UBSan build.

`bra/test/qasm2qcx_angle_range_runtime_range_compound_numerical.py` checks runtime
range compounds with runtime angle ranges against independent bit-vector
references. It covers all three operators, changing and overlapping destinations,
skipped paths, and failure order. Run with `--bra /path/to/bra`, including a
UBSan build.

`bra/test/qasm2qcx_static_angle_bit_selection_numerical.py` checks selected bits
and packed angle storage against independent integer references at every
supported width. It covers constants, variables, runtime array-element indices,
negative indices, sparse/reversed/repeated selections, register expressions,
casts, and overlapping writes that preserve unselected bits. Integration cases
include loops, shadowing, reached/skipped access errors, phase gates, modifiers,
and fusion. The documented selection example above is checked for its stored
patterns and final quantum state, normally and with fusion. Unit tests also
cover static-selector restrictions, `bit[1]` typing, read-only sources, empty
loops, once-only destination capture, temporary ownership, and failed lowering
without partial writes or CLI output. Run with `--bra /path/to/bra`, including
a UBSan build.

`bra/test/qasm2qcx_angle_selection_compound_numerical.py` checks all five
compound operators against independent bit-list references across supported
widths. It covers sparse/reversed selections, overlapping operands, preserved
unselected bits, runtime array indices, constant/runtime counts, negative and
oversized counts, UINT_MAX, reentry, transfers, shadowing, and reached/skipped
errors. Complete phase-gate outputs are checked for scalar and array targets,
with modifiers and fusion, including the compound example above. Unit tests
additionally verify operand typing, empty-loop validation, once-only destination
and count capture, temporary ownership, capture-map cleanup, and failed
lowering without destination writes. Run with `--bra /path/to/bra`, including
a UBSan build.

`bra/test/qasm2qcx_runtime_angle_bit_numerical.py` checks runtime single-bit
reads and all six assignment operators against independent integer references
at every supported angle width. Cases cover constants, variables, array
elements, INT/UINT and negative indices, nested accesses, register consumers,
loop reentry and shadowing, overlapping operands, unchanged unselected bits,
reached/skipped bounds and arithmetic errors, and negative/oversized shift
counts. Phase gates, modifiers, and fusion are checked against full quantum
states. The runtime-selection example above is checked for its stored patterns
and final state. Unit tests additionally check index evaluation order and
once-only destination capture, old-bit snapshots before RHS reads, visibility,
read-only targets, temporary ownership, and capture-map cleanup on failures.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_angle_bit_selection_loop_numerical.py` checks selection order,
repeated bits, copied iterator values, and live source mutation at every supported
angle width. It covers constants, scalar variables, angle iterators, array
elements, static selectors and runtime single indices, once-only index capture,
shadowing, nested loops, reentry, nearest-loop transfers, reached/skipped errors,
and INT/UINT boundaries. Full quantum states check phase gates, modifiers, and
fusion, including the documented loop example above. Unit tests additionally
check body-once lowering, expansion accounting, source-order visibility, typed
validation in empty loops, read-only bindings, declaration hoisting, and storage
cleanup after index-capture and body failures. Run with `--bra /path/to/bra`,
including a UBSan build.

`bra/test/qasm2qcx_runtime_angle_bit_set_numerical.py` checks runtime-valued
set reads, plain/compound writes, and iteration at every supported angle width.
Independent integer references cover mixed INT/UINT and constant entries,
negative indices, repeated reads, selected registers wider than the source,
full-width writes, overlapping operands, preserved bits, and shift boundaries.
Cases also cover normalized duplicate destinations before RHS failures,
reached/skipped bounds and domains, oversized literals, nested loops, shadowing,
captured selector variables, live reads, and transfers. Complete quantum states
check phase gates, modifiers, and fusion, including the example above. Unit tests
additionally check fixed-width metadata, visibility, empty-loop validation,
once-only ordered capture, old-bit snapshots, single destination commits, and
cleanup after partial capture, RHS, duplicate-check, and update failures.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_angle_bit_range_loop_numerical.py` checks complete
range capture and executable loops against independent integer references.
Cases cover every supported angle width, ascending/descending ranges, omitted
steps, negative indices, repeated positions, unselected stop bounds, empty
selections, signed endpoints, and full-width UINT bounds and steps. It also
checks once-only capture, live reads, source and iterator shadowing, nested
loops, reentry, transfers, and reached/skipped bounds and arithmetic failures
before body execution. Complete quantum states check phase gates, inverse and
floating-point power modifiers, fusion, and the documented example. Unit tests
additionally check validation without values, declaration hoisting, expansion
accounting, reserved names, and storage cleanup after partial capture or body
failures. Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_angle_phase_numerical.py` compares whole diagonal phase
matrices with the generated angle-phase circuits, without discarding global
phase. It covers all five names, basis states and complex superpositions,
widths and bit patterns, conversions, source preservation, broadcasting,
positive/negative controls, integer power boundaries and nested products,
loops, reached/skipped errors, and fusion. It also observes the powered private
UINT snapshot to verify exact modular results at every supported width, including
low bits hidden by amplitude printing. It executes the angle-phase example
above. Unit tests additionally check exact exponent emission, once-only
capture, declaration hoisting, temporary cleanup, zero-identity validation,
unsupported uses, CLI errors without partial output, and expansion budgets.

`bra/test/qasm2qcx_angle_global_phase_numerical.py` checks absolute global and
controlled relative phases without removing global phase. It covers zero-qubit
programs, constants and runtime angles, every supported width, bit patterns,
casts, mixed controls, broadcasting, integer power boundaries and nested
products, repeated entry, skipped/reached errors, source preservation, and
fusion. Exact powered-snapshot checks verify low bits hidden by amplitude
printing. It executes both global-phase examples above. Unit tests additionally
verify native REAL scaling, once-only capture, validation on skipped paths,
declaration hoisting, temporary cleanup, and output budgets.

`bra/test/qasm2qcx_angle_power_numerical.py` checks the complete matrices of
floating-powered angle phase gates against an independent exact-turn oracle.
It covers all six names, constants and runtime inputs, branch-cut neighbors,
ordered inverse/power chains, exact large integer factors, binary64 factors,
casts, source preservation, broadcasts, mixed control polarities, zero qubits,
classical control flow, reached/skipped errors, resource limits, and fusion.
It executes both floating-angle examples above. Unit tests additionally check
descriptor serialization, capture-once lowering, static limits, hoisted
declarations, cleanup, and output budgets. Native precision/rounding checks
are in `bra/test/anglebits2phase_numerical.py`; `--sanitize` enables UBSan.

`bra/test/qasm2qcx_math_function_numerical.py` compares constant, executable
literal, and variable calls, with independent trigonometric references for
complex exponentials. It checks promotions, component extraction, nesting,
source preservation, arrays, loops, short-circuiting, reached failures, gate
parameters with and without fusion, and the mathematical-function example
above. Without `--bra`, its two reference tests require only the standard
library. Unit tests additionally check overload selection, diagnostics,
empty-loop validation, once-only argument evaluation, and temporary cleanup.

`bra/test/qasm2qcx_sqrt_numerical.py` checks real and complex roots using
independent scaled Cartesian complex-root references, not `cmath.sqrt`.
It covers constant/literal/variable calls, promotions, signed zeros and branch
endpoints, nearby and extreme values, stored subnormal components, nesting,
self-assignment, arrays, source preservation, loops, reached/skipped failures,
complex negation, fusion, and the square-root example above. Without `--bra`,
its two reference tests require only the standard library. Native
`bra/test/classical_function_numerical.py` checks the native functions in float,
double, and long-double debug/release builds; `--sanitize` additionally runs
UBSan checks. These tests do not change the existing build flags or
floating-point control modes.

`bra/test/qasm2qcx_log_numerical.py` compares constant, executable literal,
and variable logarithms against independent high-precision standard-library
`Decimal.ln` references. It covers values near one, extreme normal values,
positive subnormals, input promotions, nesting, arrays, loops, source
preservation, reached/skipped errors, and gate parameters with and without
fusion, including the logarithm example above. Without `--bra`, its two
reference tests require only the standard library. Native LOG tests also
cover stored negative subnormals, destination preservation on failure, and
unchanged floating-point control modes in all three configured precisions.

`bra/test/qasm2qcx_trigonometric_numerical.py` checks constant, executable
literal, and variable calls against independent Decimal Taylor-series
references. High-precision argument reduction uses Machin's formula for π
and retains over 100 digits even for maximum-double inputs; it does not
reuse binary libm trigonometric functions as the reference. The audit covers
large arguments, zero and tangent-pole neighbors, signed zeros, subnormals,
promotions, nesting, arrays, loops, reached/skipped failures, source
preservation, fusion, and the trigonometric example above. Without `--bra`,
its two reference tests require only the standard library. Native tests also
check all three precisions against a separately compiled non-fast-math
library oracle and verify destination preservation and unchanged control modes.

`bra/test/qasm2qcx_angle_trigonometric_numerical.py` checks constant evaluation
against independent high-precision Decimal series over the full represented
angle, using Machin-formula π without the converter's quadrant reduction.
With `--bra`, it audits exact storage patterns at every angle width, exhaustive
small widths, axes and adjacent patterns, octants, deterministic random patterns,
and source preservation. Integration checks cover executable constants, nested
calls, resizing, rotations, arrays, conditions, loops, iterator shadowing,
reached/skipped errors, gate parameters, fusion, and the angle example above.
Without `--bra`, it runs the converter constant-reference checks. Unit tests
also cover overload selection, skipped validation, temporary ownership, and
the continued rejection of general angle-to-float casts.

`bra/test/qasm2qcx_inverse_trigonometric_numerical.py` checks constant,
executable literal, and variable calls against independent Decimal square-root
and arctangent-series references, with half-angle reduction. It does not reuse
binary libm inverse-trigonometric functions as the reference. Cases include
domain endpoints and their neighbors, signed zeros, subnormals, large finite
arctangents, promotions, nesting, arrays, loops, source preservation,
reached/skipped errors, gate parameters with and without fusion, and the
inverse-trigonometric example above. Without `--bra`, its two reference tests
require only the standard library. Native tests additionally verify strict
domain checks, principal ranges, captured inputs, and unchanged destinations
on failure in all three precisions, debug/release builds, and UBSan.

`bra/test/qasm2qcx_rounding_numerical.py` checks constant, executable literal,
and variable calls against exact integer-ratio references, not `math.ceil`
or `math.floor`. Cases cover fractions, integer-boundary neighbors, signed
zeros, stored subnormals, large finite values outside native integer ranges,
promotions, nesting, arrays, loops, source preservation, reached/skipped
errors, gate parameters with and without fusion, and the rounding example
above. Without `--bra`, its two reference tests require only the standard
library. Native tests also check every supported precision in debug/release
builds and UBSan, captured inputs, destination preservation, independence
from the caller's rounding direction, and unchanged floating-point modes.

`bra/test/qasm2qcx_mod_numerical.py` compares integer and real calls with exact
integer and binary-ratio references, not `math.fmod`. It checks constants,
executable literals, live variables, signs and signed zeros, native integer
endpoints, huge real ratios, subnormals, promotions and checked unsigned
conversions, nested calls, indexed self-assignment, conditions and runtime loops,
skipped failures, and gate parameters with and without fusion. Unit tests also
check once-only argument evaluation, reserved temporary lifetimes, reuse, and
cleanup when the second argument fails. `bra/test/mod_numerical.py` checks native
QCX calls independently of the converter. Native kernel tests cover all three
precisions in debug, optimized, and mixed-build modes, with and without UBSan;
the mixed mode verifies unoptimized code under release startup's DAZ/FTZ settings
without changing the production build configuration.

`bra/test/qasm2qcx_popcount_numerical.py` checks literal, constant, and live-register
counts, including registers wider than native `UINT`, static selections with
repeated positions, register expressions and runtime shifts, unsigned arithmetic,
array assignments, nested functions, loops, measurement, skipped shift errors,
and the example above with and without fusion. Unit tests also cover scalar
versus register typing, result-width metadata, non-evaluating validation, iterator
shadowing, and temporary cleanup on failure.
It also checks every supported sized-UINT width, exact source preservation at
native UINT_MAX, masked intermediate expressions, arithmetic promotions,
indexed/self-referencing assignments, runtime loop bounds and steps, typed UINT
array iterators, skipped and reached errors, and unsigned gate parameters with
and without fusion. The same driver can be run against optimized and UBSan bra
builds without native source or build-option changes.

`bra/test/qasm2qcx_runtime_range_popcount_numerical.py` checks direct runtime
ranges on mutable and constant sources through 129 bits. It covers varying
selection lengths, forward/reverse strides, negative indices, repeated normalized
positions, native INT/UINT operands, and advancement past native endpoints.
Scalar integration includes native result widths, narrow assignments, arithmetic,
casts, nested bounds, array destinations, fresh loop values, loop transfers,
short-circuiting, measurement feedback, and quantum parameters with and without
fusion. Reached failures verify validation before counting and destination writes,
including bounded position capacity. The runtime-range population-count example
above is executed. Unit tests also check non-evaluating metadata, source visibility,
once-only bound capture, declaration placement, temporary ownership, output limits,
and cleanup after failures. Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_popcount_complement_numerical.py` checks odd
and even complement chains at every native width and on sources through 129 bits.
The independent reference flips individual selected bits. Cases cover zero/all-one
patterns, runtime lengths, negative indices, repeated normalized positions, wide
constants, strides, native endpoint advancement, UINT result widths, and narrow
assignments. Integration checks include nested bounds and scalar calls, array
indices, fresh loop values, transfers, skipped errors, measurement feedback, and
gate parameters and powers with and without fusion. Reached failures verify
validation before counting and destination writes for both parities. The complement
example above is executed. Unit tests also check immutable parity metadata,
non-evaluating validation, single captures and source passes, caller state and
temporary cleanup, and output limits. Run with `--bra /path/to/bra`, including a
UBSan build.

`bra/test/qasm2qcx_runtime_range_popcount_shift_numerical.py` checks both shift
directions at every native width and on sources through 129 bits. Its independent
reference uses arbitrary-width Python integer shifts on selected patterns. Cases
cover varying lengths, negative indices, repeated positions, native and huge
literal counts, guarded UINT conversion, narrow assignments, nested scalar
counts and bounds, array consumers, fresh loop values, transfers, and skipped
errors. Reached failures check range-before-distance ordering and absence of
source counting or destination writes. Measurement feedback, gate parameters,
and powers are checked with and without fusion. The shifted-count example above
is executed, including a successful call after an oversized-count branch. Unit
tests also cover metadata, source visibility, single captures, declaration
placement, temporary ownership, output limits, and failure cleanup. Run with
`--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_popcount_complement_shift_numerical.py` checks
all inner/outer parity combinations, both shift directions, every native width,
and sources through 129 bits. Its independent reference flips individual bits,
shifts with zero fill, then flips again. Cases cover varying selected lengths,
reverse strides, negative indices, repeated normalized positions, native endpoint
and huge literal counts, narrow assignments, nested counts and bounds, array
consumers, fresh loop values, transfers, skipped errors, measurement feedback,
and normal/fused gate parameters and powers. Reached failures verify range/count
checks before source counting and destination writes. The mixed-complement
example above is executed, including storage reuse after an oversized shift.
Unit tests also check independent immutable parities, original AST nodes,
non-evaluating validation, single capture/evaluation, temporary ownership,
declaration placement, output limits, and failure cleanup. Run with
`--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_popcount_rotation_numerical.py` checks both
rotation directions, all complement parities, every native width, and sources
through 129 bits. Its independent reference flips individual bits, circularly
permutes them, then flips again. Cases cover negative and endpoint distances,
checked UINT conversions, Boolean promotions, reverse strides, negative indices,
repeated normalized positions, narrow assignments, nested calls and bounds,
array consumers, fresh loop values, transfers, skipped errors, measurement
feedback, and normal/fused gate parameters and powers. Reached failures check
range/distance validation before source counting and destination writes. The
rotation/popcount example above is executed. Unit tests additionally check
immutable metadata, source visibility, single capture/evaluation, native result
typing, temporary cleanup, declaration placement, and output limits. Run with
`--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_popcount_nested_shift_numerical.py` checks all
two-shift direction combinations and endpoint complement parities, every native
width, and sources through 129 bits. Its independent reference constructs every
intermediate bit vector with zero fill. Cases include exhaustive small vectors
and distances, chains up to 16 shifts, reverse strides, negative indices,
repeated normalized positions, native endpoints, huge literals, and storage
reuse after empty survivor windows. Further cases cover nested calls, bounds,
arrays, narrow results, fresh loop values, transfers, skipped errors,
measurement feedback, and normal/fused gate parameters and powers. Reached
failures verify range validation and later-distance errors after complete bit
loss before source counting or destination writes. The nested-shift example
above is executed. Unit tests additionally check original AST nodes, ordered
metadata validation, single capture/evaluation, variable comparison operands,
declaration placement, temporary cleanup, output limits, and unsupported forms.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_popcount_interleaved_complement_numerical.py`
checks all complement placements and shift directions, every native width, and
sources through 129 bits. Its independent reference constructs every
intermediate bit vector, applying zero fill and complements directly. Cases
include exhaustive small vectors and distances, even complement groups, chains
up to 16 shifts, generated ones after complete original-bit loss, reverse
strides, negative indices, repeated normalized positions, native endpoints,
huge literals, and storage reuse. Further cases cover nested calls, bounds,
arrays, narrow results, fresh loop values, transfers, skipped errors,
measurement feedback, and normal/fused gate parameters and powers. Reached
failures verify range and distance errors before main-source counting or
destination writes, including errors in a third distance after complete bit
loss. The interleaved-complement example above is executed. Unit tests check
original AST nodes, ordered validation and capture, forward distance evaluation
before backward analysis, checked narrowing, variable comparison operands,
declaration placement, temporary cleanup, output limits, and unsupported forms.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_popcount_nested_rotation_numerical.py` checks
all two-rotation direction combinations and complement placements, every native
width, and sources through 129 bits. Its independent reference constructs and
rotates each intermediate bit vector, then applies complements directly. Cases
cover exhaustive small vectors and signed distances, chains up to 16 rotations,
even complement groups, INT endpoints whose sum would overflow, checked UINT
and Boolean promotions, reverse strides, negative indices, repeated normalized
positions, and source preservation. Further cases cover nested calls, bounds,
arrays, narrow results, fresh loop values, transfers, skipped errors,
measurement feedback, and normal/fused gate parameters and powers. Reached
failures check range validation and first/later distance errors before main
counting or destination writes, including third-distance failures. The
nested-rotation example above is executed. Unit tests additionally check
original AST nodes, ordered metadata validation, single capture/evaluation,
independent INT materialization, total complement parity, temporary cleanup,
declaration placement, output limits, and unsupported mixtures. Run with
`--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_popcount_mixed_operations_numerical.py` checks
mixed shift/rotation chains with complements using an independent reference
that materializes every intermediate bit vector. Cases cover both operation
orders and directions, exhaustive small vectors and complement placements,
native-width boundaries, sources through 129 bits, longer chains with several
surviving fragments, INT endpoints, oversized shifts, Boolean/UINT rotation
promotions, reverse strides, negative aliases, and source preservation.
Integration cases cover nested calls, bounds, casts, arrays, fresh loop values,
transfers, skipped errors, deterministic and superposed measurement feedback,
and normal/fused gate parameters and powers. Reached errors are checked before
main counting or destination writes, including later errors after complete bit
loss. The mixed-operation example above is executed. Unit tests check original
AST order, distinct distance rules, validation and capture order, bounded
interval storage, declaration placement, temporary cleanup, and output limits.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_popcount_bitwise_numerical.py` checks all
three bitwise operators and both operand orders against independent bit-vector
results. It covers exhaustive small patterns, every native width, sources
through 129 bits, mutable/constant sources, literal and computed fixed operands,
reverse/nonunit strides, negative aliases, singleton ranges, and source
preservation. Integration cases cover casts, arrays, nested consumers, fresh
loop values, captured loop bounds, transfers, skipped domains, measurement
feedback, and normal/fused gate parameters and powers. Reached errors check
operand order, range/count validation, later fixed-operand failures, selected
range reads, and destination-write suppression. The bitwise example above is
executed. Unit tests additionally check metadata validation, snapshot timing,
single evaluation, declaration placement, temporary cleanup, storage/output
limits, and unsupported forms. Run with `--bra /path/to/bra`, including a UBSan
build.

`bra/test/qasm2qcx_runtime_range_pair_popcount_numerical.py` checks all
operators and operand orders over two independently selected runtime ranges.
Its reference selects and combines little-endian bit vectors directly. Cases
cover exhaustive small patterns, every native width, sources through 129 bits,
all mutable/constant source combinations, different containing widths, reverse
and nonunit strides, negative aliases, singletons, and native step endpoints.
Integration cases cover nested consumers, arrays, narrowing casts, self-referencing
destinations, shrinking/fresh loop selections, captured loop bounds, iterator
shadowing, transfers, skipped errors, measurement feedback, and normal/fused
gate parameters and powers. Reached failures check left-snapshot completion,
right-read suppression, length mismatches, capacity limits, and destination-write
suppression. The range-pair example above is executed. Unit tests additionally
check metadata/capture order, bounded buffers, code size independent of mutable
source width, declaration placement, caller-state preservation, temporary
cleanup, output limits, and unsupported forms. Run with `--bra /path/to/bra`,
including a UBSan build.

`bra/test/qasm2qcx_runtime_popcount_bitwise_complement_numerical.py` checks
all left/right/result complement parities in the range/range and both
range/fixed-register operand orders, for every bitwise operator. Its independent
reference explicitly flips each bit vector for every complement, then combines
and counts. Cases cover exhaustive small patterns, native-width boundaries,
sources through 129 bits, all mutable/constant combinations, aliases, reverse
strides, singletons, longer even/odd chains, computed fixed operands, and source
preservation. Integration cases cover nested consumers, arrays, narrowing,
self-referencing destinations, shrinking/fresh loop selections, transfers,
skipped domains, measurement feedback, and normal/fused gate parameters and
powers. Reached errors verify capture order, right-read and destination-write
suppression, and retained validation for even parity. The complemented-bitwise
example above is executed. Unit tests additionally check original AST retention,
ordered metadata validation and capture, parity reduction, legacy fixed-operand
readers, caller state, scratch cleanup, declaration placement, and output limits.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_popcount_bitwise_result_shift_numerical.py` checks
outer shift/complement chains around all three binary operators for range/range
and both range/fixed-register operand orders. Its reference applies each
operation forward to little-endian bit vectors, including zero fill. Cases
cover exhaustive small patterns and complement parities, every native width
and wider selections through 129 bits, zero/width/oversized INT and UINT
distances, huge literals, longer mixed chains, aliases, strides, computed fixed
operands, nested distances, and source preservation. Integration cases cover
casts, arrays, self-referencing destinations, fresh/shrinking loop selections,
transfers, skipped domains, measurement feedback, and normal/fused gate
parameters and powers. Reached errors verify operand-before-distance ordering,
later errors after oversized shifts, capacity checks, and destination-write
suppression. The shifted-bitwise example above is executed. Unit tests also
check original AST retention, ordered validation/capture, bounded storage,
declaration placement, scratch cleanup, output limits, and deferred forms.
Independent cases are batched to stay within per-program output limits.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_popcount_bitwise_result_rotation_numerical.py` checks
outer rotation/shift/complement chains around all three binary operators in
range/range and both range/fixed-register operand orders. Its independent
reference applies each operation forward to individual little-endian bits.
Cases cover exhaustive small patterns and complement parities, every native
width and wider selections through 129 bits, negative/zero/width-multiple
rotation distances, native INT endpoints, oversized shifts, long mixed chains,
aliases, strides, source kinds, computed fixed operands, nested distances, BOOL
promotion, and source preservation. Integration cases cover arrays, casts,
self-referencing destinations, shrinking/fresh loop selections, loop conditions
and transfers, skipped domains, measurement feedback, and normal/fused gate
parameters and powers. Reached errors verify complete operand capture before
distances, checked UINT-to-INT conversion even for pure rotations and after bit
loss, later distance errors, capacity checks, and destination-write suppression.
The rotated-bitwise example above is executed. Unit tests additionally check
original AST retention, metadata/capture order, bounded interval storage,
declaration placement, scratch cleanup, output limits, and deferred forms.
Independent cases are batched below per-program output limits. Run with
`--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_popcount_operand_chain_numerical.py` checks independent
operand chains and mixed outer-result chains for all three binary operators,
with two runtime ranges or a range and fixed register in either order. Its
independent reference transforms each little-endian bit vector forward before
combining and counting. Cases cover exhaustive small patterns and complement
parities, native-width boundaries and selections through 129 bits, aliases,
strides, source kinds, INT endpoints, oversized shifts, long independent
chains, computed fixed operands, nested distances, BOOL promotions, and source
preservation. Integration cases cover arrays, casts, self-referencing
destinations, shrinking/fresh loop selections, loop conditions and transfers,
skipped domains, measurement feedback, and normal/fused gate parameters and
powers. Reached errors verify operand completion order, selected-length checks
before reads and inner distances, later errors after bit loss, checked rotation
conversions, capacity limits, outer-distance order, and destination-write
suppression. The operand-chain example above is executed. Unit tests also check
original AST retention, metadata/capture order, bounded buffers, generated code
size independent of pair source capacities, declaration placement, caller-state
preservation, temporary cleanup, output limits, and deferred trees. Independent
cases are batched below per-program output limits. Run with `--bra /path/to/bra`,
including a UBSan build.

`bra/test/qasm2qcx_runtime_popcount_tree_numerical.py` checks left-deep,
right-deep, balanced, and seeded deeper trees against independent little-endian
bit-vector references. It covers exhaustive small patterns, all binary operator
pairs and complement parities, native widths and wider selections through 129
bits, aliases, strides, source kinds, intermediate operations, INT endpoints,
oversized shifts, BOOL promotions, nested counts, source preservation, and fresh
reads. Integration cases cover arrays, casts, self-referencing destinations,
shrinking loop selections, loop conditions and transfers, skipped domains,
measurement feedback, and normal/fused gate parameters and powers. Reached
errors verify subtree completion, length checks after both children, later
errors after bit loss, checked conversions, capacity limits, parent-distance
order, and destination-write suppression. The tree example above is executed.
Unit tests additionally check original AST retention, ordered metadata/capture,
capacity bounds independent of runtime lengths, child-buffer reuse, caller and
consumer storage ownership, declaration placement, cleanup after failures,
output limits, and computed-result indexing rejection. Independent cases are
batched below per-program output limits. Run with `--bra /path/to/bra`, including
a UBSan build.

`bra/test/qasm2qcx_angle_range_popcount_numerical.py` checks direct and nested
runtime angle-range operands against independent bit-vector references. It
covers mixed trees, source widths, repeated positions, wide counts, nested
calls, loops, skipped paths, and failure ordering. Run with `--bra /path/to/bra`,
including a UBSan build.

`bra/test/qasm2qcx_constant_scalar_bit_numerical.py` checks scalar constant
bit truth tables, runtime copies, indexed destinations, casts, array and nested
function integration, short-circuiting, loops and transfers, iterator shadowing,
measurement-controlled execution, source preservation, and gate parameters
with and without fusion. It executes the scalar constant-bit example above.
Unit tests additionally cover declaration visibility, scalar/register distinctions,
invalid initializers, writes on skipped paths, and temporary cleanup.

`bra/test/qasm2qcx_constant_bit_register_numerical.py` checks named constant
registers through 129 bits, folded expressions and selections, runtime copies
and mixed expressions, scalar extraction, loop shadowing, short-circuiting,
source preservation, reached errors, gate parameters, measurement-controlled
execution, and the constant-register example with and without fusion.
Unit tests additionally check declaration visibility, immutability on skipped
paths, width/shape validation, and temporary cleanup.

`bra/test/qasm2qcx_rotation_numerical.py` compares constant and runtime rotations
against a textual circular-rotation reference. It covers native and narrow UINT,
wide bit registers, negative and boundary distances, nested calls, selections,
overlapping assignments, arithmetic promotions, loops, measurement and fusion,
skipped and reached errors, and the rotation example above. Unit tests check
width metadata, empty-loop validation, temporary ownership, and failure cleanup.

`bra/test/qasm2qcx_u_phase_numerical.py` independently transcribes the published
built-in `U` matrix and the mathematical active block of `cu`. It verifies the
native QCX `U2`/`U3` matrices and converter aliases, absolute and controlled
relative phases, constant/runtime parameters, indexed expressions,
broadcasting, inversion, mixed controls, and constant/runtime integer powers.
It also executes the U-family phase example above and records the conflicting
upstream `cu` definition explicitly. Without `--bra`, only its three reference
tests run; backend-dependent tests are skipped.

`bra/test/qasm2qcx_inverse_numerical.py` checks each supported gate against
independently calculated complex matrices on every three-qubit basis state and
a complex superposition. It covers odd/even repeated inversions, runtime
parameters, `gphase`, single- and controlled-gate broadcasting, and the published
forward `U`/`cu` phase conventions described above. Amplitudes are compared
directly, without removing global phase. Integration cases cover every supported
loop type, nested transfers, measurement-controlled branching, skipped runtime
arithmetic/access checks, executed failures, and source-variable preservation.
Unit tests additionally verify mnemonic coverage and involution, parameter and
phase snapshot separation, temporary hoisting, cleanup preserving caller storage,
unsupported modifiers and durations, empty-loop validation, and CLI errors
without partial output. Each numerical execution has a timeout.

`bra/test/qasm2qcx_control_numerical.py` compares each supported gate on every
basis state and a complex superposition with independently conditioned base
matrices. It covers one/two added controls, repeated controls, constant counts,
runtime parameters, already-controlled gates, operand permutations, `gphase`,
and control/target/paired-register broadcasting. The references preserve the
published `U`/`cu` matrices and compare amplitudes without discarding
global or relative phase. Integration cases cover all supported loop types,
nested transfers, indexed parameters, measurement-controlled branches, skipped
checks, source preservation, and executed failures even with inactive quantum
controls. Unit tests additionally cover modifier/count errors, empty loops,
iterator shadowing, arity, duplicate operands, parameter snapshots, temporary
hoisting, caller-storage preservation, and CLI errors without partial output.

`bra/test/qasm2qcx_negative_control_numerical.py` checks negative and mixed
control groups, modifier order, inversion, already-controlled gates, phase
corrections and `gphase` against independent complex matrices. It covers every
supported gate, basis states and complex superpositions, runtime parameters,
broadcasting, loops, and executed/skipped parameter checks. The references
retain global phase. Unit tests additionally check compact maximum-size counts,
invalid modifiers/counts, arity, duplicate operands, loop-iterator shadowing,
temporary lifetime, and CLI failures without partial output.

`bra/test/qasm2qcx_power_numerical.py` checks integer powers of every supported
gate against independent complex matrix references on basis states and complex
superpositions, retaining global and relative phases. It covers positive,
negative, and zero powers, periodic reductions, scaled constant angles, runtime
parameters, mixed controls, nested powers, broadcasting, loop re-entry, skipped
parameter checks, and finite angles whose scaled product would round or
overflow. Unit tests additionally check exponent
validation, empty loops, zero/identity validation, parameter capture and storage
lifetime, expansion budgets, large nested powers, the documented example,
and CLI failures without partial output.

`bra/test/qasm2qcx_runtime_power_numerical.py` checks runtime integer gate powers
against independent complex matrix references, retaining global phase. It
covers every supported gate, positive/negative/zero counts, signed and unsigned
boundaries, nested sign parity and periodic reductions, mixed controls,
broadcasting, indexed expressions, integer loop iterators, source-value
preservation, and executed/skipped parameter checks. Unit tests additionally
check private storage lifetime, zero guards, separate nested counters,
constant-factor limits, and conversion of the documented runtime example.

`bra/test/qasm2qcx_float_power_numerical.py` uses independent analytic spectral
projectors and exact rational eigenphases. It checks principal-branch endpoints,
both sides of the cut, nested powers and inversion order, and preservation of
global phase. With `--bra`, it compares converted circuits with the spectral
matrices for every supported fixed family, intrinsic and added controls,
mixed polarities, basis and complex input states, extreme exponents,
broadcasting, classical control flow, unrolled-loop qubit indices, and the
documented floating-power example. Without `--bra`, only its 14 reference
tests run. Unit tests also cover rejected types/families, zero validation,
temporary cleanup, output budgets, and CLI failures without partial output.

`bra/test/qasm2qcx_phase_power_numerical.py` checks parameterized phase powers
against independent scalar spectral references, including the documented
example, large angles, cut boundaries, nested modifiers, mixed controls,
broadcasting, arrays, classical control flow, fusion, and runtime failures.

`bra/test/qasm2qcx_rotation_power_numerical.py` uses independent analytic Pauli
projectors to check both eigenphases, exact and nearby branch cuts, global
phase, nested modifiers, and finite phases whose difference overflows. With
`--bra`, it also audits ordinary/inverse rotation conventions, paired-phase
preparation, and converted powers of all six supported rotation families.
Coverage includes matrix columns and complex input states, intrinsic and mixed
added controls, noncontiguous wires, large angles and exponents, arrays,
broadcasting, loops, skipped/reached runtime failures, fusion, and the documented
rotation example. Without `--bra`, only the 18 reference tests run. Unit tests
additionally check temporary lifetime, once-only capture, type/operand validation
in empty loops, output budgets, and unchanged integer-only lowering.

`bra/test/qasm2qcx_u_power_numerical.py` checks the shared U-family spectrum
against independent direct matrices, known Hadamard roots, and a separate
quadratic eigensolver for well-separated cases. Its 19 standard-library-only
reference tests cover phase conventions, scalar/polar and nearly degenerate
cases, principal branches, nested modifiers, and large finite parameters.
With `--bra`, it audits existing ordinary/inverse gates, runtime spectral
preparation, and public floating-power conversion. Coverage includes full
matrix columns and complex states, mixed controls, noncontiguous wires,
arrays, broadcasting, source preservation, loop re-entry, reached/skipped
failures, large exponents, fusion, and the documented example. Unit tests also
check temporary cleanup, once-only evaluation, empty-loop validation, output
budgets, CLI failures without partial output, and unchanged integer-only paths.

`bra/test/qasm2qcx_cu_power_numerical.py` adds 17 standard-library-only reference
tests for the full active spectrum of `cu`, including gamma's effect on both
principal branches and controlled scalar phases. Independent direct matrices
and a quadratic eigensolver cross-check the spectral reference. With `--bra`,
it checks existing ordinary/integer conversion, intrinsic-controlled projector
instructions, four-parameter runtime preparation, and complete floating-power
circuits. Coverage includes inactive intrinsic and mixed added controls,
full matrix columns and complex states, noncontiguous wires, large parameters,
branch neighbors, arrays, broadcasts, native integer parameters, loop re-entry,
reached/skipped failures, fusion, and the documented example. Unit tests cover
once-only captures, temporary cleanup, invalid calls in empty loops, output
budgets, and CLI errors without partial output.

`bra/test/qasm2qcx_cu_phase_numerical.py` checks the ordinary/integer `cu`
phase-factor fix against independent direct matrices and integer matrix
products, without using floating-power spectral preparation. Four reference
tests need only Python's standard library. With `--bra`, the audit compares
absolute complex amplitudes for lost-small-angle and overflowing-sum cases,
both parameter signs, maximum finite angles, literal/runtime parameters,
inverse and signed/unsigned integer powers, mixed controls, indexed arrays,
noncontiguous broadcasting, loop re-entry, skipped checks, and fusion.
Native assertions additionally verify source-variable preservation, including
after queued fused gates are flushed. The large-gamma example above is tested
in normal and fused execution. Unit tests also check once-only argument visits,
temporary ownership, separated operands, and output budgets.

`bra/test/negative_controls_numerical.py` tests the native QCX syntax and gate
families directly, both normally and inside fusion blocks, including adjoints,
mixed polarities, aliases, comments and invalid operands. The focused C++ tests
`negative_controls.cpp` and `negative_controls_fused.cpp` check representation,
fusion classification, remapped control values, phase-batching exclusions and
threaded fused execution in both indexing modes. The MPI driver
`bra/test/negative_controls_mpi_numerical.py` checks native and converted
circuits with initial wire permutations, simple/unit policies, paged/nonpaged
states, and normal/fused execution against independent references.

`bra/test/integer_bitwise_numerical.py` checks bra's three integer bitwise
assignment operators against Python results, including negative values and
native endpoints, indexed operands, explicit casts, skipped instructions,
invalid types, and malformed syntax. `bra/test/integer_bitwise_state.cpp`, linked
with non-MPI bra objects excluding `bra.o`, checks instruction rendering,
diagnostics, failed-write preservation, and unchanged pending jump state in
release builds.

`bra/test/qasm2qcx_bitwise_numerical.py` compares constant-folded and runtime
integer results, checks scalar bit truth tables and complement masks, and
exercises both expressions and compound assignments. Coverage includes
self-references, overlapping indexed operands, singleton storage, index capture,
native endpoints, casts, precedence, left-to-right eager evaluation,
short-circuiting, loop nesting, gate parameters, and runtime failures. Unit tests
additionally cover typing, temporary cleanup, unsupported selections, empty-loop
validation, and read-only iterators.

`bra/test/qasm2qcx_bit_register_bitwise_numerical.py` exhaustively checks
register expressions and compound assignments for all operand pairs at widths
1 through 4. It also covers leading zeroes, nested literal folding,
self-references, 33- and 65-bit registers, branches, range/set/register iteration,
`while`, loop transfers, declaration-scope widths, iterator shadowing, and
measurement-controlled execution. Unit tests additionally cover width/type
errors, unsupported selections and scalar contexts, empty-loop validation,
temporary reuse, and cleanup after failed lowering.

`bra/test/qasm2qcx_bit_register_shift_numerical.py` compares literal folding,
runtime expressions, ordinary assignments, and compound shifts with Python
bit patterns at widths 1, 2, 4, 33, and 65. It covers zero and boundary counts,
native integer maximum counts, negative-count failures, bit order, nested
results, self-referencing counts, single index capture, skipped errors,
loop transfers, live register iteration, and measurement-controlled execution.
Unit tests additionally check width/type rejection, guard-before-cast ordering,
buffer reuse, cleanup, empty-loop validation, and unsupported scalar contexts.

`bra/test/qasm2qcx_bit_register_equality_numerical.py` checks literal folding and
runtime equality/inequality for widths 1, 2, 4, 33, and 65, including exhaustive
pairs through four bits. It covers static selections (including repeated sources),
nested register expressions, Boolean/scalar-bit values and casts, branches
(including measurement-controlled execution), loops, complete operand evaluation, and logical
short-circuiting around runtime failures. Unit tests also cover width/type errors,
width rejection, empty-loop validation, iterator shadowing, and
temporary cleanup. The equality numerical suite also passes with UBSan enabled.

`bra/test/qasm2qcx_runtime_range_equality_numerical.py` checks runtime-range
equality and inequality against independent little-endian bit-vector
references. It covers exhaustive small patterns, every native width, widths
through 129 source bits, reversed and negative ranges, repeated/overlapping
source positions, mutable and constant sources, and nested bitwise, complement,
shift, and rotation expressions. It also checks scalar consumers, destination
addresses, loop reentry, measurement feedback, normal/fused gate parameters,
skipped domains, and complete operand evaluation before length checks or a
mismatching-bit exit. Error markers verify that failures suppress destination
writes. Unit tests cover declaration visibility, iterator shadowing, output
limits, and temporary cleanup after nested failures. The audit passes on native
and UBSan builds and executes the runtime-range equality example above.

`bra/test/qasm2qcx_angle_range_equality_numerical.py` checks equality and
inequality of runtime angle-bit ranges against independent bit vectors. It
covers sized angle widths, constant and array sources, bit-register operands,
nested expressions, repeated positions, angle iterators, skipped errors, and
failure ordering before result writes. It runs with native and UBSan builds.

`bra/test/qasm2qcx_register_ordering_numerical.py` checks fixed and runtime
register ordering against unsigned reference values, including bit-register
and angle-bit ranges, nested expressions, selections, wide values, skipped
paths, and failure ordering. Run with `--bra /path/to/bra`, including a UBSan
build.

`bra/test/qasm2qcx_runtime_range_expression_assignment_numerical.py` checks whole-register
assignments and initializers against independent little-endian bit vectors.
It covers exhaustive small patterns, every native width, wide sources through
129 bits, repeated and overlapping source positions, and nested bitwise,
complement, shift, and rotation operations. It checks self-assignment, changing
ranges and source values in loops, scalar consumers nested in the RHS, constant
sources, skipped domains, measurement feedback, and normal/fused gate consumers.
Error markers verify subtree completion before final size checks and suppression
of destination writes on failures, including later loop iterations and storage
capacity errors. Unit tests check declaration scope, visibility, bounded copy
code, output limits, and cleanup preserving caller-owned state and storage.
The audit passes on native and UBSan builds and executes the assignment example above.

`bra/test/qasm2qcx_static_selection_expression_assignment_numerical.py` checks
runtime-range RHS expressions assigned into static ranges and index sets against
independent bit-vector references. It compares complete destination registers,
including untouched positions, with exhaustive small patterns, every native
width, wide sources through 129 bits, negative and reordered destinations,
singletons, and repeated source positions. It also covers self-overlap, nested
operations, changing source values and ranges in loops, skipped domains,
measurement feedback, and normal/fused gate consumers. Error markers check RHS
completion, final length checks, and suppression of selected writes on failures,
including later iterations and capacity errors. Unit tests cover destination
scope, visibility, once-only captures, and cleanup after a final output-budget
failure. The audit passes on native and UBSan builds and executes the static
selection expression assignment example above.

`bra/test/qasm2qcx_runtime_set_expression_assignment_numerical.py` checks
runtime-range expression assignments into runtime index sets against independent
bit-vector references. It compares complete destination registers with exhaustive
small patterns, every native width, wide sources through 129 bits, mixed
INT/UINT and negative indices, array-supplied addresses, constant and mutable
sources, nested operations, and repeated source positions. It checks overlap,
indices reading the original destination, changing indices and source ranges
in loops, nested scalar counts, skipped domains, measurement feedback, and
normal/fused gate consumers. Failure markers verify destination validation
before RHS evaluation and no selected writes after failures, including later
loop iterations. Unit tests cover visibility, shadowing, once-only capture,
output limits, and cleanup preserving caller-owned state and storage. The audit
passes on native and UBSan builds and executes the runtime index-set expression
assignment example above.

`bra/test/qasm2qcx_runtime_range_destination_expression_numerical.py` checks
runtime-range expression assignments into runtime-range destinations against
independent bit-vector references. It compares complete destination registers
with exhaustive small source patterns, every native width, and wide sources
through 129 bits. It covers mixed INT/UINT bounds, negative and descending ranges,
nonunit strides, unselected stop bounds, native endpoint steps, singletons,
repeated source bits, overlap, bounds reading the original destination, changing
ranges and source values in loops, nested scalar consumers, skipped domains,
measurement feedback, and normal/fused gate consumers. Failure markers verify
destination checks before RHS evaluation and suppression of writes on length,
duplicate, domain, and capacity failures, including later loop iterations.
Unit tests cover capture order, held destination storage, bounded write loops,
output limits before capture/after validation/before writes, and cleanup
preserving caller-owned state and storage. The audit passes on native and UBSan
builds and executes the runtime-range destination expression example above.

`bra/test/qasm2qcx_runtime_expression_bitwise_compound_numerical.py` checks all
three bitwise compound operators across whole registers, static ranges/sets,
runtime index sets, and runtime ranges against independent bit-vector references.
It compares complete destination registers with exhaustive small patterns, every
native width, and sources through 129 bits, including constant sources, repeated
source bits, deep expression trees, and native endpoint distances. It covers
original LHS values, overlap, indices reading the destination, changing counts
and sources across loop iterations, transfers, singleton storage, mixed INT/UINT
indices and bounds, skipped domains, scalar consumers, measurement feedback,
and normal/fused gate consumers. Failure markers verify destination validation
before original reads, complete RHS evaluation before final length checks, and
suppression of combinations/writes on errors, including later iterations and
capacity failures. Unit tests cover once-only captures, held snapshots, caller
state/storage preservation, and exact output budgets versus final-write failures.
The audit passes on native and UBSan builds and executes the compound-assignment
example above.

`bra/test/qasm2qcx_static_bit_selection_numerical.py` checks all four-bit source
patterns across ascending/descending ranges, negative indices, reordered sets,
singleton views, and repeated sources. A Python snapshot model checks ordinary
and compound assignments with overlapping destinations, verifying untouched
bits and selected-width shifts. It also covers nested expressions, runtime
counts, self-references, loops, skipped checks, and negative-count failures.
Unit tests cover immutable metadata, constant bounds, width errors, repeated
destination rejection, iterator shadowing, source declaration ordering, and
temporary reuse/cleanup.

`bra/test/integer_shift_numerical.py` checks native signed and unsigned shifts
at every valid count, negative values and endpoints, invalid counts, overflow,
indexed operands, and skipped errors. `bra/test/integer_shift_state.cpp`, linked
with non-MPI bra objects excluding `bra.o`, verifies rendering, exception types
and diagnostics, destination preservation, and unchanged pending jump state in
release builds. Both the native tests and converter numerical tests have also
been run with undefined-behavior sanitization of the shift implementation.

`bra/test/qasm2qcx_shift_numerical.py` compares constant folding, expressions,
and compound assignments against Python results for signed native integers and
unsigned widths 1, 4, 8, and native width. It checks count-type independence,
intermediate masks, self-references, captured array indices, loop bounds and
sets, skipped errors, invalid counts, and signed overflow. Unit tests additionally
cover type rejection, non-native signed operands, empty-loop validation, and
temporary cleanup.

`bra/test/qasm2qcx_negation_numerical.py` compares constant-folded and runtime
native INT negation at zero, native endpoints, and ordinary positive/negative
values. It checks minimum literal spellings, overflowing constants and runtime
values, nested/indexed operands and assignments, cast order, UINT modular
negation, floating-point/complex negation, short-circuiting, branches and loops,
declaration-scope widths, gate parameters, and measurement-controlled execution.
Unit tests additionally cover unsupported widths/types, empty-loop metadata,
single index evaluation, CLI diagnostics without partial output, temporary
hoisting, and cleanup after failed lowering while preserving caller storage.
The negation numerical suite also passes with UBSan enabled.

`bra/test/qasm2qcx_integer_power_numerical.py` compares constant folding, runtime
powers, and compound assignments with Python reference results. It checks signed
bases and native endpoints, zero/one exponents, unsigned widths and wrapping,
mixed promotions, nested powers, single destination-index capture, self-references,
casts, comparisons, short-circuiting, branches, range/set/array/register/while loops,
declaration-scope widths, gate parameters, and measurement-controlled execution.
Negative exponents, signed-result overflow, and checked assignment-conversion
failures are tested at runtime. Unit tests additionally cover precedence,
invalid operands and widths, empty loops, read-only iterators, temporary cleanup,
and failed RHS lowering. The power numerical suite also passes with UBSan enabled.

`bra/test/qasm2qcx_integer_remainder_numerical.py` verifies runtime remainder
for signed operands and native endpoints (including the minimum signed integer
modulo `-1`), constant/runtime agreement, nested expressions, operand preservation,
short-circuiting, compound assignments, indexed self-references, unsigned widths,
and zero-divisor diagnostics. Unit tests additionally check temporary reuse and
cleanup. `bra/test/integer_remainder_state.cpp` and
`bra/test/integer_remainder_numerical.py` test native `INT`/`UINT` remainder,
checked casts and nested indexing, aliasing, unchanged destinations after errors,
and skipped execution. The remainder tests also pass with UBSan enabled.

`bra/test/qasm2qcx_uint_numerical.py` compares constant folding and runtime
unsigned arithmetic, remainder, bitwise expressions, complement, and compound
assignments at widths 1, 4, 8, and native UINT. It covers intermediate masks,
native endpoints, mixed signed/unsigned promotions, casts, comparisons,
checked conversions, and skipped runtime errors. Integration cases verify
UINT range bounds and steps, set captures, `while`, `break`/`continue`, array
and bit indices, measurement destinations, quantum parameters, singleton
storage, and bounds failures. Unit tests also cover invalid widths, unsupported
signed-width combinations, constant loop capture limits, and temporary reuse.

`bra/test/qasm2qcx_uint_bit_cast_numerical.py` checks exact UINT/register
patterns at every native-supported width, exhaustive small widths, native high
bits, deterministic random patterns, constant/runtime round trips, source
preservation, static selection order, repeated positions, and wide source
registers. Integration cases cover narrow intermediate wraparound, width
promotion, numeric conversions, array initializers and elements, shift counts,
loop bounds and transfers, typed iterator shadowing, live bit iteration, skipped
operand errors, measurement feedback, and gate parameters with and without
fusion. Later-iteration failures preserve earlier successful assignments. The
example above is executed. Unit tests cover side-effect-free metadata, width
and type rejection, nested constant-register loop bounds, once-only operand
capture, unsigned packing, private extraction before writes, temporary ownership,
output limits, declaration hoisting, and cleanup preserving caller context and
storage on failure. Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_bit_set_uint_cast_numerical.py` checks UINT packing
from runtime index sets at every native-supported width, singleton and repeated
positions, negative aliases, INT/UINT indices, and wide mutable/constant sources
across constant lookup word boundaries. It checks nested casts and indices,
register expressions, numeric and array consumers, captured loop bounds,
iterator scopes, skipped errors, measurement feedback, and quantum parameters
with and without fusion. Reached index failures precede selected reads and
destination writes, including failures after an earlier successful assignment.
The runtime-set example above is executed. Unit tests also cover width and
type validation, skipped-body visibility, snapshot and result ownership,
allocation/output failures, and restoration of caller state. Run with
`--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_bit_range_uint_cast_numerical.py` checks direct
runtime range-to-UINT casts at every native-supported width, ascending and
descending ranges, nonunit/runtime steps, mixed INT/UINT operands, negative
indices, repeated aliases, singleton sources, and native high result bits.
Wide constant sources exercise full and partial lookup words. Integration
checks cover nested scalar casts in bounds and steps, arrays, captured loop
bounds, iterator scopes, fresh values across iterations, short-circuiting,
measurement feedback, and normal/fused gate parameters. Invalid positions,
zero steps, empty selections, and count mismatches fail before selected reads
and writes, including later-iteration failures after successful assignments.
The runtime-range example above is executed. Unit tests also verify metadata,
count-check ordering, bounded capture capacity, unsigned packing, snapshot
ownership, output/allocation failures, and cleanup preserving caller state.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_unary_shift_uint_cast_numerical.py` checks
runtime range complement/shift chains inside UINT casts at every native width.
It covers complement placement, truncation at each intermediate step, literal
and runtime counts, native endpoints, oversized counts, negative aliases,
strides, wide constant sources, unsized targets, and overlapping assignments.
Integration cases cover scalar/register/array consumers, nested casts in bounds
and counts, loop capture and iterator scopes, fresh runtime operands, skipped
failures, measurement feedback, broadcast gates, and normal/fused execution.
Reached failures verify range-before-count checking and preservation of prior
successful assignments. The chain example above is executed. Unit tests also
verify operation order and occurrence identity, once-only captures/counts,
snapshot ownership and binding restoration, output limits, and cleanup after
allocation or consumer failures. Run with `--bra /path/to/bra`, including a
UBSan build.

`bra/test/qasm2qcx_runtime_range_bitwise_uint_cast_numerical.py` checks all
three binary bitwise operators inside UINT casts at every native width, with
runtime ranges on either or both sides and supported fixed-width counterparts.
It covers nested complement/shifts, negative aliases, repeated accesses, native
endpoints, wide constants, unsized targets, and overlapping writes. Integration
cases cover arrays, scalar and register consumers, nested casts in bounds and
counts, captured loop bounds and iterator scopes, fresh runtime operands,
skipped failures, measurement feedback, broadcast gates, and fused execution.
Reached failures verify left-to-right reads/checks and preservation of earlier
successful assignments. The binary-tree example above is executed. Unit tests
also check tree metadata and width propagation, bounded static expansion,
independent occurrence identities, one-time capture, snapshot ownership,
output limits, and cleanup preserving caller state and bindings after failures.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_rotation_uint_cast_numerical.py` checks both
rotation directions inside UINT casts at every native width. It covers signed
distance endpoints, negative/zero/oversized distances, Boolean and checked UINT
promotions, constant and mutable sources, strides, negative indices, wide
sources, nested bitwise/shift/rotation trees, unsized targets, and overlapping
assignments. Integration cases cover arrays, nested distance casts, fresh loop
operands, transfers, skipped errors, measurement feedback, and quantum parameters
with and without fusion. Reached failures check operand order and absence of
destination writes. The rotation example above is executed. Unit tests verify
width propagation, non-evaluating validation, once-only capture and distance
evaluation, temporary ownership, binding restoration, and failure cleanup.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_integer_array_numerical.py` verifies declarations and runtime
initializers, static, runtime, and negative element indices, compound assignments,
live reads versus set snapshots, dimension-constant shadowing, repeated entry,
mixed nesting, measurement-controlled transfers, gate parameters, skipped and
executed zero-divisor operations, temporary reuse, and QCX integer endpoints.
Mutation and position-based `break`/`continue` results are checked against
explicit Python reference loops. Unit tests additionally cover invalid sizes,
indices and initializers, unsupported array operations, singleton arrays,
storage cleanup, and expansion budgets. Runtime-indexing coverage includes
nested reads and writes, index capture, signed compound assignments against a
Python reference model, bounds failures, skipped assertions, mixed control flow,
casts, indexed initializers, and temporary-storage lifetime. Unit tests interpret
the emitted normalization guard using checked native integer arithmetic.

`bra/test/qasm2qcx_uint_array_numerical.py` checks every supported element width,
initialization, width-preserving reads, wrapping intermediate results, and
compound assignments against both Python results and equivalent scalar UINT
operations. It covers mixed signed/unsigned promotions, negative/static/runtime
indices, nested self-references, captured destinations, shifts, exponentiation,
skipped checks, bounds failures, zero divisors, and checked conversions.

`bra/test/qasm2qcx_runtime_array_slice_loop_numerical.py` checks runtime slice
loops against an independent index model, including 600 range shapes, all
supported array element types, widening, live source mutations, captured bounds,
empty and skipped loops, extreme steps, and failures before the body. It executes
the documented example. Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_array_slice_source_numerical.py` checks runtime
source copies against an independent snapshot model, including 300 range pairs,
all supported element types, overlap, repeated source positions, empty and
skipped paths, extreme steps, capacity failures, and pre-write error ordering.
It executes the documented example. Run with `--bra /path/to/bra`, including a
UBSan build.

`bra/test/qasm2qcx_runtime_array_slice_destination_numerical.py` checks runtime
destinations for all supported element types, overlap, negative indices, empty
and skipped copies, and range validation failures. It executes the documented
example. Run with `--bra /path/to/bra`.

`bra/test/qasm2qcx_static_array_slice_assignment_numerical.py` checks slice
copies against an independent snapshot model, including overlap, negative and
repeated source indices, reverse strides, empty and skipped copies, all supported
array element types, and UINT and angle widths. It also executes the documented
example. Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_static_array_slice_loop_numerical.py` compares static slice
iteration with an independent index model. It covers forward, reverse, strided,
negative, repeated, singleton and empty selections, all supported array element
types, iterator widening, mutations, nested loops, transfers, and gate
parameters. It also executes the documented example. Run with
`--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_uint_array_loop_numerical.py` checks live reads, copied iterator
values, mutation, singleton arrays, loop transfers against Python reference
loops, repeated entry, mixed nesting, width shadowing, every supported widening
pair and compatible native-width forms, casts, comparisons, gate parameters, skipped accesses, and executed
conversion/count/bounds errors. Unit tests additionally verify unsupported
widths/types, scoped read-only bindings, empty-loop validation, and cleanup
without releasing caller-owned storage.

`bra/test/qasm2qcx_float_array_numerical.py` verifies native REAL declarations,
runtime initializers and scalar numeric promotions, static/runtime/negative
indices, singleton arrays, indexed assignments and compound assignments,
captured destination indices, nested integer-array indices, and bounds failures.
Integration cases cover arithmetic, negation, casts, mixed numeric comparisons,
short-circuiting, skipped accesses, integer loops and `while`, iterator shadowing,
gate parameters, and executed index division errors. Unit tests additionally
cover invalid dimensions, initializers, indices, unsupported operations and
iterator types, empty-loop validation, and temporary cleanup.

`bra/test/qasm2qcx_float_array_loop_numerical.py` checks element order, live reads,
current-element mutation and copied-value preservation, singleton arrays,
position-based transfers against Python reference loops, repeated entry after
`break`, nested typed shadowing, and mixed loop nesting. It also covers casts,
comparisons, gate parameters, source-angle preservation, skipped runtime
checks, and executed access/division errors. Unit tests additionally verify
matching iterator types, read-only bindings, empty enclosing loops, temporary
hoisting, cleanup after failed body lowering, and CLI rejection without partial
output.

`bra/test/qasm2qcx_bool_array_numerical.py` checks Boolean initialization,
static/runtime/negative indexing, singleton arrays, self-referential assignments,
logical truth tables, short-circuiting, casts, bit interoperability, branches,
position-based loops, measurement-controlled conditions, and gate parameters.
It also checks executed bounds and index-division failures. Unit tests cover
invalid dimensions, initializers, operations and indices, empty-loop validation,
captured destination indices, and temporary cleanup after failed RHS lowering.

`bra/test/qasm2qcx_bool_array_loop_numerical.py` compares live reads, mutation,
copied iterator values, singleton arrays, visit order, and `break`/`continue`
against Python reference loops. It also covers repeated entry, mixed-type
shadowing, nested loops, casts, bit interoperability, gate parameters, skipped
checks, and executed access/division failures. Unit tests additionally verify
matching iterator types, read-only bindings, empty enclosing loops, resolved
labels, temporary hoisting, and cleanup preserving caller-owned storage.

`bra/test/qasm2qcx_complex_array_numerical.py` checks complex storage and precision
policy, literal/runtime initializers and scalar promotions, static/runtime/negative
indices, singleton arrays, arithmetic against Python complex reference values,
compound assignments, self-references, nested indices, casts, loop integration,
short-circuiting, gate parameters, and executed bounds/index-division failures.
Unit tests additionally cover unsupported declarations, operations, implicit
narrowing, captured destination indices, empty-loop validation, and cleanup.

`bra/test/qasm2qcx_complex_array_loop_numerical.py` compares live reads, visit order,
mutation, copied iterator values, and loop transfers against Python reference
loops. It also checks repeated entry after `break`, scoped mixed-type shadowing,
source-array shadowing, mixed loop nesting, casts, gate parameters, skipped
checks, and executed failures. Unit tests verify matching complex base types,
the native precision policy, singleton indexing and endpoint guards, read-only
bindings, empty enclosing loops, hoisted temporary declarations, resolved labels,
and cleanup preserving caller-owned storage.

`bra/test/qasm2qcx_bit_register_numerical.py` verifies singleton and larger
registers, bit-string element order, live mutation, casts and conditions,
scalar assignments, mixed loop nesting and typed shadowing, repeated entry,
measurement-controlled transfers and source updates, gate parameters, and
skipped arithmetic and bounds checks. Mutation and position-based transfers
are checked against 288 explicit Python reference cases. Unit tests additionally
cover iterator typing, invalid sources and iterator types, storage cleanup,
reserved names, fixed-size code generation, and expansion budgets.

`bra/test/qasm2qcx_constant_bit_register_loop_numerical.py` checks literal-bit
visit order and counts through 129 bits, nested and repeated entry, live mutable
inner registers, nearest loop transfers, typed shadowing, declaration-scope
widths, measurement-controlled execution, skipped and reached errors, source
preservation, gate parameters with and without fusion, and the constant-register
loop example above. Unit tests additionally check source visibility and iterator
types, rejected writes on skipped paths, expansion budgets, unique resolved
labels, and failure cleanup preserving caller-owned storage.

`bra/test/qasm2qcx_static_bit_selection_loop_numerical.py` checks static range
and set iteration on mutable and constant registers, including singleton and
129-bit sources. Independent bit-string references check order, counts,
negative indices, repetitions, and source preservation. Cases also cover live
mutations and measurements, copied iterator values, shadowing, mixed nesting,
reentry, nearest-loop transfers, skipped/reached arithmetic failures, and the documented example.
Full quantum states check phase gates, inverse and power modifiers, negative
controls, global phase, and fusion. Unit tests additionally check unsupported
contexts, source visibility, validation in empty loops, body-once mutable
lowering, constant expansion limits, declaration hoisting, reserved names, and
failure cleanup. Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_singleton_bit_loop_numerical.py` checks mutable and
constant singleton-loop sources through 129 bits, including every bit around
native UINT word boundaries. Independent bit-string references cover INT/UINT
indices, negative normalization, captured values, unchanged constants, shadowing,
nested table reuse, mixed loops, reentry, transfers, measurements, and reached/skipped
index and arithmetic failures before body execution. Complete quantum states
check phase gates, inverse and floating-point power modifiers, negative controls,
global phase, fusion, and the documented example. Unit tests additionally check
metadata without values, unsupported contexts, source-size and output limits,
single-copy lowering, reserved storage, declaration hoisting, and cleanup after
partial capture, lookup, or body failures. Run with `--bra /path/to/bra`, including
a UBSan build.

`bra/test/qasm2qcx_runtime_bit_set_loop_numerical.py` checks mutable and constant
multi-entry loop sources through 129 bits, with independent bit-string references
for order, repetitions, mixed INT/UINT entries, negative aliases, and every bit
around native UINT word boundaries. It covers captured positions, live source
writes and measurements, copied values, shadowing, nested table reuse, reentry,
transfers, skipped/reached bounds and arithmetic failures, and the example above.
Full quantum states check phase gates, global phase, and fusion. Unit tests also
check metadata without evaluation, validation in empty loops, unsupported
contexts, body-once lowering, table preparation before the loop, output limits,
declaration hoisting, and failure cleanup. Run with `--bra /path/to/bra`, including
a UBSan build.

`bra/test/qasm2qcx_runtime_bit_range_loop_numerical.py` checks mutable and constant
range sources through 129 bits against independent integer-progression and
bit-string references. It covers signed/unsigned operands, native endpoints,
full-width UINT steps, ascending/descending ranges, negative aliases, unselected
stop bounds, captured positions, live writes and measurements, scoped shadowing,
mixed nesting, reentry, transfers, and skipped/reached arithmetic and bounds
failures. Constant cases cross native word boundaries and include partial words.
It also verifies table-capacity limits before the body, phase gates, global
phase, fusion, and the documented example. Unit tests cover metadata, validation
in empty loops, single-copy bodies, table initialization and reservation, output
limits, declaration hoisting, and cleanup preserving caller storage after failures.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_bit_set_expression_numerical.py` checks runtime set
expression snapshots on mutable and constant registers through 129 bits. Python
bit-string references verify entry order, repetitions, negative aliases, mixed
INT/UINT indices, every source bit around native word boundaries, and partial
final words. Cases cover overlapping assignments, nested bitwise/shift
expressions, short-circuit comparisons, index expressions, iterator dependencies,
rotations, population counts, exact angle casts, skipped/reached failures, and
the executable example above. Full quantum states check phase gates and fusion.
Unit tests also check metadata without evaluation, scalar/register distinctions,
unsupported contexts, source visibility, skipped-body validation, index/check/read
ordering, one table per constant access, output limits, and temporary ownership
on success, capture failure, lookup failure, and consumer failure.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_bit_set_assignment_numerical.py` checks plain and
compound assignments through 129 bits against Python bit-list references. Cases
cover selection order, mixed INT/UINT entries, negative aliases, unchanged
unselected bits, overlapping swaps and self-copies, nested RHS expressions,
selection-width shifts, oversized UINT counts, loops, shadowing, and live reads.
Skipped/reached failures cover destination bounds, duplicates beyond native word
width, index arithmetic, negative shifts, and RHS accesses. Full quantum states
check updated gate parameters and fusion, including the executable example above.
Unit tests additionally check capture/check/snapshot/write ordering, once-only
implicit LHS reads, independent explicit RHS accesses, output budgets, and
restoration of expression state, temporary ownership, and snapshot bindings after
success and failure. Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_bit_range_copy_numerical.py` checks whole-register copies
from runtime ranges on mutable and constant sources through 129 bits. Independent
Python progression and bit-list references cover ordering, negative aliases,
descending/nonunit steps, native endpoints, singleton storage, repeated source
positions, wide constants, self-copies, and the executable example above.
Branches and loops exercise fresh captures, changing source values, iterator
shadowing, transfers, skipped invalid ranges, and measurement-driven copies and
quantum feedback. Instrumented failures verify that bad ranges and count or
arithmetic failures precede source reads and writes, including on later visits.
Full quantum states check copied phase parameters with and without fusion.
Unit tests cover metadata without evaluation, unsupported consumers, source
visibility, capture/check/snapshot/write ordering, buffer capacities, output
limits, declaration hoisting, label resolution, and scope/temporary cleanup on
success and failed lowering while preserving caller-owned storage.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_static_selection_range_copy_numerical.py` checks runtime range
copies into static range/set destinations against independent Python bit lists.
Cases cover destination ordering, negative indices, descending/nonunit ranges,
singleton storage, wide constants and partial final words, repeated source
positions, overlapping self-copies, and unchanged unselected bits. Branches and
loops exercise fresh captures, changing source values, iterator shadowing,
transfers, and skipped failures. Measurements drive source selection and the
copied bits control quantum feedback. Instrumented later-iteration failures
verify no new selected reads or destination writes while retaining the earlier
successful result. Full quantum states check copied phase parameters and fusion;
the static-selection example above is executed. Unit tests check metadata,
bounded expansion, skipped-body validation, capture/check/snapshot/write
ordering, declaration hoisting, output budgets before all selected writes,
label resolution, and scope/temporary cleanup preserving caller-owned storage
and snapshot bindings on success and failed lowering.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_set_range_copy_numerical.py` checks runtime range copies
into runtime index-set destinations against independent Python bit lists. Cases
cover mixed INT/UINT entries, negative indices, singleton storage, wide sources
and destinations, constant-word boundaries, repeated source positions,
self-referential destination indices, overlapping self-copies, and unchanged
unselected bits. Branches and loops exercise changing addresses, source ranges
and values, shadowing, transfers, skipped failures, and live bit iteration.
Measurements drive both destination indices and source selection, and copied
bits control quantum feedback. Instrumented failures verify destination-error
priority over source arithmetic and no new reads or writes after a later failure.
Full quantum states check phase parameters with and without fusion; the runtime
set copy example above is executed. Unit tests cover metadata without evaluation,
once-only captures, held destination addresses, comparison/output budgets,
declaration hoisting, labels, and cleanup preserving caller-owned storage and
snapshot bindings on success and failed lowering.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_to_range_copy_numerical.py` checks independent
runtime source and destination ranges against Python progression and bit-list
references. Cases cover mixed INT/UINT operands, descending/nonunit strides,
negative aliases, singleton storage, wide constants, overlap, repeated source
positions, preserved unselected bits, and the executable example above.
Loops exercise changing counts, directions, source values and both position
tables, live iteration, shadowing, transfers, and differently sized branch copies.
Measurement-driven ranges feed quantum feedback, and full states check phase
parameters with and without fusion. Instrumented failures verify destination
error priority, no partial writes, and preservation of earlier successful copies,
including later capacity failures. Small storage limits allow short selections
on wide registers and reject oversized reached selections before reads/writes.
Unit tests cover six once-only operand captures, destination-only duplicate
checks, matching counts before reads, three bounded buffers, constant-size loop
code, declaration hoisting, labels, output budgets, and scope/temporary cleanup
preserving caller-owned storage and snapshot bindings on success and failure.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_range_bitwise_rhs_numerical.py` checks whole/static
`&=`, `|=`, and `^=` destinations with runtime range RHS values through 129 bits
against independent Python bit-list references. Cases cover mutable/constant
sources, destination order, negative indices, descending/nonunit strides,
repeated source positions, overlap, singleton storage, wide constants, native
integer endpoints, and unchanged unselected bits. Branches and loops exercise
fresh snapshots, changing source values and directions, live iteration,
shadowing, transfers, skipped invalid bounds, and overlapping self-references.
Measurements drive source ranges and the results control quantum feedback;
full states check angle phase parameters with and without fusion. Instrumented
failures verify that invalid ranges cause no selected RHS reads or writes,
including later failures after a successful update. The example above is
executed. Unit tests cover metadata without evaluation, unsupported contexts,
original/RHS/result/write ordering, bounded buffers, constant-size loops,
output budgets, declaration hoisting, labels, and cleanup preserving caller
storage, expression state, and outer snapshot bindings on success and failure.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_selection_range_bitwise_numerical.py` checks runtime
index-set and range destinations with runtime range RHS values for `&=`, `|=`,
and `^=`. Independent Python bit-list references cover ordering, mixed INT/UINT
operands, negative aliases, nonunit/descending strides, repeated sources,
self-referential indices and bounds, overlap, singleton storage, wide constants,
and preserved unselected bits. Branches and loops exercise changing counts,
directions, positions and source values, live iteration, iterator shadowing,
transfers, skipped invalid selections, and reentry with differently sized
self-references. Measurements choose both destination positions and source
ranges, and result bits drive quantum feedback; full states check phase
parameters with and without fusion. Instrumented later failures verify no new
RHS reads or writes and preserve earlier successful updates, including storage
capacity failures. The runtime-selected example above is executed. Unit tests
cover metadata without evaluation, destination/check/original/RHS/result/write
ordering, once-only captures, four bounded range buffers, constant-size loop
code, declaration hoisting, unique labels, output limits, and cleanup preserving
caller storage, expression context, and outer snapshot bindings on success and
failure. Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_bit_range_assignment_numerical.py` checks plain range
assignments through 129 bits against Python progression and bit-list references.
Cases cover signed/unsigned bounds and steps, negative aliases, nonunit strides,
unselected stop values, native integer endpoints, singleton storage, overlapping
assignments, nested operands, and the executable example above. Loops exercise
fresh captures, changing direction, live register iteration, scoped shadowing,
transfers, and measurement-controlled feedback. Instrumented failures verify no
destination writes on failed bounds, counts, duplicate checks, or RHS evaluation,
including on later iterations. Full quantum states check phase parameters and
fusion. Unit tests cover metadata without evaluation, skipped-body validation,
capture/check/snapshot/write ordering, table capacity, output limits, declaration
hoisting, label resolution, and cleanup preserving caller-owned storage.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_bit_range_bitwise_numerical.py` checks `&=`, `|=`, and
`^=` through 129 bits against independent bit-list references. Cases cover
descending/nonunit ranges, negative indices, mixed INT/UINT operands, singleton
storage, full-width self-operations, overlapping RHS selections, nested bounds,
and register expressions. Loops exercise reentry, changing direction, live
iteration, transfers, scoped shadowing, and measurement-controlled feedback.
Instrumented failures verify that destination checks and RHS failures precede
all writes, including on later iterations. Full quantum states check phase gates
and fusion; the documented example is executed. Unit tests cover metadata without
evaluation, skipped-body validation, original LHS snapshots, independent RHS
captures, output limits, declaration hoisting, and restoration of snapshot
bindings, temporary ownership, and scopes on success and failed lowering.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_bit_range_shift_numerical.py` checks both shift
directions through 129 bits against independent bit-list references. Cases cover
runtime lengths, descending/nonunit ranges, signed/unsigned counts, zero and
oversized distances, native endpoints, singleton storage, self-referential counts,
and normalized aliases checked against independent sets. Capacity tests verify
short selections on wider registers and failures before distance evaluation.
Loops exercise changing lengths/directions, stale-buffer prevention, transfers,
shadowing, live register iteration, skipped/reached failures, and measurement
feedback. Instrumented failures prove no new destination writes on later failed
iterations. Full quantum states check phase gates and fusion, and the documented
example is executed. Unit tests cover metadata without evaluation, capture and
snapshot ordering, guarded UINT conversion, bounded buffers and generated code,
output limits, declaration hoisting, label resolution, and scope/temporary cleanup.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_bit_set_measurement_numerical.py` checks both measurement
syntaxes with static qubit sources and runtime index-set destinations through
129 bits. Independent bit-list references cover selection order, mixed INT/UINT
indices, negative aliases, singleton storage, repeated qubits, and unchanged
unselected bits. Cases exercise original-value destination captures, skipped
measurements without quantum collapse, loop reentry, nested transfers, iterator
shadowing, entangled-state correlations, repeated measurements, feedback, and
the executable example above. Instrumented failure cases verify that bad indices,
duplicates, and arithmetic failures stop before measurement, including on later
loop iterations. Unit tests cover metadata without evaluation, scalar/register
distinctions, skipped-body validation, capture/check/measurement/write ordering,
output budgets, and scope/temporary cleanup after success and failed conversion.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_bit_range_measurement_numerical.py` checks both syntaxes
with static qubit sources and runtime ranges through 129 destination bits.
Independent bit-list references cover ordering, negative indices, descending and
nonunit strides, native endpoints, singleton storage, repeated source qubits,
unchanged unselected bits, original-value capture, and the executable example.
Loops exercise reentry, differing source widths, shadowing, transfers, and live
register iteration. Entangled-state tests check correlations, repeatability,
and feedback without assuming a particular random outcome. Instrumented failures
verify that invalid ranges, count mismatches, duplicate destinations, and
arithmetic errors stop before any measurement or target write, including on later
iterations. Unit tests cover metadata without evaluation, bounded static source
expansion, skipped-body validation, immediate outcome copies, delayed writes,
buffer capacities, output limits, label resolution, declaration hoisting, and
scope/temporary cleanup after success and failed lowering.
Run with `--bra /path/to/bra`, including a UBSan build.

`bra/test/qasm2qcx_runtime_bit_numerical.py` verifies runtime reads, assignments,
both measurement-destination syntaxes, singleton storage, negative indices,
integer casts, nested bit/integer-array indexing, and short-circuit evaluation.
Overlapping reads and writes and loop-driven mutations with `break`/`continue`
are checked against 360 Python reference cases. Failure cases cover bounds,
native integer endpoints, nested accesses, and index arithmetic. Unit tests
additionally check bit semantics, capture and measurement ordering, temporary
reservation and reuse, invalid syntax and unreachable accesses, static-output
compatibility, register-size limits, and CLI errors without partial output.

`bra/test/qasm2qcx_for_loop_numerical.py` verifies unrolled quantum operations,
measurement and reset, runtime classical accumulation, conditions, nested loops,
outer-dependent bounds, shadowing, and skipped runtime division. It also checks
`break` and `continue` against a Python reference loop across ascending,
descending, strided, singleton, and empty ranges, plus measurement-controlled
transfers, nearest-loop targeting, and temporary reuse after skipped operations.
Set-loop coverage includes nonmonotonic and repeated values, constants and casts,
outer-dependent elements, mixed nesting, and Python reference checks for
transfers by iterator value or runtime iteration position.

`bra/test/qasm2qcx_while_loop_numerical.py` verifies runtime condition
reevaluation, zero and multiple iterations, logical short-circuiting,
measurement-controlled termination and transfers, mixed nested loops, iterator
shadowing, and temporary reuse. Transfer results are also checked against Python
reference loops. Each numerical program execution has a timeout.

`bra/test/qasm2qcx_runtime_for_numerical.py` verifies ascending, descending,
singleton, and empty runtime ranges with unit and nonunit steps against Python
reference values, captured bounds, arithmetic and casted bounds, repeated entry,
iterator shadowing, mixed nesting, nearest-loop transfers, measurement-controlled exits, gate
parameters, skipped arithmetic, temporary reuse, and QCX integer endpoints.
Stride coverage includes aligned and unaligned stops, position-based transfers,
extreme representable steps, repeated entry, and the last visited value.
Unit tests additionally interpret the emitted advance guard with checked signed
arithmetic, exhaustively covering a small integer model and native endpoints.

`bra/test/qasm2qcx_runtime_step_numerical.py` verifies runtime-valued steps with
constant and runtime bounds, captured values, arithmetic and casts, positive and
negative directions, zero-step errors, repeated entry with changing signs,
iterator shadowing, mixed nesting, nearest-loop transfers, measurement-controlled
breaks, skipped arithmetic and assertions, temporary reuse, and native integer
endpoints. Position-based transfers and extreme-stride endpoint combinations
are checked against Python reference loops. Unit tests also verify capture
storage reservation, cleanup after emission failure, and expansion budgets.

`bra/test/assert_numerical.py` checks all six assertion comparisons for integer
and real operands, indexed values, skipped checks, diagnostics, and malformed
instructions. `bra/test/assert_state.cpp`, compiled with the same macros as bra
and linked with non-MPI bra objects excluding `bra.o`, checks that successful
and failed assertions preserve variables and pending jump state, including in
release builds.

`bra/test/qasm2qcx_runtime_set_numerical.py` verifies source order, duplicates,
capture-before-body behavior, repeated entry, casts, shadowing, mixed nesting,
nearest-loop transfers, short-circuit expressions, skipped capture and body
arithmetic, storage reuse, measurement-controlled exits, gate parameters, and
QCX integer endpoints. Transfer results are checked against Python reference
loops. A deliberate runtime-error test verifies that a later zero-divisor
element is evaluated even when the first body contains `break`.

Run the converter and numerical tests from the repository root:

```console
python3 -m unittest bra/test/test_qasm2qcx.py
python3 bra/test/classical_function_angle_numerical.py
python3 bra/test/classical_function_angle_numerical.py --sanitize
ulimit -c 0
python3 bra/test/jumpif_numerical.py --bra bra/bin/bra
python3 bra/test/assert_numerical.py --bra bra/bin/bra
python3 bra/test/integer_division_numerical.py --bra bra/bin/bra
python3 bra/test/integer_remainder_numerical.py --bra bra/bin/bra
python3 bra/test/mod_numerical.py --bra bra/bin/bra
python3 bra/test/integer_bitwise_numerical.py --bra bra/bin/bra
python3 bra/test/integer_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_if_else_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_math_function_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_array_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_array_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_bit_cast_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_angle_cast_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_angle_range_cast_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_angle_range_uint_cast_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_angle_range_register_assignment_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_angle_range_expression_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_range_static_selection_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_range_runtime_set_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_range_runtime_range_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_range_register_compound_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_range_static_compound_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_range_runtime_set_compound_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_range_runtime_range_compound_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_angle_selection_uint_cast_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_static_angle_bit_selection_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_selection_compound_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_angle_bit_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_angle_bit_set_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_bit_selection_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_angle_bit_range_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_phase_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_global_phase_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_sqrt_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_log_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_trigonometric_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_trigonometric_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_inverse_trigonometric_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_rounding_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_mod_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_popcount_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_popcount_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_popcount_complement_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_popcount_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_popcount_complement_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_popcount_rotation_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_popcount_nested_rotation_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_popcount_nested_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_popcount_interleaved_complement_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_popcount_mixed_operations_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_popcount_bitwise_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_pair_popcount_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_popcount_bitwise_complement_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_popcount_bitwise_result_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_popcount_bitwise_result_rotation_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_popcount_operand_chain_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_popcount_tree_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_range_popcount_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_constant_bit_register_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_constant_bit_register_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_static_bit_selection_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_singleton_bit_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_set_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_range_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_set_expression_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_set_assignment_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_bitwise_rhs_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_selection_range_bitwise_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_range_assignment_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_range_bitwise_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_range_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_range_measurement_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_constant_scalar_bit_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_rotation_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_u_phase_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_inverse_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_control_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_negative_control_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_power_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_power_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_float_power_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_cu_power_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_cu_phase_numerical.py --bra bra/bin/bra
python3 bra/test/negative_controls_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_integer_remainder_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_integer_power_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_negation_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_for_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_while_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_for_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_step_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_set_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_integer_array_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_static_array_slice_assignment_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_array_slice_source_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_array_slice_destination_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_static_array_slice_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_array_slice_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_uint_array_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_uint_array_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_float_array_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_float_array_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bool_array_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bool_array_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_complex_array_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_complex_array_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bit_register_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bitwise_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bit_register_bitwise_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bit_register_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bit_register_equality_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_equality_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_register_ordering_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_angle_range_equality_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_expression_assignment_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_static_selection_expression_assignment_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_set_expression_assignment_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_destination_expression_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_expression_bitwise_compound_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_static_bit_selection_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_uint_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_uint_bit_cast_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_set_uint_cast_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_range_uint_cast_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_unary_shift_uint_cast_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_bitwise_uint_cast_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_range_rotation_uint_cast_numerical.py --bra bra/bin/bra
```

For an MPI-enabled `bra` executable, run the distributed control-value driver
separately (MPI processes must be allowed to communicate on the host):

```console
python3 bra/test/negative_controls_mpi_numerical.py --bra bra/bin/bra --ranks 2 --mode simple --page-qubits 0
python3 bra/test/negative_controls_mpi_numerical.py --bra bra/bin/bra --ranks 4 --mode unit --page-qubits 2 --cache-qubits 3 --threads 2
```

Repeat with ranks `2`/`4`, modes `simple`/`unit`, and page counts `0`/`2` to
exercise all distributed state variants. Each invocation checks eight native
and converted matrix-reference circuits, with identity/nontrivial initial wire
permutations and normal/fused execution. The driver uses a temporary shared QCX
file so every rank reads the same program. A cache-aware build and
`--cache-qubits 3` exercise fused blocks larger than the cache. The unit-policy
tests use one unit qubit and two processes per unit; ten total wires leave
sufficient local/non-page capacity for the five-wire operations tested.
