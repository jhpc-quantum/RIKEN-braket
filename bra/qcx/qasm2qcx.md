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
- integer remainder expressions and compound assignments using `%` and `%=`;
- scalar constants, variables, assignments, and numeric casts;
- scalar and register `bit` values, including bit-string initialization;
- scalar `bool` variables and constants, Boolean literals, and expression
  assignments;
- projective measurement of individual qubits and register selections;
- reset of individual qubits and register selections;
- OpenQASM barriers as ordering-only operations;
- `if`, `else if`, and `else` control flow with comparisons and logical
  conditions;
- compile-time unrolling of constant integer-range and integer-set `for` loops,
  including `break` and `continue`;
- runtime-bound integer-range `for` loops with constant unit steps, including
  nested loops, `break`, and `continue`;
- integer-set `for` loops with runtime-valued elements captured at loop entry;
- runtime `while` loops, including nested loops, `break`, and `continue`;
- scalar expressions used as gate parameters; and
- final-state amplitude output through a namespaced pragma.

QCX has no unsigned integer type, so OpenQASM `uint` values are represented by
QCX `INT`, just like OpenQASM `int`. Consequently, unsigned ranges and
wraparound behavior are not preserved. Declared integer and floating-point
widths are also accepted but are not enforced by QCX.
OpenQASM `bit` values are represented by QCX `INT` variables whose elements
are restricted to zero or one by the converter. A measurement is emitted as a
QCX `M` operation followed immediately by assignment from `:OUTCOME`.

### Integer remainder

The `%` operator supports scalar `int` and INT-backed `uint` operands, including
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

The remainder is computed as `a - (a / b) * b`, so a nonzero remainder has
the sign of the dividend. Constant evaluation uses integer calculations without
floating-point conversion. `uint` uses the existing INT-backed representation;
declared widths and unsigned wraparound are not enforced.

Runtime remainder is lowered to existing QCX integer division, multiplication,
and subtraction instructions. Temporaries preserve both operands until the
complete result is available. Computations remain in their original branches
and short-circuit operands; only temporary declarations may be moved earlier.

`%=` is supported for scalar `int` and INT-backed `uint` variables with integer
right-hand operands. It uses the same lowering as `%`, followed by assignment
of the completed result back to the target. Self-references such as `a %= a`
or `a %= (a % b) + 1` therefore read the original target throughout evaluation.

Evaluated constant `/` and `%` expressions with zero divisors produce
converter-specific errors, reported by the command-line tool without a Python
traceback. Skipped operands of constant Boolean expressions are still validated
without performing the arithmetic. Runtime division remains deferred to QCX;
`bra` checks integer zero divisors when the division instruction executes and
throws `bra::integer_zero_divisor_error`. Integer overflow remains unchecked.
Literal zero-divisor `/` and `%` operations in runtime
expressions are also deferred so short-circuited operands can skip them.
Non-integer remainder operands are
rejected unless explicitly cast to an integer type first. Indexed classical
integer assignment targets remain unsupported.

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

Whole-register casts, including `bool(register)` when `register` is a `bit[1]`
register, and `bit[n](...)` casts are not yet supported. Complex-to-Boolean casts,
Boolean arithmetic, and mixed Boolean/numeric comparisons remain unsupported;
explicitly cast Boolean values
to a numeric type before using them in these operations. Boolean arrays and
block-local declarations remain unsupported.

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

The converter supports `if`/`else` statements with comparisons, direct scalar
bit and Boolean conditions, and logical operators. The supported comparison
operators are `==`, `!=`, `>`, `<`, `>=`, and `<=`. OpenQASM `!=` is emitted as the QCX
not-equal operator `\=`.

Conditions can compare scalar `int`, `uint`, `float`, and `bit` expressions.
A statically indexed element of a bit register is also accepted. Compatible
integer and floating-point operands are promoted when necessary. Complex
values and complete bit registers cannot be compared.
Boolean and scalar bit operands can be compared with one another using `==`
or `!=`.

A scalar `bit`, a statically indexed element of a bit register, or a Boolean
literal, constant, or variable can also be used directly as a condition:
zero is false and one is true. Logical negation
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
Boolean gate parameters require an explicit numeric cast. Bitwise operators
such as `&`, `|`, `^`, and `~` are not yet supported.
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
Runtime variables are not constant bounds, even when initialized with a literal;
such ranges use the runtime lowering described below instead of unrolling.
Only `int` iteration variables are supported; `uint` constants may still appear
in bounds under the converter's existing INT-backed representation.
Declared integer widths retain the limitations described above.

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
Runtime-dependent indices remain unsupported.

### Constant integer sets

The converter also supports `for int name in {value, ...}`, with a
single-statement or braced body. Each element must evaluate to an integer during
conversion. Integer literals, previously declared constants, supported constant
expressions, explicit integer casts, and outer iteration values are accepted.
As with ranges, only `int` iterators are supported; integer widths and INT-backed
`uint` values retain their existing limitations.

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
supported. Iteration over arrays, bit registers, or aliases remains unsupported.
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
General local declarations and iteration over arrays remain unsupported.
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
body copies of runtime-valued set loops; repeated set elements each count as an
expanded iteration. A `break` does not reduce the generated iteration count.
Runtime-bound range loops do not expand their runtime iteration count.

## Runtime-bound for loops

The converter supports `for int name in [start:stop]` and
`for int name in [start:step:stop]` when either bound depends on a runtime
variable or an enclosing runtime iterator. Both bounds must be present and have
integer type. Scalar `int` and INT-backed `uint` variables, supported integer
arithmetic, and explicit integer casts are accepted. Floating-point, Boolean,
and scalar bit operands require an explicit integer cast; statically indexed
bit-register elements are also accepted through a cast. Whole bit registers
are not integer bounds.

The step must evaluate during conversion to `1` or `-1`; an omitted step is `1`.
Runtime steps and other constant strides are unsupported for runtime-bound
ranges. Fully constant ranges still use unrolling and retain support for any
nonzero integer step. A runtime variable remains a runtime dependency even if
its initializer is a literal.

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

The start and stop are evaluated once, in that order, on each entry to the loop,
before binding the new iterator. Mutating their source variables in the body
does not change the captured range. The stop is inclusive, and a range whose
direction does not match its step executes zero iterations. A nested loop can
use its outer iterator in either bound, including when the inner iterator
shadows the outer name. The outer binding is restored after the inner loop.

The body is generated once using existing QCX labels, `LET`, `JUMP`, and
`JUMPIF`. Private integer storage holds the iterator and captured stop for the
whole loop, independently of expression temporaries and nested loops. An entry
check skips empty ranges. After each iteration, an endpoint check exits before
incrementing or decrementing at the last value, avoiding iterator advancement
overflow at QCX's integer endpoints. This is not a general arithmetic overflow
check: declared widths and unsigned semantics retain their existing limitations.

Bodies support the same statements as constant loops. The iterator is a scoped,
read-only runtime integer usable in arithmetic, gate parameters, and conditions.
Assignments and measurements into active iterators remain rejected. Runtime
iterators cannot supply static indices or range steps. Using a runtime iterator
in a set element selects runtime-valued set lowering.
Dynamic indexing remains unsupported. Constant range and set loops,
runtime-valued set loops, and runtime `while` loops may be nested in either direction.

`break` exits the nearest loop. In a runtime-bound `for`, `continue` jumps to
the endpoint check and iterator advancement, rather than reevaluating the
bounds. For example:

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
Runtime computations in bound expressions, including integer division and
remainder, remain at loop entry; body computations remain on the paths that
reach them. A skipped branch or outer runtime loop therefore skips generated
nested-bound computations too. Integer zero divisors are checked by `bra` when
the division executes. Conversion-time constant validation still applies to
fully constant ranges and independent constant bounds of nested loops; those
checks are not suppressed by a runtime-skipped path.

Runtime iteration counts do not consume the constant-loop expansion budgets,
and there is no runtime iteration limit. Constant loops nested in a runtime
loop are expanded once per generated instance and still consume the budgets.
A runtime loop nested in an expanded constant loop contributes its generated
statements and instructions to the enclosing expansion budgets.

## Runtime-valued integer-set for loops

The converter supports `for int name in {value, ...}` when one or more elements
depend on a runtime variable or an enclosing runtime iterator. Elements must
have integer type: scalar `int`, INT-backed `uint`, supported integer arithmetic,
and supported explicit integer casts are accepted. Floating-point, Boolean,
and scalar bit values require integer casts; statically indexed bit-register
elements may also be cast. Whole-register casts and runtime complex-to-integer
casts remain unsupported. Existing constant numeric casts retain their behavior.

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
No new bra instruction or runtime array indexing is needed.
Fully constant sets continue to use the existing constant-iterator unrolling.

The iterator is scoped to the body and may shadow a source variable, constant,
or outer iterator. Element expressions use the outer binding, and that binding
is restored after the loop. The iterator is a read-only runtime integer usable
in arithmetic, conditions, and gate parameters, but not in static indices or
constant range steps. General block-local declarations remain unsupported.

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
conversion-time iterations. Integer widths and unsigned semantics retain the
converter's existing limitations. Array, bit-register, and alias iteration
remain unsupported; the parser still rejects an empty source set `{}`.

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

General block-local declarations, runtime-dependent indexing, and writes to
active `for` iterators remain
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

- delays, runtime range steps or nonunit steps in runtime-bound ranges,
  iteration over arrays, bit registers, or aliases,
  or `switch` statements;
- user-defined gates or gate modifiers;
- dynamically computed indices, ranges with omitted bounds, or
  multidimensional indexing;
- general classical arrays, whole-register casts, complex-to-Boolean casts,
  Boolean arithmetic, mixed Boolean/numeric comparisons, or block-local
  declarations;
- bitwise operations or classical functions; or
- arithmetic operators other than `+`, `-`, `*`, `/`, and integer `%`.

The characterization tests in `bra/test/test_qasm2qcx.py` define the working
baseline. `bra/test/qasm2qcx_if_else_numerical.py` additionally converts and
executes deterministic measurement-controlled programs with `bra`, verifies
logical truth tables for conditions and values, and checks Boolean conversions,
self-referencing assignments, short-circuit evaluation, and temporary reuse.
It also executes the complete Boolean example above. These tests cover the
converter's supported Boolean subset; the limitations above remain explicit
future work.

`bra/test/qasm2qcx_integer_remainder_numerical.py` verifies runtime remainder
for signed operands, nested expressions, operand preservation, short-circuiting,
compound assignments, self-references, and temporary reuse.

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
singleton, and empty runtime ranges against Python reference values, captured
bounds, arithmetic and casted bounds, repeated entry, iterator shadowing,
mixed nesting, nearest-loop transfers, measurement-controlled exits, gate
parameters, skipped arithmetic, temporary reuse, and QCX integer endpoints.

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
python3 bra/test/jumpif_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_if_else_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_integer_remainder_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_for_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_while_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_for_numerical.py --bra bra/bin/bra
ulimit -c 0
python3 bra/test/qasm2qcx_runtime_set_numerical.py --bra bra/bin/bra
```
