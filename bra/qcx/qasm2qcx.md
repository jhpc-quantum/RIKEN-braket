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
- literal indices, static index ranges, discrete index sets, and complete
  qubit registers as gate operands;
- equal-sized register-selection gate broadcasting;
- scalar `int`, `uint`, `float`, and `complex` arithmetic;
- native unsigned storage and modulo-width arithmetic for scalar `uint` and
  supported `uint[n]` widths;
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
- static bit-register range/set selections as bitwise and shift operands
  and overlap-safe assignment destinations;
- equal-width bit-register `==` and `!=` comparisons, including static
  selections, bit-string literals, and nested bitwise/shift expressions;
- scalar integer shift expressions (`<<`, `>>`) and compound assignments
  (`<<=`, `>>=`), with checked counts and signed left-shift overflow;
- scalar constants, variables, assignments, and numeric casts;
- fixed-size, one-dimensional `int` arrays with static and runtime element
  reads and writes;
- scalar and register `bit` values, including bit-string initialization;
- runtime-indexed bit-register element reads, assignments, and scalar
  measurement destinations;
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
- runtime iteration over one-dimensional `int` arrays;
- runtime iteration over bit registers with scalar `bit` iterators;
- runtime `while` loops, including nested loops, `break`, and `continue`;
- scalar expressions used as gate parameters; and
- final-state amplitude output through a namespaced pragma.

OpenQASM `int` uses QCX `INT`; scalar `uint` uses native QCX `UINT`.
Narrow `uint[n]` values use the same native storage with explicit masks, as
described below. Declared signed-integer and floating-point widths generally
remain unenforced; shifted signed operands and mixed signed/unsigned operations
have the explicit width restrictions documented below.
OpenQASM `bit` values are represented by QCX `INT` variables whose elements
are restricted to zero or one by the converter. A measurement is emitted as a
QCX `M` operation followed immediately by assignment from `:OUTCOME`.

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
bits of the unsigned operand's width. General signed-width emulation,
UINT arrays, and `for uint` iterators remain unsupported.

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
negative value. UINT arrays and UINT iterators remain separate future work.

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

The remainder is computed as `a - (a / b) * b`, so a nonzero signed remainder
has the sign of the dividend. Unsigned remainder is nonnegative after operand
promotion. Constant evaluation uses integer calculations without floating-point
conversion; unsigned widths follow the normalization rules above.

Runtime remainder is lowered to existing QCX integer division, multiplication,
and subtraction instructions. Temporaries preserve both operands until the
complete result is available. Computations remain in their original branches
and short-circuit operands; only temporary declarations may be moved earlier.

`%=` is supported for scalar `int` and `uint` variables with integer
right-hand operands. It uses the same lowering as `%`, followed by assignment
of the completed result back to the target. Self-references such as `a %= a`
or `a %= (a % b) + 1` therefore read the original target throughout evaluation.

Evaluated constant `/` and `%` expressions with zero divisors produce
converter-specific errors, reported by the command-line tool without a Python
traceback. Skipped operands of constant Boolean expressions are still validated
without performing the arithmetic. Runtime division remains deferred to QCX;
`bra` checks integer zero divisors when the division instruction executes and
throws `bra::integer_zero_divisor_error`. Signed integer overflow remains unchecked;
unsigned arithmetic wraps at its supported width.
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
RHS. No implicit padding, truncation, or scalar broadcasting is performed.
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

Register values are supported in register assignments and the equality
comparisons described below; they are not bare conditions. Integer/register casts,
indexing of computed results, or register expressions as loop bounds or
iteration collections remain unsupported. Static selection operations are described below.
Concatenation, aliases, `const bit[n]` declarations,
and functions such as `popcount` remain outside this subset. Existing direct
bit-register copies and indexed scalar-bit operations retain their behavior.

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
selection shifts are described below. This does not add rotations, aliases, concatenation, register
casts or ordered comparisons, `const bit[n]` declarations, or register-valued scalar
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
paths. Active iterators remain read-only. Dynamic selections, multidimensional
indexing, indexing computed register results, aliases, concatenation, register
casts, ordered comparisons, and iteration over selections remain separate work.

### Bit-register equality

`==` and `!=` accept two register-valued operands of equal width and produce a
scalar Boolean result. Operands may be whole `bit[n]` variables, bit-string
literals, supported static range/set selections, or nested register bitwise and
shift expressions. Selection order defines the corresponding bits to compare;
repeated source indices are allowed. A one-bit register or selection remains
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
Constant Boolean declarations accept comparisons of literal register expressions
with constant shift counts. `const bit[n]` variables remain unsupported.

Both complete operands are evaluated left to right before any mismatching bit
can end the comparison. Outer logical operators retain short-circuit evaluation,
so runtime checks in an unreached comparison are skipped. Types, names, widths,
and static selection bounds are still checked on skipped paths and in empty
loop bodies, without evaluating runtime shift counts or reading bits.

Runtime equality uses element-wise QCX `JUMPIF`/`JUMP` instructions, not packing
into a native integer, so registers wider than native `UINT` are supported.
Private intermediate storage stays alive until both operands are evaluated and
is released afterward. No new bra instruction is required. Unequal operand
widths, scalar/register mixing, ordered register comparisons (`<`, `<=`, `>`,
`>=`), bare register conditions, dynamic selections, aliases, and indexing of
computed register results remain unsupported.

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
runtime-indexed `int` array elements. They evaluate the count and complete a
checked shift in private storage before writing the destination. Destination
indices are captured and checked once before the RHS; self-references read the
original value. Active loop iterators remain read-only. Names, types, and index
syntax are validated in empty loop bodies without evaluating counts or index
values there. General signed arithmetic and count-expression overflow checks,
scalar-bit shifts, UINT arrays, and signed-width emulation remain future work.
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

Multidimensional arrays, non-`int` arrays (including `uint` arrays), slices,
whole-array arithmetic or assignments, array-copy initializers, aliases, and
block-local array declarations remain unsupported.

### Runtime integer-array indexing

Runtime indices support scalar `int`, `uint`, supported integer
arithmetic, enclosing runtime iterators, and indexed integer-array elements.
Nested accesses such as `values[indices[i]]` are accepted. Floating-point,
Boolean, and scalar bit indices require an explicit integer cast, for example
`values[int(flag)]`. Whole-register casts retain their existing restrictions.

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
Dynamic ranges and discrete sets, multidimensional indexing, and aliases remain
unsupported. Whole-register and static-selection bitwise operations and shifts
are described above.

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

Qubit indices, range bounds, range steps, and discrete indices must be signed
integer literals outside constant loop scopes. Bit selections also accept
supported constant `int` and `uint` expressions, including named constants and
casts. Both range bounds must be present. A zero step, an empty
range, or an out-of-bounds index is rejected during conversion.

## Classical control flow

The converter supports `if`/`else` statements with comparisons, direct scalar
bit and Boolean conditions, and logical operators. The supported comparison
operators are `==`, `!=`, `>`, `<`, `>=`, and `<=`. OpenQASM `!=` is emitted as the QCX
not-equal operator `\=`.

Conditions can compare scalar `int`, `uint`, `float`, and `bit` expressions.
A bit-register or integer-array element with a supported static or runtime
index is also accepted.
Compatible integer and floating-point operands are promoted when necessary. Complex
values cannot be compared. Equal-width bit-register `==` and `!=` comparisons
are supported as described in [Bit-register equality](#bit-register-equality).
Boolean and scalar bit operands can be compared with one another using `==`
or `!=`.

A scalar `bit`, a statically or runtime-indexed element of a bit register, or a
Boolean literal, constant, or variable can also be used directly as a condition:
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
supported. Integer-array and bit-register iteration use the runtime lowering
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
General local declarations remain unsupported. Integer-array and bit-register
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
body copies of runtime-valued set loops; repeated set elements each count as an
expanded iteration. A `break` does not reduce the generated iteration count.
Runtime-bound range, integer-array, and bit-register loops do not expand their
runtime iteration count.

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
bit-register elements may also be cast. Whole-register casts and runtime complex-to-`int`
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
follow the normalization and promotion rules above. Integer-array and
bit-register iteration are described below; alias iteration remains unsupported.
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
instructions. Bit-register iteration is described below. Iteration over slices,
non-`int` arrays, aliases, or arbitrary array expressions remains unsupported.

## Bit-register for loops

The converter supports `for bit name in flags` when `flags` is a previously
declared `bit[N]` register. Elements are visited in increasing index order,
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

The converter uses the same live-read policy as integer-array loops: each
iteration copies the current element into the iterator before executing its
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
iterator shadows a size constant. A register used for iteration must have a
positive size that fits the host-derived QCX `INT` range described under
runtime-bound loops. This assumes a backend with compatible integer storage.

Bodies support the same statements as other loops, including measurement,
conditional gates, and nested range, set, array, bit-register, and `while` loops.
`break` exits the nearest loop; `continue` advances to the next register element.
Computations stay on their original execution paths; skipped branches and
transfers skip runtime arithmetic and accesses, while conversion-time validation
still applies.

The body is generated once using existing QCX storage, indexed operands, labels,
`LET`, `JUMP`, and `JUMPIF`. Private storage holds the current index and copied
bit value throughout the body. Singleton registers use their existing scalar
QCX storage. An endpoint check precedes advancement, keeping loop-control
arithmetic representable for accepted register sizes. No new bra instruction is
required. Runtime iteration counts do not consume expansion budgets, but nested
expanded loops and enclosing expansions retain their existing accounting.

Bit iterators may index integer-array and bit-register elements through an
explicit integer cast. Runtime user-supplied qubit indices, register slices as
iterable sources, aliases, general array-type iteration, and block-local declarations
remain unsupported.

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

- delays, iteration over non-`int` arrays or aliases,
  or `switch` statements;
- user-defined gates or gate modifiers;
- runtime qubit indices, dynamic register ranges or discrete sets, ranges with
  omitted bounds, or multidimensional indexing;
- classical arrays beyond the one-dimensional `int` subset above,
  whole-register casts, complex-to-Boolean casts,
  Boolean arithmetic, mixed Boolean/numeric comparisons, or block-local
  declarations;
- scalar-bit shifts and runtime-dependent register selections,
  ordered bit-register comparisons, whole-array bitwise operations, direct
  Boolean bitwise operations, or classical functions; or
- arithmetic operators other than `+`, `-`, `*`, `/`, and integer `%` and `**`.

Integer limitations additionally include `uint[n]` widths above native `UINT`,
UINT arrays or iterators, fixed-width signed reinterpretation casts from UINT,
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
ordered-comparison rejection, empty-loop validation, iterator shadowing, and
temporary cleanup. The equality numerical suite also passes with UBSan enabled.

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
for signed operands, nested expressions, operand preservation, short-circuiting,
compound assignments, self-references, and temporary reuse.

`bra/test/qasm2qcx_uint_numerical.py` compares constant folding and runtime
unsigned arithmetic, remainder, bitwise expressions, complement, and compound
assignments at widths 1, 4, 8, and native UINT. It covers intermediate masks,
native endpoints, mixed signed/unsigned promotions, casts, comparisons,
checked conversions, and skipped runtime errors. Integration cases verify
UINT range bounds and steps, set captures, `while`, `break`/`continue`, array
and bit indices, measurement destinations, quantum parameters, singleton
storage, and bounds failures. Unit tests also cover invalid widths, unsupported
signed-width combinations, constant loop capture limits, and temporary reuse.

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

`bra/test/qasm2qcx_bit_register_numerical.py` verifies singleton and larger
registers, bit-string element order, live mutation, casts and conditions,
scalar assignments, mixed loop nesting and typed shadowing, repeated entry,
measurement-controlled transfers and source updates, gate parameters, and
skipped arithmetic and bounds checks. Mutation and position-based transfers
are checked against 288 explicit Python reference cases. Unit tests additionally
cover iterator typing, invalid sources and iterator types, storage cleanup,
reserved names, fixed-size code generation, and expansion budgets.

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
ulimit -c 0
python3 bra/test/jumpif_numerical.py --bra bra/bin/bra
python3 bra/test/assert_numerical.py --bra bra/bin/bra
python3 bra/test/integer_division_numerical.py --bra bra/bin/bra
python3 bra/test/integer_bitwise_numerical.py --bra bra/bin/bra
python3 bra/test/integer_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_if_else_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_integer_remainder_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_integer_power_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_negation_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_for_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_while_loop_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_for_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_step_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_set_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_integer_array_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bit_register_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_runtime_bit_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bitwise_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bit_register_bitwise_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bit_register_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_bit_register_equality_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_static_bit_selection_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_uint_numerical.py --bra bra/bin/bra
```
