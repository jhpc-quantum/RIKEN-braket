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
  `p`/`phase`/`u1`, `cp`/`cphase`, `gphase`, and
  `rx`/`ry`/`rz`/`crx`/`cry`/`crz`, `u2`, `u3`, built-in `U`, and `cu`,
  with principal spectral powers and ordered modifier evaluation;
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
- scalar built-in `exp`, `sqrt`, `log`, `sin`, `cos`, `tan`, `arcsin`, `arccos`,
  `arctan`, `ceiling`, `floor`, `real`, and `imag` calls in constant and runtime
  expressions, including nested calls;
- fixed-size, one-dimensional `int`, `uint`, `float`, `bool`, and `complex` arrays
  with static and runtime element reads and writes;
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
- runtime iteration over one-dimensional `int`, `uint`, `float`, `bool`, and
  `complex` arrays;
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
| `arcsin(x)` | `float` | `float` | `APPLY result <- ARCSIN operand` |
| `arccos(x)` | `float` | `float` | `APPLY result <- ARCCOS operand` |
| `arctan(x)` | `float` | `float` | `APPLY result <- ARCTAN operand` |
| `ceiling(x)` | `float` | `float` | `APPLY result <- CEILING operand` |
| `floor(x)` | `float` | `float` | `APPLY result <- FLOOR operand` |
| `mod(x, y)` | `int`, `int` | `int` | Copy dividend to a temporary, then `LET temporary %= divisor` |
| `mod(x, y)` | `float`, `float` | `float` | `APPLY result <- MOD dividend divisor` |
| `popcount(bits)` | `bit[n]` | `uint` | Unsigned accumulation over register bits |
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
arguments; `popcount` explicitly accepts a register rather than a scalar.
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

`sin`, `cos`, and `tan` currently accept only real-valued arguments, interpreted
in radians. Complex arguments are rejected, even with a zero imaginary part
or a complex assignment destination. `angle`-typed arguments remain
unsupported. Constant expressions use Python's `math.sin`, `math.cos`, and
`math.tan`; executable calls, including literal arguments, use native bra
`SIN`, `COS`, and `TAN`. Both paths reject nonfinite inputs and nonfinite
results. Invalid evaluated constants fail conversion; executable failures
occur only when reached, including in conditions and loop expressions.

For exact zero, `sin` and `tan` preserve its sign and `cos` returns positive
one. Subnormal arithmetic follows the host or backend floating-point
environment and may flush to zero. Native calls use the C library at the
configured precision, without preliminary angle reduction or changes to
build flags or floating-point control modes. Large finite arguments remain
valid, subject to library accuracy. Values near zeros and tangent poles are
not snapped: an approximation of π/2 is not an exact pole, so its large
finite tangent is accepted. Constant and runtime results can differ because
their precisions and floating-point environments can differ.

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

`popcount(bits)` counts the set (`1`) bits of a bit register without modifying
it. It returns native, unspecified-width `uint`, independently of the destination
type. Inputs can be bit-string literals, whole registers (including `bit[1]`),
static range/set selections, or supported register `~`, `&`, `|`, `^`, `<<`, and
`>>` expressions. Repeated positions in a discrete selection are counted each
time they appear. Scalar `bit`, scalar register elements, integers, Booleans,
and whole classical arrays are not register arguments; use a one-position range
such as `bits[0:0]` when a one-bit register selection is intended.

Constant calls accept literal register expressions, using exact bit counting.
Named bit registers are runtime values even if their initializers are literals;
`const bit[n]` declarations remain unsupported. Runtime calls use a `UINT`
temporary initialized to the literal-bit subtotal, then add each live bit with
an explicit unsigned cast. Registers wider than native `UINT` are supported
without packing their contents into an integer; the register/selection width
must fit the native `UINT` range so the count cannot overflow.

The result works in unsigned arithmetic, powers, casts, numeric conditions,
indexed assignments, and supported loop expressions. As with other numeric
results, compare it with a number or use `bool(...)` for a Boolean condition.
Register-expression and count temporaries remain reserved through counting.
Skipped calls do not evaluate runtime shifts or their errors, but names, types,
widths, and static selections are validated even in empty loops. Runtime-dependent
register selections, concatenation, and casts to a whole register remain outside
the supported subset. No new native bra instruction is required.

<!-- popcount-example -->
```qasm
OPENQASM 3.0;
bit[8] flags = "10110100";
uint count = popcount(flags);          // 4
uint selected = popcount(flags[2:5]); // 3
```

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
negative value. `for uint` is supported for matching-width UINT arrays, not
for integer ranges or discrete sets.

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

Register values are supported in `popcount`, register assignments, and the equality
comparisons described below; they are not bare conditions. Integer/register casts,
indexing of computed results, or register expressions as loop bounds or
iteration collections remain unsupported. Static selection operations are described below.
Concatenation, aliases, `const bit[n]` declarations,
and register rotations remain outside this subset. Existing direct
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

Unsigned, floating-point, Boolean, and complex arrays are supported as described below.
Multidimensional arrays, arrays of other element types, slices,
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
arithmetic. Unsupported array element types, multidimensional arrays, slices,
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

Multidimensional arrays, slices, whole-array expressions or assignments,
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
slices, whole-array expressions or assignments, array-copy initializers, aliases,
and block-local declarations remain unsupported.

### Runtime array indexing

Signed/unsigned integer, floating-point, Boolean, and complex arrays accept runtime
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
OpenQASM `gphase(angle)` is emitted as the QCX global-phase instruction
`PHASE angle`.

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
- `inv @ gphase(angle)` emits `PHASE` with the negated angle. Runtime
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
- `ctrl @ gphase(angle) c` emits `U1 c angle`; it needs no extra target.
  With `n` controls, the converter emits a `U1` conditioned on the other
  `n - 1` control operands, so only the all-one basis component acquires the phase.

The same phase conditioning applies to negative and mixed controls. For
example, `negctrl @ U(1, 2, 3) c, t` emits `U1 c#0 0.5` and
`CU3 c#0 t 1 2 3`. Its inverse emits `CU3+ c#0 t 1 2 3` followed by
`U1 c#0 -0.5`, negating the phase correction exactly once. For `cu`, added
negative controls do not change the original positive control's polarity.
`negctrl @ ctrl @ gphase(angle) a, b` emits `CU1 a#0 b angle`: only the
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
inverse repeatedly. Exponent zero emits no quantum instruction, but does not
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

Constant angles are scaled for `p`/`phase`/`u1`, `rx`/`ry`/`rz`, their supported
controlled forms, and `gphase`, when the multiplier is exactly representable
as a Python float and multiplication produces an exact, finite host-float
result. Inexact integer-to-float conversion is not introduced by this
optimization. A rounded product of a large angle could otherwise change its
phase arbitrarily modulo `2*pi`, despite remaining finite.
Runtime-valued angles retain repetition: scaling an unknown angle
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
negation overflow. Nested factors use separate QCX loops instead of multiplying
full-width magnitudes. All zero magnitudes are checked before entering any
repetition loop. Source variables are not modified.

For the exact periodic gate families above, runtime factors are reduced modulo
the effective period before multiplication. Only small residues are multiplied,
avoiding overflow and bounding repetitions independently of exponent magnitude.
Other gates repeat the complete lowered operation, preserving the global and
control-phase corrections of `U` and `cu`. Runtime exponents are never used to
scale angles; constant factors may still use the constant-angle optimization
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
repetition control flow. Exponent capture and parameter preparation precede
this budget. The converter checks the budget before allocating a repeated
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
A bit-register, signed/unsigned integer-array, floating-point-array, or
Boolean-array element with a supported static or runtime index is also accepted.
Compatible integer and floating-point operands are promoted when necessary. Complex
values cannot be compared. Equal-width bit-register `==` and `!=` comparisons
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
complex-array, and bit-register iteration use the runtime lowering described below;
alias iteration remains unsupported.
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
body copies of runtime-valued set loops; repeated set elements each count as an
expanded iteration. A `break` does not reduce the generated iteration count.
Runtime-bound range, signed/unsigned integer-array, floating-point-array,
Boolean-array, complex-array, and bit-register loops do not expand their runtime
iteration count.

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
follow the normalization and promotion rules above. Signed/unsigned integer-array,
floating-point-array, Boolean-array, complex-array, and bit-register iteration are
described below; alias iteration remains unsupported.
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
complex-array, and bit-register iteration are described below. Iteration over
slices, other unsupported array types, aliases, or arbitrary array expressions
remains unsupported.

## Unsigned-integer-array for loops

The converter supports `for uint name in values` and `for uint[n] name in values`
over previously declared one-dimensional UINT arrays. The iterator and element
must have equal effective widths. Therefore an unsized `uint` iterator requires
a native-width array; `uint` and `uint[W]` are compatible when `W` is native width.
Different specified widths, including widening, are rejected rather than
implicitly converted. The iterator retains its own specified/unsized metadata.
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
UINT range/set iteration, implicit cross-base-type iteration, different-width
UINT iteration, slices, multidimensional arrays, aliases, and arbitrary array
expressions remain unsupported.

## Floating-point-array for loops

The converter supports `for float name in values` when `values` is a previously
declared one-dimensional floating-point array. Elements are visited in increasing
index order, including repeated values. Sized float base and iterator types such
as `float[32]` and `float[64]` follow the existing native-`REAL` policy: their widths
are not enforced, and differing declared float widths do not introduce rounding
or precision conversion. Signed-integer arrays require `int` iterators,
unsigned-integer arrays require matching-width `uint` iterators, and float arrays
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
ranges or discrete sets. Slices, multidimensional arrays, aliases, arbitrary
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
iterators do not enable Boolean ranges or sets. Slices, multidimensional arrays,
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
temporaries while preserving caller-owned storage. Slices, multidimensional
arrays, aliases, arbitrary array expressions, and block-local declarations remain
outside this subset.

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

- delays, iteration over arrays other than the supported
  `int`/`uint`/`float`/`bool`/`complex` subset,
  alias iteration, or `switch` statements;
- user-defined gates, gate durations, runtime floating-point `pow` exponents,
  mixed runtime-integer/floating-point power chains, floating-point powers
  outside the fixed, phase, Pauli-rotation, and `u2`/`u3`/`U` families above, or
  runtime-dependent control counts;
- runtime qubit indices, dynamic register ranges or discrete sets, ranges with
  omitted bounds, or multidimensional indexing;
- classical arrays beyond the one-dimensional `int`, `uint`, `float`, `bool`, and
  `complex` subsets above, whole-register casts, complex-to-Boolean casts,
  Boolean arithmetic, mixed Boolean/numeric comparisons, or block-local
  declarations;
- scalar-bit shifts and runtime-dependent register selections,
  ordered bit-register comparisons, whole-array bitwise operations, direct
  Boolean bitwise operations, or classical functions other than the
  `exp`/`sqrt`/`log`/`sin`/`cos`/`tan`/`arcsin`/`arccos`/`arctan`/`ceiling`/`floor`/
  `real`/`imag` subset above;
- `angle` declarations and `angle`-typed trigonometric arguments; or
- arithmetic operators other than `+`, `-`, `*`, `/`, and integer `%` and `**`.

Integer limitations additionally include `uint[n]` widths above native `UINT`,
UINT range/set iterators, different-width UINT array iterators,
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

`bra/test/qasm2qcx_uint_array_loop_numerical.py` checks live reads, copied iterator
values, mutation, singleton arrays, loop transfers against Python reference
loops, repeated entry, mixed nesting, width shadowing, compatible native-width
forms, casts, comparisons, gate parameters, skipped accesses, and executed
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
python3 bra/test/integer_remainder_numerical.py --bra bra/bin/bra
python3 bra/test/mod_numerical.py --bra bra/bin/bra
python3 bra/test/integer_bitwise_numerical.py --bra bra/bin/bra
python3 bra/test/integer_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_if_else_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_math_function_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_sqrt_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_log_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_trigonometric_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_inverse_trigonometric_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_rounding_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_mod_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_popcount_numerical.py --bra bra/bin/bra
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
python3 bra/test/qasm2qcx_static_bit_selection_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_shift_numerical.py --bra bra/bin/bra
python3 bra/test/qasm2qcx_uint_numerical.py --bra bra/bin/bra
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
