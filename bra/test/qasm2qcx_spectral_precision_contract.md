# Spectral precision-contract study

This document and `qasm2qcx_spectral_target_prototype.py` are prototypes for
#636. They do not change converter behavior or certify a QCX program.

## Target model

`bra::real_type` is selected at build time as C++ `float`, `double`, or
`long double`, with `double` the default. The converter can run on a different
machine from `bra`. Consequently, the word `long double`, Python's host float,
or the converter machine's `numeric_limits` alone do not describe the target.
In particular, C++ `long double` precision and exponent range are
implementation-dependent.

The prototype describes a binary target explicitly with:

- significand precision in bits, including the leading bit;
- smallest and largest normal leading-bit exponents;
- whether subnormal values are represented; and
- a nearest-even rounding assumption.

`binary32` and `binary64` are example mathematical profiles, not automatic
claims about a particular `bra` executable. An extended-precision descriptor
must also be supplied explicitly. The effective floating-point environment,
including flush-to-zero behavior and rounding mode, must be checked separately.
Decimal or non-nearest targets are outside this prototype's model.

The converter currently does not accept or discover this descriptor. A
production interface could accept verified build metadata, an explicit target
profile, or a requested accuracy with compatible target metadata. This study
does not yet select a user interface or establish that arbitrary profiles are
implemented correctly by a particular backend.

## Accuracy request

The prototype stores a positive absolute operator-error target as an exact
`Fraction` alongside the format descriptor. This is a request, not evidence
that it is met. Stage 2 should define the exact comparison between the ideal
powered unitary and the operation described by serialized QCX parameters,
including phase, basis, exponent, and serialization errors. Branch choices
must be established separately; an error tolerance must not silently merge or
snap eigenvalues near the cut.

The request does not include errors from bra's decimal parsing, transcendental
functions, or quantum execution. The QCX interpreter currently parses numeric
gate operands into `bra::real_type` using `boost::lexical_cast`; its rounding
and error behavior on each supported toolchain still need verification. Tests
of resulting bra amplitudes are a separate backend audit. A converter-side
bound must not be reported as an end-to-end backend guarantee.

The converter's existing constant floating exponents are evaluated using
Python's floating-point precision. A future contract must say whether its ideal
operation uses that evaluated value or a different exact interpretation of the
source expression; the prototype does not change those semantics.

## Conditional operation bound

`qasm2qcx_spectral_error_bound_prototype.py` evaluates a conservative bound
using exact rational arithmetic. It assumes that an independent process has
already established the branch, paired each eigenphase with the corresponding
eigenbasis column, and bounded each supplied error. Matching labels in the
prototype catches an accidental mismatch; labels alone are not proof that
the pairing or branch is correct.

For `V = U3(beta, eta, 0)` and `D = diag(exp(i*alpha0), exp(i*alpha1))`,
the ideal operation is `V D V+`. If the basis parameter errors are `dbeta`
and `deta`, a telescoping operator-norm bound for both basis operations is
`dbeta + 2*deta`. For each diagonal phase, the prototype bounds the powered
angle error by `abs(r)*dtheta + (22/7)*dr + dout`, where `r` is the exponent
used to form the emitted angle, `dtheta` bounds the selected unpowered
principal phase, `dr` bounds the difference from the ideal exponent, and
`dout` covers subsequent angle calculation and serialization. The comparison
uses the same labeled branch and a consistent real-valued phase representative;
all ideal principal phases have magnitude at most pi.

The total is the basis bound plus the larger of the two phase bounds. This
includes global phase. The same converter-side operator bound applies to a
controlled version with an exact identity inactive block. It does not include
bra parsing or execution. Large exponents magnify phase uncertainty, and a
bound exceeding the requested error is rejected by the prototype.

This is a conditional bound only. The prototype does not yet derive certified
phase, eigenbasis, or serialization error bounds from exact matrices, and it
does not prove that the target descriptor matches a bra binary. Numerical
perturbation tests check the formula on representative 2x2 operations but are
not a proof of those missing inputs or of backend behavior.

## Backend precision experiment

`spectral_precision_backend_probe.cpp` is a separate C++ probe. It parses
decimal angles with `boost::lexical_cast` and calls ket's `exp_i` for C++
`float`, `double`, and `long double`. It reports parsing error against a
100-decimal-digit reference and `exp_i` error against a reference evaluated at
the parsed value. The latter separates the phase primitive's observed error
from the initial parsing error. This high-precision reference is numerical,
not a certified enclosure.

On the host used for this study, the three C++ types reported 24, 53, and 64
significand bits respectively. Decimal inputs at pi and on either side by
`1e-20` parsed to the same native value within each type. The `large-phase`
sample, approximately `3.14e6`, had parsing errors of about `9.64e-2`,
`2.23e-10`, and `8.97e-14`, respectively. These are observations for the
particular inputs and toolchain, not universal error bounds. They illustrate
why an already-certified branch cannot be recovered by reclassifying a rounded
native angle and why large-phase parsing belongs in an end-to-end error budget.

The probe is compiled separately with `-O2`; it is not a `bra` build and does
not establish behavior for release builds, which currently use `-Ofast`, or
for other floating-point environments. The available `bra/bin/bra` executable
passed the existing controlled exact-endpoint QCX amplitude experiment. Its
type was not independently identified, and separate float and long-double
executables were not available for this study. The amplitude output uses
default stream precision, so that check cannot establish double- or
long-double-level accuracy. No change to build options is proposed here.

## Recommendation and remaining work

The proposed contract is an explicit target descriptor and a requested
absolute operator-norm error for the serialized converter output. It retains
the exact branch decision and full phase, pairs each eigenphase with its basis
column, and includes angle arithmetic and serialization in the bound. The
conditional formula above is one way to combine independently certified input
errors. A result is acceptable only if the supplied bounds are established and
the resulting total meets the request; a branch label or a passing numerical
sample alone is not a certificate. The same converter-side bound applies to
controlled use with an exact identity inactive block.

This study does **not** establish an end-to-end QCX/bra guarantee or a universal
default tolerance. Backend parsing and execution are separate from the
converter's exact-decimal-output contract. Actual target formats and operating
environments must be identified, and backend rounding, transcendental, and
gate-execution behavior must be analyzed or explicitly treated as outside that
contract. The existing Python binary64 folding of floating exponents also
needs a defined relationship to the ideal source operation.

Do not integrate the bound into supported converter paths yet. Before that,
implement and independently validate certified phase, eigenbasis, and decimal
serialization error bounds; demonstrate consistent eigenvalue-to-eigenvector
pairing; define behavior for exact scalar, near-cut, and nearly degenerate
matrices; and establish coefficient-size, refinement-work, memory, and output
limits. Resource exhaustion or an uncertifiable result should produce a clear
conversion error instead of silently switching to an uncertified branch.

The next practical work should focus on obtaining certified input bounds for
the restricted single-power fixed-gate subset and comparing the cost of an
audited interval library with a reviewed standard-library implementation. The
current target descriptor and conditional formula are prototypes, not an
endorsed final CLI or an established numerical guarantee. Runtime parameters,
ordered nested floating powers, and multi-qubit bodies remain separate work.

## Stage 1 validation

The descriptor tests check exact unit-roundoff calculations, normal and
subnormal exponent models, explicit extended precision, validation, and exact
accuracy requests. They do not check an actual bra build or establish a
floating-point error bound.

```bash
python3 -m unittest discover -s bra/test -p 'test_qasm2qcx_spectral_target_prototype.py' -v
python3 -m unittest discover -s bra/test -p 'test_qasm2qcx_spectral_error_bound_prototype.py' -v
python3 -m unittest discover -s bra/test -p 'test_qasm2qcx_spectral_precision_probe.py' -v
g++ -std=c++14 -O2 -Iket/include bra/test/spectral_precision_backend_probe.cpp -o /tmp/braket-spectral-precision-probe
/tmp/braket-spectral-precision-probe
python3 bra/test/qasm2qcx_exact_spectral_numerical.py --bra bra/bin/bra
```
