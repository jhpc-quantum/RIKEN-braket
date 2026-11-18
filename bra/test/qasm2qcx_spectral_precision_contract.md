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

## Stage 1 validation

The descriptor tests check exact unit-roundoff calculations, normal and
subnormal exponent models, explicit extended precision, validation, and exact
accuracy requests. They do not check an actual bra build or establish a
floating-point error bound.

```bash
python3 -m unittest discover -s bra/test -p 'test_qasm2qcx_spectral_target_prototype.py' -v
```
