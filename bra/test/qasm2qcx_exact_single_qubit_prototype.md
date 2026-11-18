# Exact single-qubit composition feasibility study

This prototype for #632 investigates fixed, unparameterized one-qubit gates.
It is separate from `qasm2qcx.py` and does not add converter support.

## Representation and results

Each complex entry is represented exactly as
`(a + b*sqrt(2) + i*(c + d*sqrt(2))) / 2**k`, using Python integers and a
nonnegative integer `k`. Common powers of two are removed after arithmetic.
The gate matrices for `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, and
`sx` are exactly unitary in this representation. Gate matrices are composed
in execution order, retaining their full phase.

Tests establish exact identities including `h; h; = id`, four `t` gates equal
`z`, eight `t` gates equal `id`, and `z; y; x; = i*id`. All 1,111 sequences
of at most three fixed gates also agree, to floating-point tolerance, with the
independent direct-matrix reference. Agreement with a floating-point reference
is a consistency check, not a proof for longer sequences.

The prototype detects scalar matrices by exact equality and tests for an
eigenvalue of `-1` using exact `det(G + I) == 0`. The latter identifies the
negative-real endpoint for an exactly unitary matrix; it does not compute the
eigenvectors, classify other eigenvalues, or safely convert the result to QCX
floating-point values. The test is also valid as a `-1` eigenvalue test for
nonunitary matrices, but the branch interpretation here assumes unitarity.

## Sample resource measurements

The growth script reports the largest numerator bit length and denominator
exponent after selected prefixes. On the tested host, a 2,048-gate `h; t`
alternating sequence reached 257 bits and exponent 257; repeating `t` alone
cycled through small entries. These are measurements of selected patterns, not
worst-case bounds. Execution times depend on the host and Python version. Any
production path needs explicit work and integer-size limits, including limits
on intermediate products and normalization, rather than relying on this sample.

## Numerical design still needed

Exact composition prevents the demonstrated `t; t; t; t;` identity from being
lost during composition. It does not by itself make a subsequent floating-point
eigensolver branch-correct. A production design must preserve an exact endpoint
decision through spectral preparation and must distinguish values arbitrarily
close to the cut from the endpoint. It also needs stable, bounded conversion
of large exact coefficients to backend precision; the prototype's
`to_complex()` is for small reference checks and has no such guarantee.

Three possible policies merit further investigation:

1. **Exact composition and certified branch decisions.** Carry exact information
   through enough of the spectral calculation to decide the cut, then round
   output angles. This could preserve fixed-gate identities, but the algebra,
   numerical error bounds, and resource requirements are not established here.
2. **Certified numerical intervals.** Evaluate with bounds and increase
   precision until the branch is determined, with explicit behavior when a
   resource limit prevents a decision. Exact endpoints still require a way to
   establish equality, rather than repeated approximation alone.
3. **Native-precision semantics.** Compose and select branches using backend
   floating point, documenting that mathematically equivalent sequences can
   select different roots at a branch boundary, as observed in #630.

Do not use epsilon-based snapping: truly distinct eigenvalues on opposite
sides of the branch cut can have different principal roots. Arbitrary floating
powers within a body can leave the exact ring; runtime parameters and multi-qubit
gates also need separate designs.

## Recommendation

Exact composition is viable for a bounded fixed-gate prototype and is a useful
basis for preserving identities. Do not integrate it into user-facing lowering
yet. The next design task should specify and test a certified spectral/branch
bridge and explicit resource limits, then compare its complexity and behavior
with the native-precision contract. In particular, prove how exact `-1`
eigenvalues, near-cut eigenvalues, scalar matrices, and ordered powers are
handled before emitting QCX.

## Reproduction

```bash
python3 -m unittest discover -s bra/test -p 'test_qasm2qcx*single_qubit*prototype.py' -q
python3 bra/test/qasm2qcx_exact_single_qubit_growth.py --max-gates 2048
```

Both commands use Python's standard library and do not run `bra`.
