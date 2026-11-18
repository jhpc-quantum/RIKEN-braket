# Certified spectral input feasibility study

This study for #638 investigates one compile-time power of an exactly composed,
fixed single-qubit gate sequence. The prototypes are separate from
`qasm2qcx.py` and do not add converter support.

## Prototypes and findings

`qasm2qcx_certified_phase_prototype.py` constructs rational principal-phase
enclosures from the exact endpoint and eigenvalue-side information of #634.
It uses an alternating arctangent series, rational remainder bounds, and
Machin's identity for pi; it does not rely on host float arithmetic. The exact
`-1` endpoint uses `+pi`. Requested enclosure width, input coefficient size,
working precision, and series terms have explicit prototype limits. Exceeding
the limits raises an error rather than guessing a branch.

`qasm2qcx_certified_projector_prototype.py` encloses spectral projectors
directly using rational complex interval operations and retains each root
index. It uses a scalar path without an arbitrary basis, refines enclosures for
distinct roots, and rejects a gap enclosure containing zero if it cannot be
resolved within the precision budget. Projectors avoid eigenvector phase
choices, but these enclosures do **not** yet provide certified `U3` basis
angles or serialization error bounds for non-diagonal gates.

`qasm2qcx_certified_diagonal_prototype.py` connects a restricted case to #636's
conditional operator bound. For an exactly diagonal unitary, the computational
basis is exact. It bounds the unpowered phases, rounds powered angles to
decimal strings with exact rational nearest-even arithmetic, and accounts for
those serialization errors. It rejects a request if the calculated bound is
too large. It also rejects non-diagonal matrices; it does not turn the general
projector enclosures into QCX basis angles.

## Evidence and limits

The prototype tests cover exact endpoints and scalar phases, independent
numeric cross-checks, projector identities, label ordering, budget errors, and
a 900-gate fixed sequence with a non-endpoint eigenphase near `-pi`. Controlled
QCX experiments for four `t` gates and a scalar-phase gate matched expected
complex amplitudes using the available `bra` executable.

The rational enclosure code and error arguments remain experimental and have
not had a formal proof or independent audit. Numeric cross-checks cannot prove
the interval bounds, and selected examples do not characterize performance
or numerical behavior for all fixed-gate sequences. Root indices rely on the
same deterministic exact-root labeling across phase and projector calls; the
prototype does not yet expose a unified certificate that binds phases to
projectors and serialized basis parameters.

The diagonal bound treats emitted decimal strings as exact real numbers and
the supplied rational exponent as the exact ideal exponent. It does not certify
bra's decimal parsing, transcendental functions, gate execution, or that the
target descriptor matches the bra binary. The prototype limits do not yet
bound total memory or all intermediate integer and rational operations.

All current calculations use Python's standard library; no interval library
was added. This is not a performance or maintenance comparison with an audited
interval-arithmetic library. The requested precision or work may still be too
expensive for a practical production limit.

## Recommendation

Continue with a separate production-design task before integrating these
prototypes into the converter. It should decide whether the first supported
subset should be exactly diagonal gates, or whether to establish a certified
bridge from projectors to a serializable QCX eigenbasis first. Either choice
needs independently reviewed interval and error arguments, explicit complete
resource limits, target-format and exponent semantics, and a clear separation
of converter guarantees from backend error.

The non-diagonal path specifically needs a unified root/projector/phase
certificate and a bounded method for converting it to QCX operations without
losing labeling near degeneracies. Runtime parameters, ordered nested floating
powers, and multi-qubit bodies remain separate work.

## Reproduction

```bash
python3 -m unittest discover -s bra/test -p 'test_qasm2qcx_certified*prototype.py' -v
python3 bra/test/qasm2qcx_certified_diagonal_numerical.py --bra bra/bin/bra
```

The first command needs Python's standard library. The second requires a
working `bra` executable and checks the controlled endpoint, scalar-phase, and
non-endpoint examples.
