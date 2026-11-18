# Exact spectral bridge feasibility study

This prototype for #634 investigates one floating-point power of a fixed,
exactly composed one-qubit gate sequence. It is separate from `qasm2qcx.py`
and does not change converter behavior.

## What the prototype does

`qasm2qcx_exact_spectral_bridge_prototype.py` validates exact unitarity and
handles scalar matrices and matrices with eigenvalue `-1` symbolically. A
nonscalar endpoint has a second exact eigenvalue `-det(G)`; the endpoint
projector is kept as an exact numerator and denominator, without requiring
division in the original ring. The endpoint represents phase `+pi` under the
converter's convention.

For branch-side investigation, exact `+1` and `-1` eigenvalues are handled
separately. Other roots of the characteristic polynomial are enclosed using
Python `Fraction` arithmetic, integer square-root bounds, and adaptive binary
precision. The resulting rational rectangles prove whether each enclosed
eigenvalue has positive or negative imaginary part; an unresolved enclosure
raises `CertificationLimit` when the requested precision budget is reached.
Exact input coefficient and denominator sizes are checked against a separate
budget. Exact-ring eigenvalues use an integer-only sign test for expressions
`a + b*sqrt(2)`.

The labels for general quadratic roots follow the principal discriminant
square root. They are not yet tied to certified eigenvectors or QCX output.
The tests cross-check selected roots against an independent binary64 quadratic
calculation; this is useful evidence, not a formal proof or broad precision
audit. The precision-exhaustion test uses a deliberately widened enclosure.

## Numerical bridge experiment

`qasm2qcx_exact_spectral_numerical.py` retains the exact `-1` decision while
constructing approximate binary64 phases and an eigenbasis. For the tested
cases, `z`, `s; s;`, and four `t` gates produce the same principal square root.
The accompanying `bra` experiment checks a controlled four-`t` example against
full complex amplitudes. Additional tests compare selected non-endpoint powers
with an independent matrix reference.

This bridge is exploratory, **not certified QCX lowering**: it uses interval
midpoints and binary64 transcendental functions, does not prove phase or
eigenvector error bounds, and has not been audited across backend precisions.
It rejects an approximation that loses a certified imaginary-side sign. It
also rejects a small binary64 eigenvalue separation using an experimental
threshold; the threshold is a rejection policy, not a rule for merging gates
or snapping an eigenvalue to a branch boundary.

## Dependency and resource considerations

The prototype demonstrates that rational enclosures for this restricted 2x2
branch problem can be explored with the Python standard library and no new
dependency. It is specialized and unaudited: its arithmetic, root labeling,
degeneracy handling, and bounds would need independent review and substantially
broader tests before production use. Precision and input-size limits alone do
not yet constitute a complete production work, memory, or output budget.

A mature ball-arithmetic library such as [Arb](https://fredrikj.net/arb/overview.html)
offers rigorous enclosures and a broader numerical foundation, but introduces
dependency, packaging, version, and compatibility choices. No dependency was
added, and this study does not benchmark the alternatives or determine that
either approach is preferable. Exact equality tests remain necessary for
endpoints: increasing interval precision alone need not establish equality.

## Recommendation

Continue the certified approach as a separate design task before integrating
it into the converter. Define a phase and eigenbasis error contract for the
backend precisions, preserve eigenvalue-to-eigenvector labeling through that
process, and establish work, memory, and output limits. Compare the cost and
maintenance of the standard-library approach and an audited interval library.
The exact endpoint case and near-cut non-endpoints should retain their proven
branch decisions without epsilon snapping or silent fallback to an uncertified
path.

Ordered nested floating powers, runtime parameters, and general multi-qubit
bodies are outside this study. Even after the initial spectrum is certified,
intermediate powered phases need their own branch and error treatment.

## Reproduction

```bash
python3 -m unittest discover -s bra/test -p 'test_qasm2qcx_exact_spectral*.py' -v
python3 bra/test/qasm2qcx_exact_spectral_numerical.py --bra bra/bin/bra
```

The first command uses the Python standard library. The second requires a
working `bra` executable and runs only the controlled endpoint example.
