# Single-qubit gate power feasibility study

This study for #630 investigates floating-point powers of fixed,
unparameterized, multi-operation one-qubit user-gate bodies. The prototype is
separate from `qasm2qcx.py` and does not add converter support.

## Approach

`qasm2qcx_single_qubit_composition_prototype.py` composes fixed gates as
`c * (a I - i v.sigma)`, keeping the overall complex phase. The accompanying
reference uses explicit 2x2 matrices and ordinary matrix multiplication. The
spectral experiment uses the same scalar/vector/eigenphase idea as the existing
U-family lowering, but is a Python host-side model; it does not call or prove
equivalence to the converter's QCX spectral preparation.

`qasm2qcx_single_qubit_spectral_numerical.py` also runs a hand-constructed
controlled QCX spectral sequence for `h; s;` against independent amplitudes.
It tests one point away from the branch cut, not every backend precision or
the converter's full lowering.

## Results and decision

Direct composition agrees with the matrix reference for all 1,111 fixed-gate
sequences of length at most three, to the test tolerance. Tests also cover
operation order, exact scalar examples, negative-real endpoints, nearby branch
values, and ordered spectral powers. The controlled QCX example agrees with
the independent complex amplitudes on the tested backend.

Rounding affects exact identities and branch decisions. On the host used for
this study, `h; h;` has a scalar component of `-1.0000000000000002` instead
of `-1`, and eight `t` gates leave a vector component of about `-1.67e-16`.
More consequentially, the composed eigenvalue corresponding to `-1` in
`t; t; t; t;` has a small negative imaginary component. The model selects the
lower side of the cut and returns approximately `sdg` for its square root;
the mathematically equivalent `z` gate uses the `+pi` endpoint and has
principal square root `s`. Exact numeric values and signs can vary by host.

**Recommendation:** do not integrate this naive floating-point composition
as general converter support yet. First decide whether native-precision branch
selection for composed fixed gates is an acceptable public contract. If exact
gate identities and endpoint consistency are required, investigate an exact or
certified representation for the chosen finite gate set, including growth and
resource limits. Such a representation is a candidate for further work, not
an outcome established by this prototype. Do not use epsilon-based snapping:
gates genuinely close to opposite sides of the cut can have different roots.

The remaining design also needs to specify how intermediate modifier powers
are represented, how errors and limits are reported, and how any eventual QCX
implementation behaves under different native precisions. Parameterized gates,
runtime floating-point exponents, and multi-qubit bodies remain separate work.

## Reproduction

```bash
python3 -m unittest discover -s bra/test -p 'test_qasm2qcx_single_qubit*.py' -v
python3 bra/test/qasm2qcx_single_qubit_composition_prototype.py
python3 bra/test/qasm2qcx_single_qubit_spectral_numerical.py --bra bra/bin/bra
```

The first two commands require Python's standard library. The last requires a
working `bra` executable. The command-line examples report experimental
behavior and do not change converter output.
