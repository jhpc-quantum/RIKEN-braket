# Bounded diagonal-certification foundation: Stage 1 design

This document proposes a maintained module and a shared budget for #642. It
does not implement the module or enable a converter path.

## Scope and module boundary

Introduce `bra/qcx/certified_diagonal.py`, independent of `bra/test` and of
the OpenQASM AST. It should use the Python standard library and accept only
validated fixed-gate names, exact values, and explicit budget objects. The
converter remains responsible for AST eligibility, source diagnostics,
qubit/control handling, and CLI atomicity when integration is implemented.

The initial internal interface can provide:

- `BudgetLimits` and a mutable `Budget` created once per source gate call;
- exact fixed-gate composition, returning an immutable exact 2x2 matrix and an
  exact diagonal/scalar classification;
- rational principal-phase enclosures for the two diagonal entries, retaining
  their computational-basis indices and the exact `-1` endpoint convention;
- exact decimal rounding/serialization and a conservative operator-error
  check for a single rational exponent and explicit exact-real error goal; and
- distinct errors for invalid inputs, unsupported gates, budget exhaustion,
  and an unmet error goal. A later integration layer maps these to converter
  diagnostics without printing partial QCX.

The maintained API need not expose general non-diagonal eigenvalue or
projector certification: #642's first consumer is the exact diagonal path.
Its exact-real certificate must not imply native bra parsing or execution
accuracy. Negative or positive zero exponents, scalar phases, and exact `-1`
must retain the semantics described in #640.

The current test suite sometimes loads `qasm2qcx.py` directly by file path,
while users can run it as a script. Integration must explicitly test both
loading modes; an unqualified sibling import must not be assumed to work in
every mode. This issue can test the maintained module directly without
changing `qasm2qcx.py` or importing prototypes at runtime.

## Shared-budget semantics

Every potentially expensive public operation receives the *same* `Budget`
instance. It must not replace it with a fresh budget for nested gates,
individual diagonal phases, retries at higher precision, serialization, or
broadcast copies. The future converter owns the budget lifetime and charges
AST expansion and output to it; #642 tests those charges through the module
interface without enabling the converter path. Budget use is monotonic even
when an operation raises. A new source call receives a new budget.

Track separately:

- expanded gate/call work and nesting depth, aligned with the existing 100,000
  expansion-step and 128-level limits;
- exact input and intermediate integer bit lengths, denominator exponents,
  rational numerator/denominator bit lengths, and maximum working precision;
- total charged arithmetic/refinement work and arctangent-series terms;
- serialized characters and prospective QCX output lines, including both
  phases and all broadcast copies.

Use the #640 values as candidate limits, not proven defaults: 4,096 bits for
exact coefficients, 8,192 bits for rational components and permitted temporary
integers, 1,024 interval precision bits, 512 terms per series, 100,000 charged
work units, and 200 decimal places per angle. Existing 100,000 gate-definition
and 1,000,000 enclosing-loop output-line limits still apply at integration.
The 4,096-bit exact-coefficient limit is separate from the larger permitted
rational/intermediate-integer limit. A 512-term cap does not guarantee that
every request up to 1,024 precision bits succeeds: exhaustion is a valid error.

Work units must have a documented meaning: at least each gate expansion,
checked integer/rational arithmetic operation, series term, refinement, and
serialized output contribution consumes work. Input lengths must be bounded
before parsing large decimal integers. No global or cross-call cache may bypass
budget charges; initially avoid caching. All checks and charges are performed
before the allocation or operation they guard, and the error identifies the
exhausted category. Limits must cover intermediate work as well as retained
results, including failed attempts.

The size and operation limits bound the arithmetic workload under stated
assumptions; they are not a hard process-RSS guarantee. Python object overhead,
the parser, and other converter work need their own accounting. Bound the
number and sizes of retained certificate values too, rather than retaining an
unlimited sequence of individually bounded intermediates.

## Preflight rules to implement and test

- For a left shift, check the operand bit length plus the shift before creating
  the result. For multiplication and addition, use conservative bit-length
  upper bounds before constructing products and sums; rejection of an
  otherwise reducible result is acceptable and should be documented.
- Apply those checks to exact-matrix multiplication, exponent and coefficient
  parsing, sign comparisons involving squares, interval square-root scaling,
  arctangent powers, midpoint calculations, and decimal `10**digits` scaling.
- `Fraction` operations can allocate large cross-products and run GCD before a
  final value is visible. Wrap or replace them with checked rational operations
  that preflight those intermediates; a final `bit_length()` check is not
  sufficient. Charge normalization and GCD work under bounded operand sizes.
- Check requested precision, series length, angle character count, and output
  contribution before allocating their buffers. Keep phase results and their
  computational-basis labels paired; do not snap near-cut values.
- Test exactly-at-limit and one-over-limit cases, shared-budget exhaustion
  across two phases and retries, cleanup after errors, no cache-dependent budget
  behavior, and deterministic diagnostics. Cross-check successful results with
  the existing independent references. A failure must never return a partial
  certificate as successful.

## Initial measurements and open decisions

As a small host-side observation, the existing prototype's final pi-enclosure
fractions used at most 92, 164, 330, and 667 bits at requested precisions 32,
64, 128, and 256, respectively. For `t` phase enclosures at 32, 64, and 128
target bits, the largest final numerator/denominator was 205, 320, and 553 bits.
For the non-diagonal sequence `(h; t;)` repeated 12 times, the corresponding
final phase fractions reached 998, 2,743, and 8,139 bits. These are final-value
measurements of an unbounded prototype, not peak intermediates, production
benchmarks, or worst-case bounds; the non-diagonal case is outside the first
consumer's scope. They show why limits must be tested against actual rational
growth rather than inferred from requested phase precision alone.

Stage 2 should implement the budget and exact composition first, using small
configurable limits in tests. Stage 3 should add checked phase/serialization
operations and measure both intermediate growth and failure behavior. An
independent review of the enclosure and preflight arguments remains necessary
before any user-facing certificate is enabled.

## Stage 4 foundation review

`bra/qcx/certified_diagonal.py` now provides the unconnected foundation:
bounded exact fixed-gate matrix composition and classification, rational
phase enclosures, exact decimal serialization, and a conservative exact-real
operator-error check for exactly diagonal unitary matrices. `Budget` counters
are shared by the caller across these operations. The rational helpers
preflight cross-products and sums before `Fraction` normalization; they may
conservatively reject values that would become smaller after reduction. Pi
and arctangent work is recomputed and charged on every call rather than using a
cross-call cache.

The current module tests cover fixed-gate sequences up to length three,
endpoint and synthetic near-cut phases, scalar phase, decimal rounding,
negative and large exponents, shared and exhausted budgets, and agreement with
the earlier diagonal prototype. `certified_diagonal_numerical.py` checks
controlled endpoint, scalar, and non-endpoint QCX examples with the maintained
module's strings. These examples check backend behavior at printed amplitude
precision; they do not certify parsing or execution error.

The module is **not connected** to `qasm2qcx.py`. Its counters do not yet
account for OpenQASM AST expansion, nested definitions, broadcast copies, or
the converter's output; a future caller must charge and enforce those using the
same budget. Input values must already be exact bounded objects; a later CLI
must bound decimal input before constructing large Python integers. The
configurable limits are trusted inputs, and modeled work/size limits are not a
hard bound on Python process memory or wall time. The cost model, candidate
defaults, interval arguments, and preflight bounds still need independent
review and more adversarial testing before user-facing integration.

The module's certificate compares the ideal powered diagonal operation with
serialized decimal angles treated as exact real values. It does not infer a
target format or certify bra parsing, transcendental functions, or gate
execution. The CLI accuracy option, exception mapping, output atomicity, and
direct-file versus script import behavior remain integration work.

```bash
python3 -m unittest discover -s bra/test -p 'test_certified_diagonal.py' -v
python3 bra/test/certified_diagonal_numerical.py --bra bra/bin/bra
```
