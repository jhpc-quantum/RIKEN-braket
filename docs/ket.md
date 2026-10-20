# ket

## Introduction

*ket* is a C++ template library to perform state-vector simulation of quantum computers.

## Requirements

*ket* requires a C++14 compliant compiler and the [Boost C++ library](https://www.boost.org/).
Any [MPI](https://www.mpi-forum.org/) libaries are also required if you would like to use *ket* in massively parallel supercomputers.

## Directory structure

```
ket
└── include
    └── ket
        ├── gate
        ├── mpi
        │   ├── gate
        │   └── utility
        └── utility
            └── parallel
```

* `ket/include/ket/`: `ket::qubit<S,B>` for target qubits, `ket::control<Q,control_value>` for compile-time control values, and `ket::runtime_control<Q>` for runtime control values
* `ket/include/ket/gate/`: non-MPI versions of quantum gate functions
* `ket/include/ket/utility`: some utility functions/classes, especially policy class `ket::utility::policy::sequential` to perform non-parallelized gate operation (similar to C++17's `std::execution::sequenced_policy`)
* `ket/include/ket/utility/parallel`: policy class `ket::utility::policy::parallel<N>` to perform multi-threading gate operation (similar to C++17's `std::execution::parallel_policy`)
* `ket/include/ket/mpi/`: MPI-related directory, which includes "page"-aware state-vector class `ket::mpi::state<C>`, permutation matrix class `ket::mpi::qubit_permutation<S,B>`, and `ket::mpi::permutated<Q>` class to represent indices of qubits after permutation
* `ket/include/ket/mpi/gate`: MPI versions of quantum gate functions
* `ket/include/ket/mpi/utility/`: some utility function/classes for MPI-related operations, especially policy classes `ket::mpi::utility::policy::simple_mpi` and `ket::mpi::utility::policy::unit_mpi<S,B,N>`

## Control qubits

For a qubit, a positive control activates a gate when its wire is `1`, and a negative control activates it when its wire is `0`.
With multiple controls, all their conditions must hold.
The wire index and the required value are separate: `ket::control<Q,0>{Q{3u}}` means wire 3 controlled on zero, not wire 0.

### Construction, literals, and validation

Include `<ket/control.hpp>` for `ket::control<Q,control_value>` and `<ket/runtime_control.hpp>` for `ket::runtime_control<Q>`.
`ket::control<Q>` still means a positive control: its default value is `ket::meta::base_v<Q> - 1u`, which is `1` for `ket::qubit<S,B>`.
`ket::make_control(q)` also retains its positive-control meaning; use `ket::make_control<0u>(q)` for a negative control.
Rebuild downstream C++ code when updating these headers: preserving the positive-control source spelling is not a binary-compatibility guarantee.

`ket::runtime_control<Q>{q, value}` stores the required value at runtime.
Construction from a compile-time control is explicit, for example `ket::runtime_control<Q>{negative}` or `ket::make_runtime_control(negative)`.
`ket::make_runtime_control(q, value)` is also available.
The default constructor and the constructor taking only a wire both select the positive-control value.
Changing between compile-time control values, converting a runtime control to a compile-time control, and implicitly removing a control are not supported.
Use `ket::remove_control(c)` to obtain the underlying wire explicitly, and `ket::control_value(c)` to read either kind of control's required value.

For qubits, the valid values are `0` and `1`.
An invalid compile-time value is a compilation error.
Runtime construction accepts integral values and throws `std::out_of_range` for a negative value or a value outside the base, checking before narrowing to the stored integer type.
Invalid constant-expression construction is a compilation error instead.
`ket::meta::base_v` is groundwork for future quantum-digit types; these gate kernels still simulate qubits, not qudits.

In `ket::literals::control_literals` (also imported by `ket::literals`), `3_cq` retains its positive-control meaning, `3_pcq` is its explicit positive synonym, and `3_ncq` is a negative control on wire 3.
All three families have the existing bit-index suffix variants: `s`, `l`, `ll`, `8`, `16`, `32`, and `64`, for example `3_pcqs` and `3_ncq32`.
There is no separate runtime-control literal; use `ket::make_runtime_control(3_ncq)` or `ket::make_runtime_control(3_q, 0u)`.

### Positional comparisons and arithmetic

Equality, ordering, and distances compare wire positions, not control conditions.
In particular, `3_q == 3_cq`, `3_cq == 3_pcq`, and `3_pcq == 3_ncq` are all true; this also applies to runtime controls on the same wire.
To compare complete control conditions, compare both `ket::remove_control(a) == ket::remove_control(b)` and `ket::control_value(a) == ket::control_value(b)`.
Position arithmetic such as `++c`, `c += 2u`, and `c + 1u` preserves the required value.
Positional equality does not permit using a wire twice in one gate: targets and controls must be distinct, even if two controls require different values.

### Gate interfaces

Variadic gate arguments must put all targets before all controls.
Compile-time positive and negative controls may appear in any order.
The controls must use the same underlying qubit type as the targets.
For example, `hadamard(..., 0_q, 1_ncq, 2_pcq)` is valid; placing a target after a control is not.

Runtime-range interfaces keep target and control ranges separate.
Use a homogeneous range of `ket::runtime_control<Q>` for mixed required values, or a range of `ket::control<Q,0u>` / `ket::control<Q>` when all required values are the same.
An ordinary qubit range is not a control range.
Use these runtime interfaces for runtime-valued controls in ordinary and fused gates; their specialized variadic overloads retain compile-time control parameters.
For non-MPI gates, `ket::gate::runtime::ranges` accepts a state range, `ket::gate::runtime::qubit_ranges` accepts state iterators and qubit ranges, and `ket::gate::runtime` supplies qubit-iterator overloads.

This complete C++14 example uses both variadic and range interfaces:

<!-- control-example: nonmpi -->
```c++
#include <cassert>
#include <cmath>
#include <complex>
#include <vector>
#include <ket/gate/hadamard.hpp>
#include <ket/runtime_control.hpp>

int main()
{
  using Q = ket::qubit<>;
  using namespace ket::literals;
  auto const negative = ket::make_control<0u>(1_q);
  auto const positive = ket::make_control(2_q);
  auto const dynamic = ket::make_runtime_control(negative);
  static_assert(3_q == 3_cq and 3_pcq == 3_ncq, "Positional comparisons");
  assert(ket::control_value(dynamic) == 0u);

  std::vector<std::complex<double>> state(8);
  state[4] = 1.0; // |100>: wire 1 is zero and wire 2 is one
  ket::gate::hadamard(state.begin(), state.end(), 0_q, negative, positive);

  std::vector<ket::runtime_control<Q>> controls{
    ket::runtime_control<Q>{negative}, ket::runtime_control<Q>{positive}};
  ket::gate::runtime::ranges::hadamard(state, 0_q, controls);
  assert(std::abs(state[4] - std::complex<double>{1.0, 0.0}) < 1e-12);
}
```

Compile-time controls in variadic interfaces and control ranges in runtime interfaces are supported by the controlled H, X/NOT, Y, Z, SWAP, phase/U1/U2/U3, controlled-V, square-root Pauli, half-pi rotation, and exponential Pauli/SWAP gate families and their adjoints, including fused kernels and current MPI gate interfaces.
The CNOT, Toffoli, and controlled-phase aliases also accept control values.
For fused gates, operand indices refer to wires within the fused block; their control values are unchanged by the mapping to state-vector indices.
Zero controls are selected directly by the kernels, without applying surrounding X gates to the state.

### Stream representation

Include `<ket/control_io.hpp>` for compile-time-control stream insertion and extraction, or `<ket/runtime_control_io.hpp>` for runtime controls.
Both use `wire#value`: a positive control on wire 3 prints as `3#1`, and a negative control prints as `3#0`.
The required value is always explicit, including for compile-time controls.
These operators support narrow (`char`) and wide (`wchar_t`) streams without requiring `<ket/qubit_io.hpp>`.

Extraction requires both fields and the `#` separator.
For `ket::control<Q,control_value>`, the input value must match `control_value`; for `ket::runtime_control<Q>`, it must be less than `ket::meta::base_v<Q>`.
Missing or malformed fields, negative fields, out-of-range wire representations, and invalid or mismatched required values set `failbit` and leave the destination unchanged.
Normal stream exception masks apply; the destination also remains unchanged if extraction throws.
Formatted extraction follows the stream's whitespace, integer-base, and locale settings, and leaves trailing delimiters for subsequent extraction.
Output preserves numeric flags and locale; width and fill apply to the entire `wire#value` operand.

This is an intentional stream-format compatibility change: position-only compile-time-control input and the old runtime-control `wire:value` form are not accepted.
Update saved control operands and downstream output expectations accordingly.
The QCX format is separate: it still accepts bare positive-control wires and uses bare wires for their canonical representation (see [bra control operands](bra.md#quantum-assembler)).

<!-- control-example: io -->
```c++
#include <cassert>
#include <sstream>
#include <ket/control_io.hpp>
#include <ket/runtime_control_io.hpp>

int main()
{
  using Q = ket::qubit<>;
  auto const negative = ket::control<Q, 0u>{Q{3u}};
  std::stringstream stream;
  stream << negative;
  assert(stream.str() == "3#0");

  auto dynamic = ket::runtime_control<Q>{};
  stream >> dynamic;
  assert(not stream.fail() and dynamic.qubit() == Q{3u} and dynamic.value() == 0u);

  auto positive = ket::control<Q>{Q{7u}};
  std::istringstream mismatch{"3#0"};
  mismatch >> positive;
  assert(mismatch.fail() and positive.qubit() == Q{7u});

  std::wstringstream wide;
  wide << dynamic;
  assert(wide.str() == L"3#0");
}
```

With `KET_PRINT_LOG` defined, MPI gate diagnostic logs also retain control values.
For example, a target on wire 0 with negative control 2 and positive control 3 is logged with operands `0 2#0 3#1`.
This applies to variadic and runtime-range interfaces; targets remain bare wire indices.
When logging is disabled, the diagnostic-formatting helpers still return only the base gate description.

## Nompi version

### Gate functions

Gate functions can be used such as
```c++
// Iterator versions
ket::gate::hadamard(parallel_policy, first, last, target_qubit, control_qubit1, control_qubit2);

// Range versions are also available
ket::gate::ranges::exponential_pauli_x(
  parallel_policy, state, phase, target_qubit1, target_qubit2, control_qubit);
// ... just calls
//   ket::gate::exponential_pauli_x(
//     parallel_policy, std::begin(state), std::end(state),
//     phase, target_qubit1, target_qubit2, control_qubit);
```

Here `parallel_policy` is a policy to specify if we use multi-threading gate functions or not.
The type of `parallel_policy` must be either `ket::utility::policy::sequential` or `ket::utility::policy::parallel<N>`, where `N` is an unsigned integer type.
Note that you can omit to specify `parallel_policy`.
In this case, `ket::utility::policy::sequential` version is called.

The types of variables `target_qubit*` and `control_qubit*` are `ket::qubit<S,B>` and a corresponding compile-time or runtime control type, respectively, where `S` is an unsigned integer type for indexing the element of the state vector and `B` is an unsigned integer type for various bit operations[^1].

[^1]: Usually it is nice to select `std::uint64_t` for `S` and `unsigned int` for `B`.

### State vector

In the above examples, `state` and the pair of iterators `first` and `last` are the state vectors.
The iterators must satisfy [*LegacyRandomAccessIterator*](https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator) whose `value_type` is a type of complex numbers[^2].
Meanwhile, it must be valid to call `std::begin(state)`, `std::end(state)`, and `boost::size(state)` in [Boost.Range](https://www.boost.org/libs/range/doc/html/range/concepts/random_access_range.html), which means the type of `state` must satisfy [*RandomAccessRange*](https://www.boost.org/libs/range/doc/html/range/concepts/random_access_range.html).
Moreover, the size of state vector must be the power-of-two.

[^2]: "A type of complex numbers" means (1) [all non-member functions](https://en.cppreference.com/w/cpp/numeric/complex#Non-member_functions) defined for [`std::complex<T>`](https://en.cppreference.com/w/cpp/numeric/complex) can be called via argument-dependent lookup (ADL), e.g., `using std::real; real(z);`, and (2) the real and imaginary parts are contiguously placed in this order.

For example, the following `vec*` variables satisfy our requirements:
```c++
using complex_type = std::complex<double>;
std::vector<complex_type> vec1(256); // std::vector
std::deque<complex_type> vec2(256); // std::deque
std::array<complex_type, 256> vec3; // std::array
complex_type vec4[256]; // static raw array

// dynamic raw array by using new
complex_type* data5 = new complex_type[256];
boost::iterator_range<complex_type*> vec5{data5, data5 + 256};

// dynamic raw array by using malloc
complex_type* data6 = static_cast<complex_type*>(std::malloc(sizeof(complex_type) * 256));
boost::iterator_range<complex_type*> vec6{data6, data6 + 256};
```

## MPI version

### Gate functions

MPI versions of gate functions become more complicated.
For example,
```c++
ket::mpi::gate::hadamard(
  mpi_policy, parallel_policy,
  state, permutation, buffer, communicator, environment,
  target_qubit, control_qubit1, control_qubit2);

ket::mpi::gate::exponential_pauli_x(
  mpi_policy, parallel_policy,
  state, permutation, buffer, communicator, environment,
  phase, target_qubit1, target_qubit2, control_qubit);
```

Here `mpi_policy` is a policy to specify if we use a conventional MPI-parallelization method or our novel memory-efficient method.
The type of `mpi_policy` must be either `ket::mpi::utility::policy::simple_mpi` (conventional method) or `ket::mpi::utility::policy::unit_mpi<S,B,N>` (novel method).
Note that you can omit to specify `mpi_policy`.
In this case, `ket::mpi::utility::policy::simple_mpi` version is called.

The type of `permutation` must be `ket::mpi::qubit_permutation<S,B>`.
This is the class of permutation matrix.

The type of `buffer` must be `std::vector<C>`, where `C` is `value_type` of `state`, that is a complex number type.
Before calling gate functions, you don't have to be `buffer.size()` is greater than 0 because `buffer` is automatically reallocated if needed.
```c++
std::vector<complex_type> buffer;
assert(buffer.size() == 0u);
// OK
ket::mpi::gate::hadamard(
  mpi_policy, parallel_policy,
  state, permutation, buffer, communicator, environment,
  target_qubit, control_qubit1, control_qubit2);
```

The types of `communicator` and `environment` are `yampi::communicator` and `yampi::environment`, respectively.
These are components of a thin-wrapper library of MPI, [yampi](https://github.com/naoki-yoshioka/yampi).

### Control values and permutation

Current variadic MPI gate interfaces use the same targets-before-controls ordering described above.
They additionally accept runtime-control descriptors mixed with compile-time controls in variadic calls.
`ket::mpi::gate::runtime::ranges` accepts separate target/control ranges, and the corresponding `ket::mpi::gate::runtime` overloads accept qubit iterators; the local state itself remains a range.
Both `simple_mpi` and `unit_mpi`, page and nonpage states, and explicit-datatype overloads support control values.

`permutation[c]`, `permutation.at(c)`, and the corresponding inverse-permutation operations preserve the control's required value and descriptor type.
Use `ket::mpi::remove_control(pc)` for the permutated wire and `ket::mpi::control_value(pc)` for the required value of a permutated control.
Wire interchange changes the permutation, not the control condition.
The following function demonstrates both interfaces; the caller must initialize MPI and provide a suitable distributed state:

<!-- control-example: mpi -->
```c++
#include <complex>
#include <vector>
#include <ket/mpi/gate/hadamard.hpp>
#include <ket/runtime_control.hpp>

void apply_controlled_h(
  std::vector<std::complex<double>>& state,
  ket::mpi::qubit_permutation<>& permutation,
  std::vector<std::complex<double>>& buffer,
  yampi::communicator const& communicator, yampi::environment const& environment)
{
  using Q = ket::qubit<>;
  using namespace ket::literals;
  ket::mpi::gate::hadamard(
    state, permutation, buffer, communicator, environment, 0_q, 1_ncq, 2_pcq);
  std::vector<ket::runtime_control<Q>> controls{
    ket::make_runtime_control(1_ncq), ket::make_runtime_control(2_pcq)};
  ket::mpi::gate::runtime::ranges::hadamard(
    ket::mpi::utility::policy::make_simple_mpi(),
    ket::utility::policy::make_sequential(),
    state, permutation, buffer, communicator, environment, 0_q, controls);
}
```

Existing compile-time positive-control calls retain their optimized dispatch.
Negative and runtime descriptors in variadic calls use the shared value-aware range kernels.
Positive-only diagonal fast paths are not used for negative or runtime control types.
These paths can interchange control wires into local storage, so they may communicate more and require more local qubit capacity than positive-only rank/unit-filtering optimizations.
The existing MPI local-capacity preconditions still apply; a very wide controlled gate is not guaranteed to fit simply because some controls initially belong to other ranks.

In the generic MPI gate framework, a zero control is localized before positive-only unit/rank filters could discard its condition.
Fused callbacks must still interpret their operands and control values according to the generic gate interface; the framework does not infer a callback's gate operation.
Page kernels fall back to logical state iterators when there are too few nonpage wires for their temporary mapping.
Internal positive-only page helpers and deprecated legacy overloads are not new value-aware interfaces; use the current `ket::mpi::gate` entry points.

The existing MPI permutated-control literal `_pcq` is in `ket::mpi::literals`, whereas the positive-control synonym `_pcq` is in `ket::literals`.
Avoid importing both literal namespaces unqualified in the same scope; `ket::mpi::make_permutated(3_pcq)` is an explicit alternative when only `ket::literals` is imported.

### State vector

The type of `state` must satisfy [*RandomAccessRange*](https://www.boost.org/libs/range/doc/html/range/concepts/random_access_range.html).
Note that iterator versions of gate functions are not supported in `ket::mpi::gate`.
The size of state vector must be the power-of-two **only if `mpi_policy`'s type is `ket::mpi::utility::policy::simple_mpi`**.
Meanwhile, it doesn't have to be the power-of-two in the case of `ket::mpi::utility::policy::unit_mpi<S,B,N>`.
See "Unit method" for more details.

You can use `ket::mpi::state<C>` for state vector.
This class supports "page" to omit copying operation of data of MPI buffer to state vector.
See "Page method" for more details.

### Unit method

### Page method

## Control-value regression tests

Run these commands from the repository root with Python 3 and a C++14-capable compiler (C++17 support is needed for `--std c++17`):

```sh
python3 ket/test/control_types.py
python3 ket/test/control_types.py --std c++17 --ubsan
python3 ket/test/control_documentation.py
python3 ket/test/control_gate_numerical.py
python3 ket/test/fused_control_gate_numerical.py
```

The numerical drivers also accept `--std c++17`, `--bit-masks`, `--ubsan`, and `--group` to select a smaller gate group.
The type suite also accepts `--std c++17` and `--ubsan`; it covers narrow/wide control I/O, unchanged destinations on extraction failure, and diagnostic formatting with logging enabled and disabled.
The type and numerical suites cover invalid API uses, control values, positional comparisons, and ordinary/fused gate results.

MPI tests additionally require an MPI C++ compiler, an MPI launcher, and YAMPI headers (default: `../yampi/include`; override with `--yampi`):

```sh
python3 ket/test/mpi/control_gate_numerical.py --regressions
python3 ket/test/control_documentation.py --mpi
python3 ket/test/mpi/control_gate_numerical.py --variant variadic --basis 128
python3 ket/test/mpi/control_gate_numerical.py --variant iterators --unit --datatype --ranks 4
python3 ket/test/mpi/control_gate_numerical.py --variant homogeneous --diagonal --bit-masks --ubsan
python3 ket/test/mpi/control_gate_numerical.py --print-log --group 1
```

These exercise local/page/rank wire layouts, permutation preservation, both control values, and mixed controls.
`--regressions` also runs existing positive-control cases and generic/fused negative-control cases, including runtime descriptors with cache-aware dispatch.
`--basis 128` adds all computational-basis inputs for the two-rank test layout, in addition to a complex superposition.
Other useful options include `--variant runtime-variadic`, `--aliases` for controlled-gate aliases in variadic mode, and `--page-qubits 4` to exercise wide-page fallbacks.
Ket stream and diagnostic formatting is independent of QCX and OpenQASM conversion; their existing negative-control support is unchanged. Qudit simulation remains separate work.

