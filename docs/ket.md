# ket

## Introduction

*ket* is a C++ template library to perform state-vector simulation of quantum computers.

## Requirements

*ket* requires a C++11 compliant compiler and the [Boost C++ library](https://www.boost.org/).
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

* `ket/include/ket/`: `ket::qubit<S,B>` and `ket::control<Q>` classes to represent indices of target and control qubits
* `ket/include/ket/gate/`: non-MPI versions of quantum gate functions
* `ket/include/ket/utility`: some utility functions/classes, especially policy class `ket::utility::policy::sequential` to perform non-parallelized gate operation (similar to C++17's `std::sequenced_policy`)
* `ket/include/ket/utility/parallel`: policy class `ket::utility::policy::parallel<N>` to perform multi-threading gate operation (similar to C++17's `std::parallel_policy<N>`)
* `ket/include/ket/mpi/`: MPI-related directory, which includes "page"-aware state-vector class `ket::mpi::state<C>`, permutation matrix class `ket::mpi::qubit_permutation<S,B>`, and `ket::mpi::permutated<Q>` class to represent indices of qubits after permutation
* `ket/include/ket/mpi/gate`: MPI versions of quantum gate functions
* `ket/include/ket/mpi/utility/`: some utility function/classes for MPI-related operations, especially policy classes `ket::mpi::utility::policy::simple_mpi` and `ket::mpi::utility::policy::unit_mpi<S,B,N>`

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

The types of variables `target_qubit*` and `control_qubit*` are `ket::qubit<S,B>` and `ket::control<ket::qubit<S,B>>`, respectively, where `S` is an unsigned integer type for indexing the element of the state vector and `B` is an unsigned integer type for various bit operations[^1].

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

