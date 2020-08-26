# braket

## Introduction

**braket** is a tool for simulations of quantum gates on (classical) computers.
It contains an interpreter of "quantum assembler" *bra* and a C++ template library *ket*.

## Getting Started

**braket** requires a MPI2-supported C++11 compliant compiler.
If you do not have any MPI libraries, get it by using a package manager of your \*nix environment.

You can retrieve the current status of **braket** by cloning the repository:

```bash
git clone --recursive https://github.com/naoki-yoshioka/braket.git
```

## Using *bra*

On the `bra/` directory, you can compile *bra* via:

```
make
```

The binary file `bra` is present on `bra/bin/` directory.

If you would like to compile it by specified C++ version, type like:

```
make release11
make release14
make release17
```

The files `bra/qcx/hadamards08.qcx` and `bra/qcx/adder6x2.qcx` are sample "quantum assembler" codes.
You can test those assembler codes via:

```
mpiexec -np 2 ./bin/bra qcx/hadamards08.qcx 1> stdout 2> stderr
```

The results are output to `stdout`.

## Using *ket*

```cpp:test.cpp
#include <iostream>
#include <complex>
#include <vector>

#include <ket/qubit.hpp>
#include <ket/utility/integer_log2.hpp>
#include <ket/mpi/state.hpp>
#include <ket/mpi/qubit_permutation.hpp>
#include <ket/mpi/gate/hadamard.hpp>

#include <yampi/environment.hpp>
#include <yampi/communicator.hpp>

int main(int argc, char* argv[])
{
  std::ios::sync_with_stdio(false);

  using state_integer_type = std::uint64_t;
  using bit_integer_type = unsigned int;
  using complex_type = std::complex<double>;

  yampi::environment environment{argc, argv};
  yampi::communicator communicator{yampi::world_communicator()};
  auto const rank = communicator.rank(environment);
  auto const num_gqubits = ket::utility::integer_log2<bit_integer_type>(communicator.size(environment));
  auto const num_qubits = bit_integer_type{12u};
  auto const num_lqubits = num_qubits - num_gqubits;
  auto const initial_state_value = state_integer_type{0};

  auto permutation = ket::mpi::qubit_permutation<state_integer_type, bit_integer_type>{num_qubits};
  auto local_state = ket::mpi::state<complex_type>{num_lqubits, initial_state_value, permutation, communicator, environment};
  auto buffer = std::vector<complex_type>{};

  using qubit_type = ket::qubit<state_integer_type, bit_integer_type>;
  auto const last_qubit = qubit_type{num_qubits};
  for (auto qubit = qubit_type{bit_integer_type{0}}; qubit < last_qubit; ++qubit)
    ket::mpi::gate::hadamard(local_state, qubit, permutation, buffer, communicator, environment);
}
```

```bash
mpiCC -Iket/include -Iyampi/include -DKET_PRINT_LOG -DNDEBUG -O3 test.cpp
```

