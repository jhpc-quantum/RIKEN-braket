# RIKEN-braket

## Introduction

**RIKEN-braket** is a tool for simulations of quantum gates on (classical) computers.
It contains an interpreter of "quantum assembler" *bra* and a C++ template library *ket*.
Documents of [*bra*](docs/bra.md) and [*ket*](docs/ket.md) are on `docs/` directory.

## Getting Started

**RIKEN-braket** requires a C++14 compliant compiler and [Boost C++ library](https://www.boost.org/).
Any [MPI](https://www.mpi-forum.org/) libaries are also required if you would like to use **RIKEN-braket** in massively parallel supercomputers.

You can retrieve the current status of **RIKEN-braket** by cloning the repository:

```bash
$ git clone --recursive https://github.com/jhpc-quantum/RIKEN-braket.git
```

## Using *bra*

On the `bra/` directory, you can compile *bra* via:

```
$ cd RIKEN-braket/bra
$ make nompi
```

The binary file `bra` is present on `bin/` directory.

If you would like to compile MPI-supported bra, then just make without any options:

```
$ make
```

The files `qcx/hadamards08.qcx` and `qcx/adder6x2.qcx` are sample "quantum assembler" codes.
You can test those assembler codes via:

```
$ ./bin/bra -f qcx/hadamards08.qcx 1> stdout 2> stderr # 1.
$ mpiexec -np 4 ./bin/bra --mode simple --file qcx/hadamards08.qcx 1> stdout 2> stderr # 2.
$ mpiexec -np 6 ./bin/bra --mode unit --file qcx/hadamards08.qcx --unit-qubits 3 --unit-processes 3 1> stdout 2> stderr # 3.
```

Each command corresponds to:

1. In the case of non-MPI environment, operate 8 Hadamard gates to qubits.
2. In the case of MPI environment, operate 8 Hadamard gates to qubits that includes 2 global qubits and 6 local qubits.
3. In the case of MPI environment, operate 8 Hadamard gates to qubits that includes 1 global qubit, 3 unit qubits and 4 local qubits.

The results are output to `stdout`, and the detailed log is written to `stderr`.

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

  yampi::environment environment{argc, argv, yampi::thread_support::funneled};
  auto communicator = yampi::communicator{yampi::tags::world_communicator};
  auto const rank = communicator.rank(environment);
  auto const num_gqubits = ket::utility::integer_log2<bit_integer_type>(communicator.size(environment));
  auto const num_qubits = bit_integer_type{12u};
  auto const num_lqubits = num_qubits - num_gqubits;
  auto const initial_state_value = state_integer_type{0};

  auto permutation = ket::mpi::qubit_permutation<state_integer_type, bit_integer_type>{num_qubits};
  auto local_state = ket::mpi::state<complex_type, false>{num_lqubits, initial_state_value, permutation, communicator, environment};
  auto buffer = std::vector<complex_type>{};

  using qubit_type = ket::qubit<state_integer_type, bit_integer_type>;
  auto const last_qubit = ket::make_qubit<state_integer_type>(num_qubits);
  using namespace ket::literals::qubit_literals;
  for (auto qubit = 0_q; qubit < last_qubit; ++qubit)
    ket::mpi::gate::hadamard(local_state, qubit, permutation, buffer, communicator, environment);
}
```

```bash
$ mpiCC -I${HOME}/RIKEN-braket/ket/include -I${HOME}/RIKEN-braket/yampi/include -DKET_PRINT_LOG -DNDEBUG -O3 test.cpp
```

