# braket

## Introduction

**braket** is a tool for simulations of quantum gates on (classical) computers.
It contains an interpreter of "quantum assembler" *bra* and a C++ template library *ket*.

## Getting Started

**braket** requires a MPI2-supported C++03 compliant compiler.
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
make release03
make release11
make release14
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

#include <boost/cstdint.hpp>

#include <ket/qubit.hpp>
#include <ket/utility/integer_log2.hpp>
#include <ket/mpi/state.hpp>
#include <ket/mpi/qubit_permutation.hpp>
#include <ket/mpi/gate/hadamard.hpp>

#include <yampi/environment.hpp>
#include <yampi/communicator.hpp>
#include <yampi/rank.hpp>
#include <yampi/basic_datatype_of.hpp>

int main(int argc, char* argv[])
{
  std::ios::sync_with_stdio(false);

  typedef boost::uint64_t state_integer_type;
  typedef unsigned int bit_integer_type;
  typedef std::complex<double> complex_type;

  yampi::environment environment(argc, argv);
  yampi::communicator communicator = yampi::world_communicator();
  yampi::rank const rank = communicator.rank(environment);
  bit_integer_type const num_gqubits = ket::utility::integer_log2<bit_integer_type>(communicator.size(environment));
  bit_integer_type const num_qubits = 12u;
  bit_integer_type const num_lqubits = num_qubits - num_gqubits;
  state_integer_type const initial_state_value = 0;

  ket::mpi::qubit_permutation<state_integer_type, bit_integer_type> permutation(num_qubits);
  ket::mpi::state<complex_type> local_state(num_lqubits, initial_state_value, permutation, communicator, environment);
  std::vector<complex_type> buffer;

  typedef ket::qubit<state_integer_type, bit_integer_type> qubit_type;
  for (bit_integer_type bit = 0; bit < num_qubits; ++bit)
    ket::mpi::gate::hadamard(
      local_state, static_cast<qubit_type>(bit), permutation,
      buffer, yampi::basic_datatype_of<complex_type>::call(), communicator, environment);
}
```

```bash
mpiCC -Iket/include -Iyampi/include -DKET_PRINT_LOG -DNDEBUG -O3 test.cpp
```

