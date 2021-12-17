#ifndef KET_MPI_GATE_DETAIL_PAULI_Z_STANDARD_HPP
# define KET_MPI_GATE_DETAIL_PAULI_Z_STANDARD_HPP

# include <boost/config.hpp>

# include <ket/gate/pauli_z.hpp>
# include <ket/mpi/gate/page/pauli_z.hpp>

# include <ket/mpi/gate/detail/before_generate_single_qubit_gate.hpp>
# include <ket/mpi/gate/detail/generate_single_qubit_gate.hpp>

KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE(pauli_z, Z)

# include <ket/mpi/gate/detail/after_generate_single_qubit_gate.hpp>

#endif // KET_MPI_GATE_DETAIL_PAULI_Z_STANDARD_HPP
