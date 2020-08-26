#ifndef KET_MPI_GATE_PAULI_X_HPP
# define KET_MPI_GATE_PAULI_X_HPP

# include <boost/config.hpp>

# include <ket/gate/pauli_x.hpp>
# include <ket/mpi/gate/page/pauli_x.hpp>

# include <ket/mpi/gate/detail/before_generate_single_qubit_gate.hpp>
# include <ket/mpi/gate/detail/generate_single_qubit_gate.hpp>

KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE(pauli_x, X)

# include <ket/mpi/gate/detail/after_generate_single_qubit_gate.hpp>

#endif // KET_MPI_GATE_PAULI_X_HPP
