#ifndef KET_MPI_GATE_PAULI_Y_HPP
# define KET_MPI_GATE_PAULI_Y_HPP

# include <boost/config.hpp>

# include <ket/gate/pauli_y.hpp>
# include <ket/mpi/gate/page/pauli_y.hpp>

# include <ket/mpi/gate/detail/before_generate_single_qubit_gate.hpp>
# include <ket/mpi/gate/detail/generate_single_qubit_gate.hpp>

KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE(pauli_y, Y)

# include <ket/mpi/gate/detail/after_generate_single_qubit_gate.hpp>

#endif

