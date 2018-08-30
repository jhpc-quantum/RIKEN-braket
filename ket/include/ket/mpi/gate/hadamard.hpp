#ifndef KET_MPI_GATE_HADAMARD_HPP
# define KET_MPI_GATE_HADAMARD_HPP

# include <boost/config.hpp>

# include <ket/gate/hadamard.hpp>
# include <ket/mpi/gate/page/hadamard.hpp>

# include <ket/mpi/gate/detail/before_generate_single_qubit_gate.hpp>
# include <ket/mpi/gate/detail/generate_single_qubit_gate.hpp>

KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE(hadamard, Hadamard)

# include <ket/mpi/gate/detail/after_generate_single_qubit_gate.hpp>

# endif

