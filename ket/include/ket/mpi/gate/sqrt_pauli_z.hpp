#ifndef KET_MPI_GATE_SQRT_PAULI_Z_HPP
# define KET_MPI_GATE_SQRT_PAULI_Z_HPP

#ifndef KET_USE_DIAGONAL_LOOP
# include <ket/mpi/gate/detail/sqrt_pauli_z_standard.hpp>
#else // KET_USE_DIAGONAL_LOOP
# include <ket/mpi/gate/detail/sqrt_pauli_z_diagonal.hpp>
#endif // KET_USE_DIAGONAL_LOOP

# include <ket/mpi/gate/detail/sqrt_pauli_z_runtime.hpp>

#endif // KET_MPI_GATE_SQRT_PAULI_Z_HPP
