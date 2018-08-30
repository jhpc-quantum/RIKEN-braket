#ifndef KET_MPI_GATE_X_ROTATION_HALF_PI_HPP
# define KET_MPI_GATE_X_ROTATION_HALF_PI_HPP

# include <boost/config.hpp>

# include <ket/gate/x_rotation_half_pi.hpp>
# include <ket/mpi/gate/page/x_rotation_half_pi.hpp>

# include <ket/mpi/gate/detail/before_generate_single_qubit_gate.hpp>
# include <ket/mpi/gate/detail/generate_single_qubit_gate.hpp>

KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE(x_rotation_half_pi, Xpi)

# include <ket/mpi/gate/detail/after_generate_single_qubit_gate.hpp>

# endif

