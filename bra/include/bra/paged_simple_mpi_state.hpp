#ifndef BRA_PAGED_SIMPLE_MPI_STATE_HPP
# define BRA_PAGED_SIMPLE_MPI_STATE_HPP

# ifndef BRA_NO_MPI
#   include <vector>

#   include <ket/mpi/utility/simple_mpi.hpp>

#   include <yampi/communicator.hpp>
#   include <yampi/environment.hpp>
#   include <yampi/intercommunicator.hpp>

#   include <bra/types.hpp>
#   include <bra/paged_mpi_state.hpp>

namespace bra
{
  class paged_simple_mpi_state final
    : public ::bra::paged_mpi_state<ket::mpi::utility::policy::simple_mpi>
  {
   public:
    using mpi_policy_type = ket::mpi::utility::policy::simple_mpi;

    paged_simple_mpi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      unsigned int const total_num_qubits,
      unsigned int const num_page_qubits,
      unsigned int const num_threads_per_process,
      ::bra::state::seed_type const seed,
      bool const is_depolarizing_channel,
      ::bra::real_type const depolarizing_px,
      ::bra::real_type const depolarizing_py,
      ::bra::real_type const depolarizing_pz,
      bool const uses_depolarizing_seed,
      ::bra::state::seed_type const depolarizing_seed,
      yampi::communicator const& circuit_communicator,
      yampi::communicator const& intercircuit_communicator,
      int const circuit_index,
      std::vector<yampi::intercommunicator> const& intercommunicators,
      yampi::environment const& environment);

    paged_simple_mpi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      std::vector<permutated_qubit_type> const& initial_permutation,
      unsigned int const num_page_qubits,
      unsigned int const num_threads_per_process,
      ::bra::state::seed_type const seed,
      bool const is_depolarizing_channel,
      ::bra::real_type const depolarizing_px,
      ::bra::real_type const depolarizing_py,
      ::bra::real_type const depolarizing_pz,
      bool const uses_depolarizing_seed,
      ::bra::state::seed_type const depolarizing_seed,
      yampi::communicator const& circuit_communicator,
      yampi::communicator const& intercircuit_communicator,
      int const circuit_index,
      std::vector<yampi::intercommunicator> const& intercommunicators,
      yampi::environment const& environment);

    ~paged_simple_mpi_state() = default;
    paged_simple_mpi_state(paged_simple_mpi_state const&) = default;
    paged_simple_mpi_state& operator=(paged_simple_mpi_state const&) = default;
    paged_simple_mpi_state(paged_simple_mpi_state&&) = default;
    paged_simple_mpi_state& operator=(paged_simple_mpi_state&&) = default;
  }; // class paged_simple_mpi_state
} // namespace bra

# endif // BRA_NO_MPI

#endif // BRA_PAGED_SIMPLE_MPI_STATE_HPP
