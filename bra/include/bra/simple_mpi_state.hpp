#ifndef BRA_SIMPLE_MPI_STATE_HPP
# define BRA_SIMPLE_MPI_STATE_HPP

# ifndef BRA_NO_MPI
#   include <vector>

#   include <ket/mpi/utility/simple_mpi.hpp>

#   include <yampi/communicator.hpp>
#   include <yampi/intercommunicator.hpp>
#   include <yampi/environment.hpp>

#   include <bra/types.hpp>
#   include <bra/nonpage_mpi_state.hpp>


namespace bra
{
  class simple_mpi_state final
    : public ::bra::nonpage_mpi_state<ket::mpi::utility::policy::simple_mpi>
  {
   public:
    using mpi_policy_type = ket::mpi::utility::policy::simple_mpi;

    simple_mpi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      unsigned int const total_num_qubits,
      unsigned int const num_threads_per_process,
      ::bra::state::seed_type const seed,
      bool const is_depolarizing_channel,
      ::bra::real_type const depolarizing_px,
      ::bra::real_type const depolarizing_py,
      ::bra::real_type const depolarizing_pz,
      bool const uses_depolarizing_seed,
      ::bra::state::seed_type const depolarizing_seed,
#   ifdef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      unsigned int const num_elements_in_buffer,
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      yampi::communicator const& circuit_communicator,
      yampi::communicator const& intercircuit_communicator,
      int const circuit_index,
      std::vector<yampi::intercommunicator> const& intercommunicators,
      yampi::environment const& environment);

    simple_mpi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      std::vector<permutated_qubit_type> const& initial_permutation,
      unsigned int const num_threads_per_process,
      ::bra::state::seed_type const seed,
      bool const is_depolarizing_channel,
      ::bra::real_type const depolarizing_px,
      ::bra::real_type const depolarizing_py,
      ::bra::real_type const depolarizing_pz,
      bool const uses_depolarizing_seed,
      ::bra::state::seed_type const depolarizing_seed,
#   ifdef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      unsigned int const num_elements_in_buffer,
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      yampi::communicator const& circuit_communicator,
      yampi::communicator const& intercircuit_communicator,
      int const circuit_index,
      std::vector<yampi::intercommunicator> const& intercommunicators,
      yampi::environment const& environment);

    ~simple_mpi_state() = default;
    simple_mpi_state(simple_mpi_state const&) = default;
    simple_mpi_state& operator=(simple_mpi_state const&) = default;
    simple_mpi_state(simple_mpi_state&&) = default;
    simple_mpi_state& operator=(simple_mpi_state&&) = default;
  }; // class simple_mpi_state
} // namespace bra


# endif // BRA_NO_MPI

#endif // BRA_SIMPLE_MPI_STATE_HPP
