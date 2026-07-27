#ifndef BRA_NO_MPI

# include <bra/simple_mpi_state.hpp>

namespace bra
{
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
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
    yampi::environment const& environment)
    : ::bra::nonpage_mpi_state<mpi_policy_type>{initial_integer, num_local_qubits, num_threads_per_process, mpi_policy_type{}, circuit_communicator, environment, total_num_qubits, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
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
    yampi::environment const& environment)
    : ::bra::nonpage_mpi_state<mpi_policy_type>{initial_integer, num_local_qubits, num_threads_per_process, mpi_policy_type{}, circuit_communicator, environment, initial_permutation, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}
  { }
#   else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::nonpage_mpi_state<mpi_policy_type>{initial_integer, num_local_qubits, num_threads_per_process, mpi_policy_type{}, circuit_communicator, environment, total_num_qubits, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, num_elements_in_buffer, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::nonpage_mpi_state<mpi_policy_type>{initial_integer, num_local_qubits, num_threads_per_process, mpi_policy_type{}, circuit_communicator, environment, initial_permutation, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, num_elements_in_buffer, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}
  { }
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
# else // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
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
    yampi::environment const& environment)
    : ::bra::nonpage_mpi_state<mpi_policy_type>{initial_integer, num_local_qubits, num_threads_per_process, mpi_policy_type{}, circuit_communicator, environment, total_num_qubits, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
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
    yampi::environment const& environment)
    : ::bra::nonpage_mpi_state<mpi_policy_type>{initial_integer, num_local_qubits, num_threads_per_process, mpi_policy_type{}, circuit_communicator, environment, initial_permutation, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}
  { }
#   else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::nonpage_mpi_state<mpi_policy_type>{initial_integer, num_local_qubits, num_threads_per_process, mpi_policy_type{}, circuit_communicator, environment, total_num_qubits, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, num_elements_in_buffer, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::nonpage_mpi_state<mpi_policy_type>{initial_integer, num_local_qubits, num_threads_per_process, mpi_policy_type{}, circuit_communicator, environment, initial_permutation, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, num_elements_in_buffer, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}
  { }
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

} // namespace bra

#endif // BRA_NO_MPI
