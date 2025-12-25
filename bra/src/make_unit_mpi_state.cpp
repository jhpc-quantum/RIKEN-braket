#ifndef BRA_NO_MPI
# include <vector>
# include <memory>

# include <yampi/communicator.hpp>
# include <yampi/intercommunicator.hpp>
# include <yampi/environment.hpp>

# include <bra/make_unit_mpi_state.hpp>
# include <bra/state.hpp>
# include <bra/unit_mpi_state.hpp>
# include <bra/paged_unit_mpi_state.hpp>


namespace bra
{
  std::unique_ptr< ::bra::state > make_unit_mpi_state(
    unsigned int const num_page_qubits,
    ::bra::state::state_integer_type const initial_integer,
    ::bra::state::bit_integer_type const num_local_qubits,
    ::bra::state::bit_integer_type const num_unit_qubits,
    ::bra::state::bit_integer_type const total_num_qubits,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
# ifdef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
    unsigned int const num_elements_in_buffer,
# endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
  {
# ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
    if (num_page_qubits == 0u)
      return std::unique_ptr< ::bra::state >{
        new ::bra::unit_mpi_state{
          initial_integer, num_local_qubits, num_unit_qubits, total_num_qubits,
          num_threads_per_process, num_processes_per_unit, seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}};
# else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
    if (num_page_qubits == 0u)
      return std::unique_ptr< ::bra::state >{
        new ::bra::unit_mpi_state{
          initial_integer, num_local_qubits, num_unit_qubits, total_num_qubits,
          num_threads_per_process, num_processes_per_unit, seed, num_elements_in_buffer, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}};
# endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS

    return std::unique_ptr< ::bra::state >{
      new ::bra::paged_unit_mpi_state{
        initial_integer, num_local_qubits, num_unit_qubits, total_num_qubits, num_page_qubits,
        num_threads_per_process, num_processes_per_unit, seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}};
  }

  std::unique_ptr< ::bra::state > make_unit_mpi_state(
    unsigned int const num_page_qubits,
    ::bra::state::state_integer_type const initial_integer,
    ::bra::state::bit_integer_type const num_local_qubits,
    ::bra::state::bit_integer_type const num_unit_qubits,
    std::vector< ::bra::state::permutated_qubit_type > const& initial_permutation,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
# ifdef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
    unsigned int const num_elements_in_buffer,
# endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
  {
# ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
    if (num_page_qubits == 0u)
      return std::unique_ptr< ::bra::state >{
        new ::bra::unit_mpi_state{
          initial_integer, num_local_qubits, num_unit_qubits, initial_permutation,
          num_threads_per_process, num_processes_per_unit, seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}};
# else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
    if (num_page_qubits == 0u)
      return std::unique_ptr< ::bra::state >{
        new ::bra::unit_mpi_state{
          initial_integer, num_local_qubits, num_unit_qubits, initial_permutation,
          num_threads_per_process, num_processes_per_unit, seed, num_elements_in_buffer, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}};
# endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS

    return std::unique_ptr< ::bra::state >{
      new ::bra::paged_unit_mpi_state{
        initial_integer, num_local_qubits, num_unit_qubits, initial_permutation, num_page_qubits,
        num_threads_per_process, num_processes_per_unit, seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment}};
  }
} // namespace bra


#endif // BRA_NO_MPI
