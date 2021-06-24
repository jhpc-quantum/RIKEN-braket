#ifndef BRA_NO_MPI
# include <vector>
# include <memory>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <bra/make_unit_mpi_state.hpp>
# include <bra/state.hpp>
# include <bra/unit_mpi_state.hpp>
# include <bra/unit_mpi_1page_state.hpp>
# include <bra/unit_mpi_2page_state.hpp>
# include <bra/unit_mpi_3page_state.hpp>
# include <bra/unsupported_num_pages_error.hpp>


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
    yampi::communicator const& communicator,
    yampi::environment const& environment)
  {
    switch (num_page_qubits)
    {
     case 0u:
      break;

     case 1u:
      return std::unique_ptr< ::bra::state >{
        new ::bra::unit_mpi_1page_state{
          initial_integer, num_local_qubits, num_unit_qubits, total_num_qubits,
          num_threads_per_process, num_processes_per_unit, seed, communicator, environment}};

     case 2u:
      return std::unique_ptr< ::bra::state >{
        new ::bra::unit_mpi_2page_state{
          initial_integer, num_local_qubits, num_unit_qubits, total_num_qubits,
          num_threads_per_process, num_processes_per_unit, seed, communicator, environment}};

     case 3u:
      return std::unique_ptr< ::bra::state >{
        new ::bra::unit_mpi_3page_state{
          initial_integer, num_local_qubits, num_unit_qubits, total_num_qubits,
          num_threads_per_process, num_processes_per_unit, seed, communicator, environment}};

     default:
      throw ::bra::unsupported_num_pages_error{num_page_qubits};
    }

    return std::unique_ptr< ::bra::state >{
      new ::bra::unit_mpi_state{
        initial_integer, num_local_qubits, num_unit_qubits, total_num_qubits,
        num_threads_per_process, num_processes_per_unit, seed, communicator, environment}};
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
    yampi::communicator const& communicator,
    yampi::environment const& environment)
  {
    switch (num_page_qubits)
    {
     case 0u:
      break;

     case 1u:
      return std::unique_ptr< ::bra::state >{
        new ::bra::unit_mpi_1page_state{
          initial_integer, num_local_qubits, num_unit_qubits, initial_permutation,
          num_threads_per_process, num_processes_per_unit, seed, communicator, environment}};

     case 2u:
      return std::unique_ptr< ::bra::state >{
        new ::bra::unit_mpi_2page_state{
          initial_integer, num_local_qubits, num_unit_qubits, initial_permutation,
          num_threads_per_process, num_processes_per_unit, seed, communicator, environment}};

     case 3u:
      return std::unique_ptr< ::bra::state >{
        new ::bra::unit_mpi_3page_state{
          initial_integer, num_local_qubits, num_unit_qubits, initial_permutation,
          num_threads_per_process, num_processes_per_unit, seed, communicator, environment}};

     default:
      throw ::bra::unsupported_num_pages_error{num_page_qubits};
    }

    return std::unique_ptr< ::bra::state >{
      new ::bra::unit_mpi_state{
        initial_integer, num_local_qubits, num_unit_qubits, initial_permutation,
        num_threads_per_process, num_processes_per_unit, seed, communicator, environment}};
  }
} // namespace bra


#endif // BRA_NO_MPI
