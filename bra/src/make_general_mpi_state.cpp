#ifndef BRA_NO_MPI
# include <vector>
# include <memory>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <bra/make_general_mpi_state.hpp>
# include <bra/state.hpp>
# include <bra/general_mpi_state.hpp>
# include <bra/general_mpi_1page_state.hpp>
# include <bra/general_mpi_2page_state.hpp>
# include <bra/general_mpi_3page_state.hpp>
# include <bra/unsupported_num_pages_error.hpp>


namespace bra
{
  std::unique_ptr< ::bra::state > make_general_mpi_state(
    unsigned int const num_page_qubits,
    ::bra::state::state_integer_type const initial_integer,
    ::bra::state::bit_integer_type const num_local_qubits,
    ::bra::state::bit_integer_type const total_num_qubits,
    unsigned int const num_threads_per_process,
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
        new ::bra::general_mpi_1page_state{
          initial_integer, num_local_qubits, total_num_qubits,
          num_threads_per_process, seed, communicator, environment}};

     case 2u:
      return std::unique_ptr< ::bra::state >{
        new ::bra::general_mpi_2page_state{
          initial_integer, num_local_qubits, total_num_qubits,
          num_threads_per_process, seed, communicator, environment}};

     case 3u:
      return std::unique_ptr< ::bra::state >{
        new ::bra::general_mpi_3page_state{
          initial_integer, num_local_qubits, total_num_qubits,
          num_threads_per_process, seed, communicator, environment}};

     default:
      throw ::bra::unsupported_num_pages_error{num_page_qubits};
    }

    return std::unique_ptr< ::bra::state >{
      new ::bra::general_mpi_state{
        initial_integer, num_local_qubits, total_num_qubits,
        num_threads_per_process, seed, communicator, environment}};
  }

  std::unique_ptr< ::bra::state > make_general_mpi_state(
    unsigned int const num_page_qubits,
    ::bra::state::state_integer_type const initial_integer,
    ::bra::state::bit_integer_type const num_local_qubits,
    std::vector< ::bra::state::qubit_type > const& initial_permutation,
    unsigned int const num_threads_per_process,
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
        new ::bra::general_mpi_1page_state{
          initial_integer, num_local_qubits, initial_permutation,
          num_threads_per_process, seed, communicator, environment}};

     case 2u:
      return std::unique_ptr< ::bra::state >{
        new ::bra::general_mpi_2page_state{
          initial_integer, num_local_qubits, initial_permutation,
          num_threads_per_process, seed, communicator, environment}};

     case 3u:
      return std::unique_ptr< ::bra::state >{
        new ::bra::general_mpi_3page_state{
          initial_integer, num_local_qubits, initial_permutation,
          num_threads_per_process, seed, communicator, environment}};

     default:
      throw ::bra::unsupported_num_pages_error{num_page_qubits};
    }

    return std::unique_ptr< ::bra::state >{
      new ::bra::general_mpi_state{
        initial_integer, num_local_qubits, initial_permutation,
        num_threads_per_process, seed, communicator, environment}};
  }
} // namespace bra


#endif // BRA_NO_MPI
