#ifndef BRA_NO_MPI
# include <vector>
# include <memory>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <bra/make_simple_mpi_state.hpp>
# include <bra/state.hpp>
# include <bra/simple_mpi_state.hpp>
# include <bra/paged_simple_mpi_state.hpp>


namespace bra
{
  std::unique_ptr< ::bra::state > make_simple_mpi_state(
    unsigned int const num_page_qubits,
    ::bra::state::state_integer_type const initial_integer,
    ::bra::state::bit_integer_type const num_local_qubits,
    ::bra::state::bit_integer_type const total_num_qubits,
    unsigned int const num_threads_per_process,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
  {
    if (num_page_qubits == 0u)
      return std::unique_ptr< ::bra::state >{
        new ::bra::simple_mpi_state{
          initial_integer, num_local_qubits, total_num_qubits,
          num_threads_per_process, seed, communicator, environment}};

    return std::unique_ptr< ::bra::state >{
      new ::bra::paged_simple_mpi_state{
        initial_integer, num_local_qubits, total_num_qubits, num_page_qubits,
        num_threads_per_process, seed, communicator, environment}};
  }

  std::unique_ptr< ::bra::state > make_simple_mpi_state(
    unsigned int const num_page_qubits,
    ::bra::state::state_integer_type const initial_integer,
    ::bra::state::bit_integer_type const num_local_qubits,
    std::vector< ::bra::state::permutated_qubit_type > const& initial_permutation,
    unsigned int const num_threads_per_process,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
  {
    if (num_page_qubits == 0u)
      return std::unique_ptr< ::bra::state >{
        new ::bra::simple_mpi_state{
          initial_integer, num_local_qubits, initial_permutation,
          num_threads_per_process, seed, communicator, environment}};

    return std::unique_ptr< ::bra::state >{
      new ::bra::paged_simple_mpi_state{
        initial_integer, num_local_qubits, initial_permutation, num_page_qubits,
        num_threads_per_process, seed, communicator, environment}};
  }
} // namespace bra


#endif // BRA_NO_MPI
