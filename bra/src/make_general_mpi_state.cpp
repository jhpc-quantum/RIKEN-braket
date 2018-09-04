#ifndef BRA_NO_MPI
# include <boost/config.hpp>

# include <string>
# include <sstream>
# include <vector>

# include <boost/move/unique_ptr.hpp>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <bra/make_general_mpi_state.hpp>
# include <bra/state.hpp>
# include <bra/general_mpi_state.hpp>
# include <bra/general_mpi_1page_state.hpp>
# include <bra/general_mpi_2page_state.hpp>
# include <bra/general_mpi_3page_state.hpp>


namespace bra
{
  std::string unsupported_num_pages_error::generate_what_string(
    unsigned int const num_pages)
  {
    std::ostringstream output_stream("num_pages=");
    output_stream << num_pages << " is not supported";
    return output_stream.str();
  }

  boost::movelib::unique_ptr< ::bra::state > make_general_mpi_state(
    unsigned int const num_page_qubits,
    ::bra::state::state_integer_type const initial_integer,
    ::bra::state::bit_integer_type const num_local_qubits,
    ::bra::state::bit_integer_type const total_num_qubits,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
  {
    switch (num_page_qubits)
    {
     case 0u:
      break;

     case 1u:
      return boost::movelib::unique_ptr< ::bra::state >(
        new ::bra::general_mpi_1page_state(
          initial_integer, num_local_qubits, total_num_qubits, seed, communicator, environment));

     case 2u:
      return boost::movelib::unique_ptr< ::bra::state >(
        new ::bra::general_mpi_2page_state(
          initial_integer, num_local_qubits, total_num_qubits, seed, communicator, environment));

     case 3u:
      return boost::movelib::unique_ptr< ::bra::state >(
        new ::bra::general_mpi_3page_state(
          initial_integer, num_local_qubits, total_num_qubits, seed, communicator, environment));

     default:
      throw ::bra::unsupported_num_pages_error(num_page_qubits);
    }

    return boost::movelib::unique_ptr< ::bra::state >(
      new ::bra::general_mpi_state(
        initial_integer, num_local_qubits, total_num_qubits, seed, communicator, environment));
  }

  boost::movelib::unique_ptr< ::bra::state > make_general_mpi_state(
    unsigned int const num_page_qubits,
    ::bra::state::state_integer_type const initial_integer,
    ::bra::state::bit_integer_type const num_local_qubits,
    std::vector< ::bra::state::qubit_type > const& initial_permutation,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
  {
    switch (num_page_qubits)
    {
     case 0u:
      break;

     case 1u:
      return boost::movelib::unique_ptr< ::bra::state >(
        new ::bra::general_mpi_1page_state(
          initial_integer, num_local_qubits, initial_permutation,
          seed, communicator, environment));

     case 2u:
      return boost::movelib::unique_ptr< ::bra::state >(
        new ::bra::general_mpi_2page_state(
          initial_integer, num_local_qubits, initial_permutation,
          seed, communicator, environment));

     case 3u:
      return boost::movelib::unique_ptr< ::bra::state >(
        new ::bra::general_mpi_3page_state(
          initial_integer, num_local_qubits, initial_permutation,
          seed, communicator, environment));

     default:
      throw ::bra::unsupported_num_pages_error(num_page_qubits);
    }

    return boost::movelib::unique_ptr< ::bra::state >(
      new ::bra::general_mpi_state(
        initial_integer, num_local_qubits, initial_permutation, seed, communicator, environment));
  }
}


#endif // BRA_NO_MPI

