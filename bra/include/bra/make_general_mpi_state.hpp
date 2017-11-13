#ifndef BRA_MAKE_GENERAL_MPI_STATE_HPP
# define BRA_MAKE_GENERAL_MPI_STATE_HPP

# include <boost/config.hpp>

# include <string>
# include <stdexcept>
# include <vector>

# include <boost/move/unique_ptr.hpp>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <bra/state.hpp>



namespace bra
{
  class unsupported_num_pages_error
    : public std::logic_error
  {
   public:
    explicit unsupported_num_pages_error(unsigned int const num_pages)
      : std::logic_error(generate_what_string(num_pages).c_str())
    { }

   private:
    std::string generate_what_string(unsigned int const num_pages);
  };

  boost::movelib::unique_ptr< ::bra::state > make_general_mpi_state(
    unsigned int const num_pages,
    ::bra::state::state_integer_type const initial_integer,
    ::bra::state::bit_integer_type const num_local_qubits,
    ::bra::state::bit_integer_type const total_num_qubits,
    ::bra::state::seed_type const seed,
    yampi::communicator const communicator,
    yampi::environment const& environment);

  boost::movelib::unique_ptr< ::bra::state > make_general_mpi_state(
    unsigned int const num_pages,
    ::bra::state::state_integer_type const initial_integer,
    ::bra::state::bit_integer_type const num_local_qubits,
    std::vector< ::bra::state::qubit_type > const& initial_permutation,
    ::bra::state::seed_type const seed,
    yampi::communicator const communicator,
    yampi::environment const& environment);
}


#endif

