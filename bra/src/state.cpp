#include <boost/config.hpp>

#include <vector>

#include <mpi.h>

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>

#include <yampi/basic_datatype_of.hpp>
#include <yampi/communicator.hpp>
#include <yampi/environment.hpp>
#include <yampi/wall_clock.hpp>

#include <bra/state.hpp>


namespace bra
{
  state::state(
    bit_integer_type const total_num_qubits,
    seed_type const seed,
    yampi::communicator const communicator,
    yampi::environment const& environment)
    : total_num_qubits_(total_num_qubits),
      maybe_expectation_values_(),
      measured_value_(),
      random_number_generator_(seed),
      permutation_(static_cast<permutation_type::size_type>(total_num_qubits)),
      buffer_(),
      state_integer_datatype_(yampi::basic_datatype_of<state_integer_type>::call()),
      real_datatype_(yampi::basic_datatype_of<real_type>::call()),
#if MPI_VERSION >= 3
      complex_datatype_(yampi::basic_datatype_of<complex_type>::call()),
#else
      derived_complex_datatype_(real_datatype_, 2, environment),
      complex_datatype_(derived_complex_datatype_.datatype()),
#endif
      communicator_(communicator),
      environment_(environment),
      operations_finish_time_(),
      expectation_values_finish_time_(),
      measurement_finish_time_()
  { }

  state::state(
    std::vector<qubit_type> const& initial_permutation,
    seed_type const seed,
    yampi::communicator const communicator,
    yampi::environment const& environment)
    : total_num_qubits_(initial_permutation.size()),
      maybe_expectation_values_(),
      measured_value_(),
      random_number_generator_(seed),
      permutation_(
        boost::begin(initial_permutation), boost::end(initial_permutation)),
      buffer_(),
      state_integer_datatype_(yampi::basic_datatype_of<state_integer_type>::call()),
      real_datatype_(yampi::basic_datatype_of<real_type>::call()),
#if MPI_VERSION >= 3
      complex_datatype_(yampi::basic_datatype_of<complex_type>::call()),
#else
      derived_complex_datatype_(real_datatype_, 2, environment),
      complex_datatype_(derived_complex_datatype_.datatype()),
#endif
      communicator_(communicator),
      environment_(environment),
      operations_finish_time_(),
      expectation_values_finish_time_(),
      measurement_finish_time_()
  { }

  ::bra::state& state::measurement(yampi::rank const root)
  {
    operations_finish_time_ = yampi::wall_clock::now(environment_);
    do_expectation_values(root);
    expectation_values_finish_time_ = yampi::wall_clock::now(environment_);
    do_measure(root);
    measurement_finish_time_ = yampi::wall_clock::now(environment_);
    return *this;
  }
}
