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
      last_outcomes_(total_num_qubits, KET_GATE_OUTCOME_VALUE(unspecified)),
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
      finish_times_and_processes_()
  { finish_times_and_processes_.reserve(2u); }

  state::state(
    std::vector<qubit_type> const& initial_permutation,
    seed_type const seed,
    yampi::communicator const communicator,
    yampi::environment const& environment)
    : total_num_qubits_(initial_permutation.size()),
      last_outcomes_(total_num_qubits_, KET_GATE_OUTCOME_VALUE(unspecified)),
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
      finish_times_and_processes_()
  { finish_times_and_processes_.reserve(2u); }

  ::bra::state& state::projective_measurement(qubit_type const qubit, yampi::rank const root)
  {
    last_outcomes_[static_cast<bit_integer_type>(qubit)]
      = do_projective_measurement(qubit, root);
    return *this;
  }

  ::bra::state& state::measurement(yampi::rank const root)
  {
    finish_times_and_processes_.push_back(
      std::make_pair(
        yampi::wall_clock::now(environment_), BRA_FINISHED_PROCESS_VALUE(operations)));

    do_expectation_values(root);
    finish_times_and_processes_.push_back(
      std::make_pair(
        yampi::wall_clock::now(environment_), BRA_FINISHED_PROCESS_VALUE(beign_measuremnet)));

    /*
    do_measure(root);
    finish_times_and_processes_.push_back(
      std::make_pair(
        yampi::wall_clock::now(environment_), BRA_FINISHED_PROCESS_VALUE(ket_measure)));
        */

    return *this;
  }
}
