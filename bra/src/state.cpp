#include <boost/config.hpp>

#include <vector>
#ifdef BRA_NO_MPI
# ifndef BOOST_NO_CXX11_HDR_CHRONO
#   include <chrono>
# else
#   define BOOST_CHRONO_HEADER_ONLY
#   include <boost/chrono/chrono.hpp>
# endif
# include <memory>
#endif

#include <mpi.h>

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>

#ifndef BRA_NO_MPI
# include <yampi/basic_datatype_of.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/wall_clock.hpp>
#endif // BRA_NO_MPI

#include <bra/state.hpp>

#ifndef BRA_NO_MPI
# define BRA_clock yampi::wall_clock
#else
# ifndef BOOST_NO_CXX11_HDR_CHRONO
#   define BRA_clock std::chrono::system_clock
# else
#   define BRA_clock boost::chrono::system_clock
# endif
#endif


namespace bra
{
#ifndef BRA_NO_MPI
  state::state(
    bit_integer_type const total_num_qubits,
    seed_type const seed,
    yampi::communicator const communicator,
    yampi::environment const& environment)
    : total_num_qubits_(total_num_qubits),
      last_outcomes_(total_num_qubits, KET_GATE_OUTCOME_VALUE(unspecified)),
      maybe_expectation_values_(),
      measured_value_(),
      generated_events_(),
      random_number_generator_(seed),
      permutation_(static_cast<permutation_type::size_type>(total_num_qubits)),
      buffer_(),
      state_integer_datatype_(yampi::basic_datatype_of<state_integer_type>::call()),
      real_datatype_(yampi::basic_datatype_of<real_type>::call()),
      derived_real_pair_datatype_(real_datatype_, 2, environment),
      real_pair_datatype_(derived_real_pair_datatype_.datatype()),
# if MPI_VERSION >= 3
      complex_datatype_(yampi::basic_datatype_of<complex_type>::call()),
# else
      complex_datatype_(derived_real_pair_datatype_.datatype()),
# endif
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
      generated_events_(),
      random_number_generator_(seed),
      permutation_(
        boost::begin(initial_permutation), boost::end(initial_permutation)),
      buffer_(),
      state_integer_datatype_(yampi::basic_datatype_of<state_integer_type>::call()),
      real_datatype_(yampi::basic_datatype_of<real_type>::call()),
      derived_real_pair_datatype_(real_datatype_, 2, environment),
      real_pair_datatype_(derived_real_pair_datatype_.datatype()),
# if MPI_VERSION >= 3
      complex_datatype_(yampi::basic_datatype_of<complex_type>::call()),
# else
      complex_datatype_(derived_real_pair_datatype_.datatype()),
# endif
      communicator_(communicator),
      environment_(environment),
      finish_times_and_processes_()
  { finish_times_and_processes_.reserve(2u); }
#else // BRA_NO_MPI
  state::state(bit_integer_type const total_num_qubits, seed_type const seed)
    : total_num_qubits_(total_num_qubits),
      last_outcomes_(total_num_qubits, KET_GATE_OUTCOME_VALUE(unspecified)),
      maybe_expectation_values_(),
      measured_value_(),
      generated_events_(),
      random_number_generator_(seed),
      finish_times_and_processes_()
  { finish_times_and_processes_.reserve(2u); }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
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
        BRA_clock::now(environment_), BRA_FINISHED_PROCESS_VALUE(operations)));

    do_expectation_values(root);
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), BRA_FINISHED_PROCESS_VALUE(begin_measurement)));

    return *this;
  }

  ::bra::state& state::generate_events(yampi::rank const root, int const num_events, int const seed)
  {
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), BRA_FINISHED_PROCESS_VALUE(operations)));

    do_generate_events(root, num_events, seed);
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), BRA_FINISHED_PROCESS_VALUE(generate_events)));

    return *this;
  }

  ::bra::state& state::exit(yampi::rank const root)
  {
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), BRA_FINISHED_PROCESS_VALUE(operations)));

    do_measure(root);
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), BRA_FINISHED_PROCESS_VALUE(ket_measure)));

    return *this;
  }
#else // BRA_NO_MPI
  ::bra::state& state::projective_measurement(qubit_type const qubit)
  {
    last_outcomes_[static_cast<bit_integer_type>(qubit)]
      = do_projective_measurement(qubit);
    return *this;
  }

  ::bra::state& state::measurement()
  {
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), BRA_FINISHED_PROCESS_VALUE(operations)));

    do_expectation_values();
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), BRA_FINISHED_PROCESS_VALUE(begin_measurement)));

    return *this;
  }

  ::bra::state& state::generate_events(int const num_events, int const seed)
  {
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), BRA_FINISHED_PROCESS_VALUE(operations)));

    do_generate_events(num_events, seed);
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), BRA_FINISHED_PROCESS_VALUE(generate_events)));

    return *this;
  }

  ::bra::state& state::exit()
  {
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), BRA_FINISHED_PROCESS_VALUE(operations)));

    do_measure();
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), BRA_FINISHED_PROCESS_VALUE(ket_measure)));

    return *this;
  }
#endif // BRA_NO_MPI
}


#undef BRA_clock

