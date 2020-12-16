#include <vector>
#include <random>
#include <iterator>
#ifdef BRA_NO_MPI
# include <chrono>
# include <memory>
#endif

#ifndef BRA_NO_MPI
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/wall_clock.hpp>
# include <yampi/predefined_datatype.hpp>
#endif // BRA_NO_MPI

#include <ket/qubit.hpp>

#include <bra/state.hpp>
#include <bra/utility/closest_floating_point_of.hpp>

#ifndef BRA_NO_MPI
# define BRA_clock yampi::wall_clock
#else
# define BRA_clock std::chrono::system_clock
#endif


namespace bra
{
#ifndef BRA_NO_MPI
  state::state(
    bit_integer_type const total_num_qubits,
    seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : total_num_qubits_{total_num_qubits},
      last_outcomes_{total_num_qubits, ket::gate::outcome::unspecified},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      random_number_generator_{seed},
      permutation_{static_cast<permutation_type::size_type>(total_num_qubits)},
      buffer_{},
      real_pair_datatype_{yampi::predefined_datatype<real_type>(), 2, environment},
      communicator_{communicator},
      environment_{environment},
      finish_times_and_processes_{}
  { finish_times_and_processes_.reserve(2u); }

  state::state(
    std::vector<qubit_type> const& initial_permutation,
    seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : total_num_qubits_{static_cast<bit_integer_type>(initial_permutation.size())},
      last_outcomes_{total_num_qubits_, ket::gate::outcome::unspecified},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      random_number_generator_{seed},
      permutation_{
        std::begin(initial_permutation), std::end(initial_permutation)},
      buffer_{},
      real_pair_datatype_{yampi::predefined_datatype<real_type>(), 2, environment},
      communicator_{communicator},
      environment_{environment},
      finish_times_and_processes_{}
  { finish_times_and_processes_.reserve(2u); }
#else // BRA_NO_MPI
  state::state(bit_integer_type const total_num_qubits, seed_type const seed)
    : total_num_qubits_{total_num_qubits},
      last_outcomes_{total_num_qubits, ket::gate::outcome::unspecified},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      random_number_generator_{seed},
      finish_times_and_processes_{}
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
        BRA_clock::now(environment_), ::bra::finished_process::operations));

    do_expectation_values(root);
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), ::bra::finished_process::begin_measurement));

    return *this;
  }

  ::bra::state& state::generate_events(yampi::rank const root, int const num_events, int const seed)
  {
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), ::bra::finished_process::operations));

    do_generate_events(root, num_events, seed);
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), ::bra::finished_process::generate_events));

    return *this;
  }

  ::bra::state& state::exit(yampi::rank const root)
  {
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), ::bra::finished_process::operations));

    do_measure(root);
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), ::bra::finished_process::ket_measure));

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
        BRA_clock::now(), ::bra::finished_process::operations));

    do_expectation_values();
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), ::bra::finished_process::begin_measurement));

    return *this;
  }

  ::bra::state& state::generate_events(int const num_events, int const seed)
  {
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), ::bra::finished_process::operations));

    do_generate_events(num_events, seed);
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), ::bra::finished_process::generate_events));

    return *this;
  }

  ::bra::state& state::exit()
  {
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), ::bra::finished_process::operations));

    do_measure();
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), ::bra::finished_process::ket_measure));

    return *this;
  }
#endif // BRA_NO_MPI

  ::bra::state& state::shor_box(bit_integer_type const num_exponent_qubits, state_integer_type const divisor, state_integer_type const base)
  {
    auto exponent_qubits = std::vector<qubit_type>(num_exponent_qubits);
    std::iota(
      std::begin(exponent_qubits), std::end(exponent_qubits),
      static_cast<qubit_type>(total_num_qubits_ - num_exponent_qubits));
    auto modular_exponentiation_qubits
      = std::vector<qubit_type>(total_num_qubits_ - num_exponent_qubits);
    std::iota(
      std::begin(modular_exponentiation_qubits), std::end(modular_exponentiation_qubits),
      qubit_type{0u});

    do_shor_box(divisor, base, exponent_qubits, modular_exponentiation_qubits);

    return *this;
  }

  ::bra::state& state::depolarizing_channel(real_type const px, real_type const py, real_type const pz, int const seed)
  {
    using floating_point_type = typename ::bra::utility::closest_floating_point_of<real_type>::type;
    auto distribution = std::uniform_real_distribution<floating_point_type>{0.0, 1.0};
    auto const last_qubit = ket::make_qubit(total_num_qubits_);
    if (seed < 0)
      for (auto qubit = ket::make_qubit(bit_integer_type{0u}); qubit < last_qubit; ++qubit)
      {
        auto const probability = static_cast<real_type>(distribution(random_number_generator_));
        if (probability < px)
          pauli_x(qubit);
        else if (probability < px + py)
          pauli_y(qubit);
        else if (probability < px + py + pz)
          pauli_z(qubit);
      }
    else
    {
      auto temporal_random_number_generator = random_number_generator_type{static_cast<seed_type>(seed)};
      for (auto qubit = ket::make_qubit(static_cast<bit_integer_type>(0u)); qubit < last_qubit; ++qubit)
      {
        auto const probability = static_cast<real_type>(distribution(temporal_random_number_generator));
        if (probability < px)
          pauli_x(qubit);
        else if (probability < px + py)
          pauli_y(qubit);
        else if (probability < px + py + pz)
          pauli_z(qubit);
      }
    }

    return *this;
  }
} // namespace bra


#undef BRA_clock
