#ifdef BRA_NO_MPI
# include <boost/config.hpp>

# include <vector>
# ifndef BOOST_NO_CXX11_HDR_RANDOM
#   include <random>
# else
#   include <boost/random/uniform_real_distribution.hpp>
# endif

# include <boost/range/algorithm_ext/iota.hpp>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/qubit.hpp>
# include <ket/gate/hadamard.hpp>
# include <ket/gate/pauli_x.hpp>
# include <ket/gate/pauli_y.hpp>
# include <ket/gate/pauli_z.hpp>
# include <ket/gate/phase_shift.hpp>
# include <ket/gate/x_rotation_half_pi.hpp>
# include <ket/gate/y_rotation_half_pi.hpp>
# include <ket/gate/controlled_not.hpp>
# include <ket/gate/controlled_phase_shift.hpp>
# include <ket/gate/controlled_v.hpp>
# include <ket/gate/toffoli.hpp>
# include <ket/gate/projective_measurement.hpp>
# include <ket/gate/clear.hpp>
# include <ket/gate/set.hpp>
# include <ket/all_spin_expectation_values.hpp>
# include <ket/measure.hpp>
# include <ket/generate_events.hpp>
# include <ket/shor_box.hpp>

# include <bra/nompi_state.hpp>
# include <bra/state.hpp>

# ifndef BOOST_NO_CXX11_HDR_RANDOM
#   define BRA_uniform_real_distribution std::uniform_real_distribution
# else
#   define BRA_uniform_real_distribution boost::random::uniform_real_distribution
# endif


namespace bra
{
  nompi_state::nompi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const total_num_qubits,
    ::bra::state::seed_type const seed)
    : ::bra::state(total_num_qubits, seed),
      parallel_policy_(),
      data_(make_initial_data(initial_integer, total_num_qubits))
  { }

  void nompi_state::do_hadamard(qubit_type const qubit)
  { ket::gate::ranges::hadamard(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_hadamard(qubit_type const qubit)
  { ket::gate::ranges::adj_hadamard(parallel_policy_, data_, qubit); }

  void nompi_state::do_pauli_x(qubit_type const qubit)
  { ket::gate::ranges::pauli_x(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_pauli_x(qubit_type const qubit)
  { ket::gate::ranges::adj_pauli_x(parallel_policy_, data_, qubit); }

  void nompi_state::do_pauli_y(qubit_type const qubit)
  { ket::gate::ranges::pauli_y(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_pauli_y(qubit_type const qubit)
  { ket::gate::ranges::adj_pauli_y(parallel_policy_, data_, qubit); }

  void nompi_state::do_pauli_z(qubit_type const qubit)
  { ket::gate::ranges::pauli_z(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_pauli_z(qubit_type const qubit)
  { ket::gate::ranges::adj_pauli_z(parallel_policy_, data_, qubit); }

  void nompi_state::do_u1(real_type const phase, qubit_type const qubit)
  { ket::gate::ranges::phase_shift(parallel_policy_, data_, phase, qubit); }

  void nompi_state::do_adj_u1(real_type const phase, qubit_type const qubit)
  { ket::gate::ranges::adj_phase_shift(parallel_policy_, data_, phase, qubit); }

  void nompi_state::do_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  { ket::gate::ranges::phase_shift2(parallel_policy_, data_, phase1, phase2, qubit); }

  void nompi_state::do_adj_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  { ket::gate::ranges::adj_phase_shift2(parallel_policy_, data_, phase1, phase2, qubit); }

  void nompi_state::do_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  { ket::gate::ranges::phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, qubit); }

  void nompi_state::do_adj_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  { ket::gate::ranges::adj_phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, qubit); }

  void nompi_state::do_phase_shift(
    complex_type const phase_coefficient, qubit_type const qubit)
  { ket::gate::ranges::phase_shift_coeff(parallel_policy_, data_, phase_coefficient, qubit); }

  void nompi_state::do_adj_phase_shift(
    complex_type const phase_coefficient, qubit_type const qubit)
  { ket::gate::ranges::adj_phase_shift_coeff(parallel_policy_, data_, phase_coefficient, qubit); }

  void nompi_state::do_x_rotation_half_pi(qubit_type const qubit)
  { ket::gate::ranges::x_rotation_half_pi(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_x_rotation_half_pi(qubit_type const qubit)
  { ket::gate::ranges::adj_x_rotation_half_pi(parallel_policy_, data_, qubit); }

  void nompi_state::do_y_rotation_half_pi(qubit_type const qubit)
  { ket::gate::ranges::y_rotation_half_pi(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_y_rotation_half_pi(qubit_type const qubit)
  { ket::gate::ranges::adj_y_rotation_half_pi(parallel_policy_, data_, qubit); }

  void nompi_state::do_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::controlled_not(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_controlled_not(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_controlled_phase_shift(
    complex_type const phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::gate::ranges::controlled_phase_shift_coeff(
      parallel_policy_,
      data_, phase_coefficient, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_phase_shift(
    complex_type const phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::gate::ranges::adj_controlled_phase_shift_coeff(
      parallel_policy_,
      data_, phase_coefficient, target_qubit, control_qubit);
  }

  void nompi_state::do_controlled_v(
    complex_type const phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::gate::ranges::controlled_v_coeff(
      parallel_policy_,
      data_, phase_coefficient, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_v(
    complex_type const phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::gate::ranges::adj_controlled_v_coeff(
      parallel_policy_,
      data_, phase_coefficient, target_qubit, control_qubit);
  }

  void nompi_state::do_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1,
    control_qubit_type const control_qubit2)
  {
    ket::gate::ranges::toffoli(
      parallel_policy_,
      data_, target_qubit, control_qubit1, control_qubit2);
  }

  void nompi_state::do_adj_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1,
    control_qubit_type const control_qubit2)
  {
    ket::gate::ranges::adj_toffoli(
      parallel_policy_,
      data_, target_qubit, control_qubit1, control_qubit2);
  }

  KET_GATE_OUTCOME_TYPE nompi_state::do_projective_measurement(qubit_type const qubit)
  {
    return ket::gate::ranges::projective_measurement(
      parallel_policy_, data_, qubit, random_number_generator_);
  }

  void nompi_state::do_expectation_values()
  {
    maybe_expectation_values_
      = ket::ranges::all_spin_expectation_values<qubit_type>(
          parallel_policy_, data_);
  }

  void nompi_state::do_measure()
  {
    measured_value_
      = ket::measure(
          ket::utility::policy::make_sequential(), // parallel_policy_,
          data_, random_number_generator_);
  }

  void nompi_state::do_generate_events(int const num_events, int const seed)
  {
    if (seed < 0)
      ket::generate_events(
        ket::utility::policy::make_sequential(), // parallel_policy_,
        generated_events_, data_, num_events, random_number_generator_);
    else
      ket::generate_events(
        ket::utility::policy::make_sequential(), // parallel_policy_,
        generated_events_, data_, num_events, random_number_generator_, static_cast<seed_type>(seed));
  }

  void nompi_state::do_shor_box(
    bit_integer_type const num_exponent_qubits,
    state_integer_type const divisor, state_integer_type const base)
  {
    std::vector<qubit_type> exponent_qubits(num_exponent_qubits);
    boost::iota(exponent_qubits, static_cast<qubit_type>(total_num_qubits_-num_exponent_qubits));
    std::vector<qubit_type> modular_exponentiation_qubits(total_num_qubits_-num_exponent_qubits);
    boost::iota(modular_exponentiation_qubits, static_cast<qubit_type>(0u));

    ket::shor_box(
      parallel_policy_,
      data_, base, divisor, exponent_qubits, modular_exponentiation_qubits);
  }

  void nompi_state::do_clear(qubit_type const qubit)
  { ket::gate::ranges::clear(parallel_policy_, data_, qubit); }

  void nompi_state::do_set(qubit_type const qubit)
  { ket::gate::ranges::set(parallel_policy_, data_, qubit); }

  void nompi_state::do_depolarizing_channel(double const px, double const py, double const pz, int const seed)
  {
    BRA_uniform_real_distribution<double> distribution(0.0, px + py + pz);
    qubit_type const last_qubit = ket::make_qubit(total_num_qubits_);
    if (seed < 0)
      for (qubit_type qubit = ket::make_qubit(static_cast<bit_integer_type>(0u)); qubit < last_qubit; ++qubit)
      {
        double const probability = distribution(random_number_generator_);
        if (probability < px)
          do_pauli_x(qubit);
        else if (probability < px + py)
          do_pauli_y(qubit);
        else
          do_pauli_z(qubit);
      }
    else
    {
      random_number_generator_type temporal_random_number_generator(static_cast<seed_type>(seed));
      for (qubit_type qubit = ket::make_qubit(static_cast<bit_integer_type>(0u)); qubit < last_qubit; ++qubit)
      {
        double const probability = distribution(temporal_random_number_generator);
        if (probability < px)
          do_pauli_x(qubit);
        else if (probability < px + py)
          do_pauli_y(qubit);
        else
          do_pauli_z(qubit);
      }
    }
  }
}


# undef BRA_uniform_real_distribution
#endif // BRA_NO_MPI

