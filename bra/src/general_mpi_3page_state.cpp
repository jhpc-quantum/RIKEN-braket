#ifndef BRA_NO_MPI
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
# include <ket/mpi/gate/hadamard.hpp>
# include <ket/mpi/gate/pauli_x.hpp>
# include <ket/mpi/gate/pauli_y.hpp>
# include <ket/mpi/gate/pauli_z.hpp>
# include <ket/mpi/gate/phase_shift.hpp>
# include <ket/mpi/gate/x_rotation_half_pi.hpp>
# include <ket/mpi/gate/y_rotation_half_pi.hpp>
# include <ket/mpi/gate/controlled_not.hpp>
# include <ket/mpi/gate/controlled_phase_shift.hpp>
# include <ket/mpi/gate/controlled_v.hpp>
# include <ket/mpi/gate/toffoli.hpp>
# include <ket/mpi/gate/projective_measurement.hpp>
# include <ket/mpi/gate/clear.hpp>
# include <ket/mpi/gate/set.hpp>
# include <ket/mpi/all_spin_expectation_values.hpp>
# include <ket/mpi/measure.hpp>
# include <ket/mpi/generate_events.hpp>
# include <ket/mpi/shor_box.hpp>

# include <bra/general_mpi_3page_state.hpp>
# include <bra/state.hpp>
# include <bra/utility/closest_floating_point_of.hpp>

# ifndef BOOST_NO_CXX11_HDR_RANDOM
#   define BRA_uniform_real_distribution std::uniform_real_distribution
# else
#   define BRA_uniform_real_distribution boost::random::uniform_real_distribution
# endif


namespace bra
{
  unsigned int general_mpi_3page_state::do_num_page_qubits() const
  { return 3u; }

  unsigned int general_mpi_3page_state::do_num_pages() const
  { return 8u; }

  general_mpi_3page_state::general_mpi_3page_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    ::bra::state::seed_type const seed,
    yampi::communicator const communicator,
    yampi::environment const& environment)
    : ::bra::state(total_num_qubits, seed, communicator, environment),
      parallel_policy_(),
      mpi_policy_(),
      data_(
        mpi_policy_, num_local_qubits, initial_integer,
        permutation_, communicator, environment)
  { }

  general_mpi_3page_state::general_mpi_3page_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<qubit_type> const& initial_permutation,
    ::bra::state::seed_type const seed,
    yampi::communicator const communicator,
    yampi::environment const& environment)
    : ::bra::state(initial_permutation, seed, communicator, environment),
      parallel_policy_(),
      mpi_policy_(),
      data_(
        mpi_policy_, num_local_qubits, initial_integer,
        permutation_, communicator, environment)
  { }

  void general_mpi_3page_state::do_hadamard(
    qubit_type const qubit)
  {
    ket::mpi::gate::hadamard(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_hadamard(
    qubit_type const qubit)
  {
    ket::mpi::gate::adj_hadamard(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_pauli_x(
    qubit_type const qubit)
  {
    ket::mpi::gate::pauli_x(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_pauli_x(
    qubit_type const qubit)
  {
    ket::mpi::gate::adj_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_pauli_y(
    qubit_type const qubit)
  {
    ket::mpi::gate::pauli_y(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_pauli_y(
    qubit_type const qubit)
  {
    ket::mpi::gate::adj_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_pauli_z(
    qubit_type const qubit)
  {
    ket::mpi::gate::pauli_z(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_pauli_z(
    qubit_type const qubit)
  {
    ket::mpi::gate::adj_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_u1(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift(
      mpi_policy_, parallel_policy_,
      data_, phase, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_u1(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift(
      mpi_policy_, parallel_policy_,
      data_, phase, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, phase1, phase2, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, phase1, phase2, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, phase1, phase2, phase3, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, phase1, phase2, phase3, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_phase_shift(
    complex_type const phase_coefficient, qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, phase_coefficient, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_phase_shift(
    complex_type const phase_coefficient, qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, phase_coefficient, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_x_rotation_half_pi(
    qubit_type const qubit)
  {
    ket::mpi::gate::x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_x_rotation_half_pi(
    qubit_type const qubit)
  {
    ket::mpi::gate::adj_x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_y_rotation_half_pi(
    qubit_type const qubit)
  {
    ket::mpi::gate::y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_y_rotation_half_pi(
    qubit_type const qubit)
  {
    ket::mpi::gate::adj_y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::controlled_not(
      mpi_policy_, parallel_policy_,
      data_, target_qubit, control_qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_controlled_not(
      mpi_policy_, parallel_policy_,
      data_, target_qubit, control_qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_controlled_phase_shift(
    complex_type const phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::controlled_phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, phase_coefficient, target_qubit, control_qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_controlled_phase_shift(
    complex_type const phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_controlled_phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, phase_coefficient, target_qubit, control_qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_controlled_v(
    complex_type const phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::controlled_v_coeff(
      mpi_policy_, parallel_policy_,
      data_, phase_coefficient, target_qubit, control_qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_controlled_v(
    complex_type const phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_controlled_v_coeff(
      mpi_policy_, parallel_policy_,
      data_, phase_coefficient, target_qubit, control_qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1,
    control_qubit_type const control_qubit2)
  {
    ket::mpi::gate::toffoli(
      mpi_policy_, parallel_policy_,
      data_, target_qubit, control_qubit1, control_qubit2,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_adj_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1,
    control_qubit_type const control_qubit2)
  {
    ket::mpi::gate::adj_toffoli(
      mpi_policy_, parallel_policy_,
      data_, target_qubit, control_qubit1, control_qubit2,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  KET_GATE_OUTCOME_TYPE general_mpi_3page_state::do_projective_measurement(
    qubit_type const qubit, yampi::rank const root)
  {
    return ket::mpi::gate::projective_measurement(
      mpi_policy_, parallel_policy_,
      data_, qubit, random_number_generator_, permutation_,
      buffer_, complex_datatype_, real_pair_datatype_, root, communicator_, environment_);
  }

  void general_mpi_3page_state::do_expectation_values(yampi::rank const root)
  {
    maybe_expectation_values_
      = ket::mpi::all_spin_expectation_values<spins_allocator_type>(
          mpi_policy_, parallel_policy_,
          data_, permutation_, total_num_qubits_,
          buffer_, real_datatype_, complex_datatype_, root, communicator_, environment_);
  }

  void general_mpi_3page_state::do_measure(yampi::rank const root)
  {
    measured_value_
      = ket::mpi::measure(
          mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
          data_, random_number_generator_, permutation_,
          state_integer_datatype_, real_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_generate_events(yampi::rank const root, int const num_events, int const seed)
  {
    if (seed < 0)
      ket::mpi::generate_events(
        mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
        generated_events_, data_, num_events, random_number_generator_, permutation_,
        state_integer_datatype_, real_datatype_, communicator_, environment_);
    else
      ket::mpi::generate_events(
        mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
        generated_events_, data_, num_events, random_number_generator_, static_cast<seed_type>(seed), permutation_,
        state_integer_datatype_, real_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_shor_box(
    bit_integer_type const num_exponent_qubits,
    state_integer_type const divisor, state_integer_type const base)
  {
    std::vector<qubit_type> exponent_qubits(num_exponent_qubits);
    boost::iota(exponent_qubits, static_cast<qubit_type>(total_num_qubits_-num_exponent_qubits));
    std::vector<qubit_type> modular_exponentiation_qubits(total_num_qubits_-num_exponent_qubits);
    boost::iota(modular_exponentiation_qubits, static_cast<qubit_type>(0u));

    ket::mpi::shor_box(
      mpi_policy_, parallel_policy_,
      data_, base, divisor, exponent_qubits, modular_exponentiation_qubits,
      permutation_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_clear(qubit_type const qubit)
  {
    ket::mpi::gate::clear(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_set(qubit_type const qubit)
  {
    ket::mpi::gate::set(
      mpi_policy_, parallel_policy_,
      data_, qubit,
      permutation_, buffer_, complex_datatype_, communicator_, environment_);
  }

  void general_mpi_3page_state::do_depolarizing_channel(real_type const px, real_type const py, real_type const pz, int const seed)
  {
    typedef typename ::bra::utility::closest_floating_point_of<real_type>::type floating_point;
    BRA_uniform_real_distribution<floating_point> distribution(0.0, px + py + pz);
    qubit_type const last_qubit = ket::make_qubit(total_num_qubits_);
    if (seed < 0)
      for (qubit_type qubit = ket::make_qubit(static_cast<bit_integer_type>(0u)); qubit < last_qubit; ++qubit)
      {
        real_type const probability = static_cast<real_type>(distribution(random_number_generator_));
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
        real_type const probability = static_cast<real_type>(distribution(temporal_random_number_generator));
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

