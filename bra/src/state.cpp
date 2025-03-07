#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#ifdef BRA_NO_MPI
# include <chrono>
# include <memory>
#endif
#include <stdexcept>

#include <boost/preprocessor/arithmetic/dec.hpp>

#ifndef BRA_NO_MPI
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/wall_clock.hpp>
#endif // BRA_NO_MPI

#include <ket/qubit.hpp>
#include <ket/control.hpp>

#include <bra/state.hpp>
#include <bra/utility/closest_floating_point_of.hpp>

#ifndef BRA_NO_MPI
# define BRA_clock yampi::wall_clock
#else
# define BRA_clock std::chrono::system_clock
#endif


namespace bra
{
  too_many_operated_qubits_error::too_many_operated_qubits_error(std::size_t const num_operated_qubits, std::size_t const max_num_operated_qubits)
    : std::runtime_error{std::string{"the number of operated qubits ("}.append(std::to_string(num_operated_qubits)).append(") is larger than its maximum value (").append(std::to_string(max_num_operated_qubits)).append(")").c_str()}
  { }

  unsupported_fused_gate_error::unsupported_fused_gate_error(std::string const& mnemonic)
    : std::runtime_error{(mnemonic + " is not supported in gate fusion").c_str()}
  { }

#ifndef BRA_MAX_NUM_FUSED_QUBITS
# ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#   define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
# else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#   define BRA_MAX_NUM_FUSED_QUBITS 10
# endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#endif // BRA_MAX_NUM_FUSED_QUBITS
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
      is_in_fusion_{false},
      fused_qubits_{},
      to_qubit_in_fused_gate_{},
      random_number_generator_{seed},
      permutation_{static_cast<permutation_type::size_type>(total_num_qubits)},
      buffer_{},
      communicator_{communicator},
      environment_{environment},
      finish_times_and_processes_{}
  {
    constexpr auto max_num_fused_qubits = bit_integer_type{BRA_MAX_NUM_FUSED_QUBITS};
    fused_qubits_.reserve(max_num_fused_qubits);

    finish_times_and_processes_.reserve(2u);
  }

  state::state(
    bit_integer_type const total_num_qubits,
    seed_type const seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : total_num_qubits_{total_num_qubits},
      last_outcomes_{total_num_qubits, ket::gate::outcome::unspecified},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      is_in_fusion_{false},
      fused_qubits_{},
      to_qubit_in_fused_gate_{},
      random_number_generator_{seed},
      permutation_{static_cast<permutation_type::size_type>(total_num_qubits)},
      buffer_(num_elements_in_buffer),
      communicator_{communicator},
      environment_{environment},
      finish_times_and_processes_{}
  {
    constexpr auto max_num_fused_qubits = bit_integer_type{BRA_MAX_NUM_FUSED_QUBITS};
    fused_qubits_.reserve(max_num_fused_qubits);

    finish_times_and_processes_.reserve(2u);
  }

  state::state(
    std::vector<permutated_qubit_type> const& initial_permutation,
    seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : total_num_qubits_{static_cast<bit_integer_type>(initial_permutation.size())},
      last_outcomes_{total_num_qubits_, ket::gate::outcome::unspecified},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      is_in_fusion_{false},
      fused_qubits_{},
      to_qubit_in_fused_gate_{},
      random_number_generator_{seed},
      permutation_{
        std::begin(initial_permutation), std::end(initial_permutation)},
      buffer_{},
      communicator_{communicator},
      environment_{environment},
      finish_times_and_processes_{}
  {
    constexpr auto max_num_fused_qubits = bit_integer_type{BRA_MAX_NUM_FUSED_QUBITS};
    fused_qubits_.reserve(max_num_fused_qubits);

    finish_times_and_processes_.reserve(2u);
  }

  state::state(
    std::vector<permutated_qubit_type> const& initial_permutation,
    seed_type const seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : total_num_qubits_{static_cast<bit_integer_type>(initial_permutation.size())},
      last_outcomes_{total_num_qubits_, ket::gate::outcome::unspecified},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      is_in_fusion_{false},
      fused_qubits_{},
      to_qubit_in_fused_gate_{},
      random_number_generator_{seed},
      permutation_{
        std::begin(initial_permutation), std::end(initial_permutation)},
      buffer_(num_elements_in_buffer),
      communicator_{communicator},
      environment_{environment},
      finish_times_and_processes_{}
  {
    constexpr auto max_num_fused_qubits = bit_integer_type{BRA_MAX_NUM_FUSED_QUBITS};
    fused_qubits_.reserve(max_num_fused_qubits);

    finish_times_and_processes_.reserve(2u);
  }
#else // BRA_NO_MPI
  state::state(bit_integer_type const total_num_qubits, seed_type const seed)
    : total_num_qubits_{total_num_qubits},
      last_outcomes_{total_num_qubits, ket::gate::outcome::unspecified},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      is_in_fusion_{false},
      fused_qubits_{},
      to_qubit_in_fused_gate_{},
      random_number_generator_{seed},
      finish_times_and_processes_{}
  {
    constexpr auto max_num_fused_qubits = bit_integer_type{BRA_MAX_NUM_FUSED_QUBITS};
    fused_qubits_.reserve(max_num_fused_qubits);

    finish_times_and_processes_.reserve(2u);
  }
#endif // BRA_NO_MPI

  state& state::i_gate(qubit_type const qubit)
  { do_i_gate(qubit); return *this; }

  state& state::adj_i_gate(qubit_type const qubit)
  { do_adj_i_gate(qubit); return *this; }

  state& state::ii_gate(qubit_type const qubit1, qubit_type const qubit2)
  { do_ii_gate(qubit1, qubit2); return *this; }

  state& state::adj_ii_gate(qubit_type const qubit1, qubit_type const qubit2)
  { do_adj_ii_gate(qubit1, qubit2); return *this; }

  state& state::in_gate(std::vector<qubit_type> const& qubits)
  { do_in_gate(qubits); return *this; }

  state& state::adj_in_gate(std::vector<qubit_type> const& qubits)
  { do_adj_in_gate(qubits); return *this; }

  state& state::hadamard(qubit_type const qubit)
  { do_hadamard(qubit); return *this; }

  state& state::adj_hadamard(qubit_type const qubit)
  { do_adj_hadamard(qubit); return *this; }

  state& state::not_(qubit_type const qubit)
  { do_not_(qubit); return *this; }

  state& state::adj_not_(qubit_type const qubit)
  { do_adj_not_(qubit); return *this; }

  state& state::pauli_x(qubit_type const qubit)
  { do_pauli_x(qubit); return *this; }

  state& state::adj_pauli_x(qubit_type const qubit)
  { do_adj_pauli_x(qubit); return *this; }

  state& state::pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  { do_pauli_xx(qubit1, qubit2); return *this; }

  state& state::adj_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  { do_adj_pauli_xx(qubit1, qubit2); return *this; }

  state& state::pauli_xn(std::vector<qubit_type> const& qubits)
  { do_pauli_xn(qubits); return *this; }

  state& state::adj_pauli_xn(std::vector<qubit_type> const& qubits)
  { do_adj_pauli_xn(qubits); return *this; }

  state& state::pauli_y(qubit_type const qubit)
  { do_pauli_y(qubit); return *this; }

  state& state::adj_pauli_y(qubit_type const qubit)
  { do_adj_pauli_y(qubit); return *this; }

  state& state::pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  { do_pauli_yy(qubit1, qubit2); return *this; }

  state& state::adj_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  { do_adj_pauli_yy(qubit1, qubit2); return *this; }

  state& state::pauli_yn(std::vector<qubit_type> const& qubits)
  { do_pauli_yn(qubits); return *this; }

  state& state::adj_pauli_yn(std::vector<qubit_type> const& qubits)
  { do_adj_pauli_yn(qubits); return *this; }

  state& state::pauli_z(qubit_type const qubit)
  { do_pauli_z(qubit); return *this; }

  state& state::adj_pauli_z(qubit_type const qubit)
  { do_adj_pauli_z(qubit); return *this; }

  state& state::pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  { do_pauli_zz(qubit1, qubit2); return *this; }

  state& state::adj_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  { do_adj_pauli_zz(qubit1, qubit2); return *this; }

  state& state::pauli_zn(std::vector<qubit_type> const& qubits)
  { do_pauli_zn(qubits); return *this; }

  state& state::adj_pauli_zn(std::vector<qubit_type> const& qubits)
  { do_adj_pauli_zn(qubits); return *this; }

  state& state::swap(qubit_type const qubit1, qubit_type const qubit2)
  { do_swap(qubit1, qubit2); return *this; }

  state& state::adj_swap(qubit_type const qubit1, qubit_type const qubit2)
  { do_adj_swap(qubit1, qubit2); return *this; }

  state& state::u1(real_type const phase, qubit_type const qubit)
  { do_u1(phase, qubit); return *this; }

  state& state::adj_u1(real_type const phase, qubit_type const qubit)
  { do_adj_u1(phase, qubit); return *this; }

  state& state::u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  { do_u2(phase1, phase2, qubit); return *this; }

  state& state::adj_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  { do_adj_u2(phase1, phase2, qubit); return *this; }

  state& state::u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  { do_u3(phase1, phase2, phase3, qubit); return *this; }

  state& state::adj_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  { do_adj_u3(phase1, phase2, phase3, qubit); return *this; }

  state& state::phase_shift(
    complex_type const& phase_coefficient, qubit_type const qubit)
  { do_phase_shift(phase_coefficient, qubit); return *this; }

  state& state::adj_phase_shift(
    complex_type const& phase_coefficient, qubit_type const qubit)
  { do_adj_phase_shift(phase_coefficient, qubit); return *this; }

  state& state::x_rotation_half_pi(qubit_type const qubit)
  { do_x_rotation_half_pi(qubit); return *this; }

  state& state::adj_x_rotation_half_pi(qubit_type const qubit)
  { do_adj_x_rotation_half_pi(qubit); return *this; }

  state& state::y_rotation_half_pi(qubit_type const qubit)
  { do_y_rotation_half_pi(qubit); return *this; }

  state& state::adj_y_rotation_half_pi(qubit_type const qubit)
  { do_adj_y_rotation_half_pi(qubit); return *this; }

  state& state::controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_v(phase_coefficient, target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_v(phase_coefficient, target_qubit, control_qubit); return *this; }

  state& state::exponential_pauli_x(real_type const phase, qubit_type const qubit)
  { do_exponential_pauli_x(phase, qubit); return *this; }

  state& state::adj_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  { do_adj_exponential_pauli_x(phase, qubit); return *this; }

  state& state::exponential_pauli_xx(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { do_exponential_pauli_xx(phase, qubit1, qubit2); return *this; }

  state& state::adj_exponential_pauli_xx(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { do_adj_exponential_pauli_xx(phase, qubit1, qubit2); return *this; }

  state& state::exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& qubits)
  { do_exponential_pauli_xn(phase, qubits); return *this; }

  state& state::adj_exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& qubits)
  { do_adj_exponential_pauli_xn(phase, qubits); return *this; }

  state& state::exponential_pauli_y(real_type const phase, qubit_type const qubit)
  { do_exponential_pauli_y(phase, qubit); return *this; }

  state& state::adj_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  { do_adj_exponential_pauli_y(phase, qubit); return *this; }

  state& state::exponential_pauli_yy(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { do_exponential_pauli_yy(phase, qubit1, qubit2); return *this; }

  state& state::adj_exponential_pauli_yy(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { do_adj_exponential_pauli_yy(phase, qubit1, qubit2); return *this; }

  state& state::exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& qubits)
  { do_exponential_pauli_yn(phase, qubits); return *this; }

  state& state::adj_exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& qubits)
  { do_adj_exponential_pauli_yn(phase, qubits); return *this; }

  state& state::exponential_pauli_z(real_type const phase, qubit_type const qubit)
  { do_exponential_pauli_z(phase, qubit); return *this; }

  state& state::adj_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  { do_adj_exponential_pauli_z(phase, qubit); return *this; }

  state& state::exponential_pauli_zz(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { do_exponential_pauli_zz(phase, qubit1, qubit2); return *this; }

  state& state::adj_exponential_pauli_zz(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { do_adj_exponential_pauli_zz(phase, qubit1, qubit2); return *this; }

  state& state::exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& qubits)
  { do_exponential_pauli_zn(phase, qubits); return *this; }

  state& state::adj_exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& qubits)
  { do_adj_exponential_pauli_zn(phase, qubits); return *this; }

  state& state::exponential_swap(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { do_exponential_swap(phase, qubit1, qubit2); return *this; }

  state& state::adj_exponential_swap(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { do_adj_exponential_swap(phase, qubit1, qubit2); return *this; }

  state& state::toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  { do_toffoli(target_qubit, control_qubit1, control_qubit2); return *this; }

  state& state::adj_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  { do_adj_toffoli(target_qubit, control_qubit1, control_qubit2); return *this; }

#ifndef BRA_NO_MPI
  state& state::projective_measurement(qubit_type const qubit, yampi::rank const root)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"M"};

    last_outcomes_[static_cast<bit_integer_type>(qubit)]
      = do_projective_measurement(qubit, root);
    return *this;
  }

  state& state::measurement(yampi::rank const root)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"MEASURE"};

    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), ::bra::finished_process::operations));

    do_expectation_values(root);
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), ::bra::finished_process::begin_measurement));

    return *this;
  }

  state& state::generate_events(yampi::rank const root, int const num_events, int const seed)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"GENERATE EVENTS"};

    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), ::bra::finished_process::operations));

    do_generate_events(root, num_events, seed);
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(environment_), ::bra::finished_process::generate_events));

    return *this;
  }

  state& state::exit(yampi::rank const root)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"EXIT"};

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
  state& state::projective_measurement(qubit_type const qubit)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"M"};

    last_outcomes_[static_cast<bit_integer_type>(qubit)]
      = do_projective_measurement(qubit);
    return *this;
  }

  state& state::measurement()
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"MEASURE"};

    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), ::bra::finished_process::operations));

    do_expectation_values();
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), ::bra::finished_process::begin_measurement));

    return *this;
  }

  state& state::generate_events(int const num_events, int const seed)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"GENERATE EVENTS"};

    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), ::bra::finished_process::operations));

    do_generate_events(num_events, seed);
    finish_times_and_processes_.push_back(
      std::make_pair(
        BRA_clock::now(), ::bra::finished_process::generate_events));

    return *this;
  }

  state& state::exit()
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"EXIT"};

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

  state& state::shor_box(bit_integer_type const num_exponent_qubits, state_integer_type const divisor, state_integer_type const base)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"SHORBOX"};

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

  state& state::begin_fusion(std::vector<qubit_type> const& fused_qubits)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"BEGIN FUSION"};

    is_in_fusion_ = true;

    fused_qubits_.reserve(fused_qubits.size());
    using std::begin;
    using std::end;
    std::copy(begin(fused_qubits), end(fused_qubits), std::back_inserter(fused_qubits_));

    auto const num_fused_qubits = fused_qubits_.size();
    for (auto qubit_in_fused_gate = decltype(num_fused_qubits){0u}; qubit_in_fused_gate < num_fused_qubits; ++qubit_in_fused_gate)
      to_qubit_in_fused_gate_.emplace(fused_qubits_[qubit_in_fused_gate], static_cast< ::bra::qubit_type >(qubit_in_fused_gate));

    do_begin_fusion();

    return *this;
  }

  state& state::end_fusion()
  {
    if (not is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"END FUSION"};

    do_end_fusion();

    to_qubit_in_fused_gate_.clear();
    fused_qubits_.clear();
    is_in_fusion_ = false;

    return *this;
  }

  state& state::clear(qubit_type const qubit)
  { do_clear(qubit); return *this; }

  state& state::set(qubit_type const qubit)
  { do_set(qubit); return *this; }

  state& state::depolarizing_channel(real_type const px, real_type const py, real_type const pz, int const seed)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"DEPOLARIZING CHANNEL"};

    using floating_point_type = typename ::bra::utility::closest_floating_point_of<real_type>::type;
    auto distribution = std::uniform_real_distribution<floating_point_type>{0.0, 1.0};
    auto const last_qubit = ket::make_qubit<state_integer_type>(total_num_qubits_);
    if (seed < 0)
      for (auto qubit = ket::make_qubit<state_integer_type>(bit_integer_type{0u}); qubit < last_qubit; ++qubit)
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
      for (auto qubit = ket::make_qubit<state_integer_type>(static_cast<bit_integer_type>(0u)); qubit < last_qubit; ++qubit)
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

  state& state::controlled_i_gate(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_i_gate(target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_i_gate(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_i_gate(target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_in_gate(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_in_gate(target_qubits, control_qubits); return *this; }

  state& state::adj_multi_controlled_in_gate(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_in_gate(target_qubits, control_qubits); return *this; }

  state& state::controlled_hadamard(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_hadamard(target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_hadamard(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_hadamard(target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_hadamard(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_hadamard(target_qubit, control_qubits); return *this; }

  state& state::adj_multi_controlled_hadamard(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_hadamard(target_qubit, control_qubits); return *this; }

  state& state::controlled_not(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_not(target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_not(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_not(target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_not(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_not(target_qubit, control_qubits); return *this; }

  state& state::adj_multi_controlled_not(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_not(target_qubit, control_qubits); return *this; }

  state& state::controlled_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_pauli_x(target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_pauli_x(target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_pauli_xn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_pauli_xn(target_qubits, control_qubits); return *this; }

  state& state::adj_multi_controlled_pauli_xn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_pauli_xn(target_qubits, control_qubits); return *this; }

  state& state::controlled_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_pauli_y(target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_pauli_y(target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_pauli_yn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_pauli_yn(target_qubits, control_qubits); return *this; }

  state& state::adj_multi_controlled_pauli_yn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_pauli_yn(target_qubits, control_qubits); return *this; }

  state& state::controlled_pauli_z(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_pauli_z(target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_pauli_z(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_pauli_z(target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_pauli_zn(target_qubits, control_qubits); return *this; }

  state& state::adj_multi_controlled_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_pauli_zn(target_qubits, control_qubits); return *this; }

  state& state::multi_controlled_swap(qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_swap(target_qubit1, target_qubit2, control_qubits); return *this; }

  state& state::adj_multi_controlled_swap(qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_swap(target_qubit1, target_qubit2, control_qubits); return *this; }

  state& state::controlled_phase_shift(complex_type const& phase_coefficient, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_phase_shift(phase_coefficient, target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_phase_shift(complex_type const& phase_coefficient, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_phase_shift(phase_coefficient, target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_phase_shift(complex_type const& phase_coefficient, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_phase_shift(phase_coefficient, target_qubit, control_qubits); return *this; }

  state& state::adj_multi_controlled_phase_shift(complex_type const& phase_coefficient, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_phase_shift(phase_coefficient, target_qubit, control_qubits); return *this; }

  state& state::controlled_u1(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_u1(phase, target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_u1(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_u1(phase, target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_u1(real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_u1(phase, target_qubit, control_qubits); return *this; }

  state& state::adj_multi_controlled_u1(real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_u1(phase, target_qubit, control_qubits); return *this; }

  state& state::controlled_u2(real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_u2(phase1, phase2, target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_u2(real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_u2(phase1, phase2, target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_u2(real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_u2(phase1, phase2, target_qubit, control_qubits); return *this; }

  state& state::adj_multi_controlled_u2(real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_u2(phase1, phase2, target_qubit, control_qubits); return *this; }

  state& state::controlled_u3(real_type const phase1, real_type const phase2, real_type const phase3, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_u3(phase1, phase2, phase3, target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_u3(real_type const phase1, real_type const phase2, real_type const phase3, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_u3(phase1, phase2, phase3, target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_u3(real_type const phase1, real_type const phase2, real_type const phase3, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_u3(phase1, phase2, phase3, target_qubit, control_qubits); return *this; }

  state& state::adj_multi_controlled_u3(real_type const phase1, real_type const phase2, real_type const phase3, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_u3(phase1, phase2, phase3, target_qubit, control_qubits); return *this; }

  state& state::controlled_x_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_x_rotation_half_pi(target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_x_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_x_rotation_half_pi(target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_x_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_x_rotation_half_pi(target_qubit, control_qubits); return *this; }

  state& state::adj_multi_controlled_x_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_x_rotation_half_pi(target_qubit, control_qubits); return *this; }

  state& state::controlled_y_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_y_rotation_half_pi(target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_y_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_y_rotation_half_pi(target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_y_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_y_rotation_half_pi(target_qubit, control_qubits); return *this; }

  state& state::adj_multi_controlled_y_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_y_rotation_half_pi(target_qubit, control_qubits); return *this; }

  state& state::multi_controlled_v(complex_type const& phase_coefficient, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_v(phase_coefficient, target_qubit, control_qubits); return *this; }

  state& state::adj_multi_controlled_v(complex_type const& phase_coefficient, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_v(phase_coefficient, target_qubit, control_qubits); return *this; }

  state& state::controlled_exponential_pauli_x(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_exponential_pauli_x(phase, target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_exponential_pauli_x(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_exponential_pauli_x(phase, target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_exponential_pauli_xn(phase, target_qubits, control_qubits); return *this; }

  state& state::adj_multi_controlled_exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_exponential_pauli_xn(phase, target_qubits, control_qubits); return *this; }

  state& state::controlled_exponential_pauli_y(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_exponential_pauli_y(phase, target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_exponential_pauli_y(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_exponential_pauli_y(phase, target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_exponential_pauli_yn(phase, target_qubits, control_qubits); return *this; }

  state& state::adj_multi_controlled_exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_exponential_pauli_yn(phase, target_qubits, control_qubits); return *this; }

  state& state::controlled_exponential_pauli_z(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_controlled_exponential_pauli_z(phase, target_qubit, control_qubit); return *this; }

  state& state::adj_controlled_exponential_pauli_z(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { do_adj_controlled_exponential_pauli_z(phase, target_qubit, control_qubit); return *this; }

  state& state::multi_controlled_exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_exponential_pauli_zn(phase, target_qubits, control_qubits); return *this; }

  state& state::adj_multi_controlled_exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_exponential_pauli_zn(phase, target_qubits, control_qubits); return *this; }

  state& state::multi_controlled_exponential_swap(real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits)
  { do_multi_controlled_exponential_swap(phase, target_qubit1, target_qubit2, control_qubits); return *this; }

  state& state::adj_multi_controlled_exponential_swap(real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits)
  { do_adj_multi_controlled_exponential_swap(phase, target_qubit1, target_qubit2, control_qubits); return *this; }
} // namespace bra


#undef BRA_clock
