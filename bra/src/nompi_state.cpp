#ifdef BRA_NO_MPI
# include <vector>

# include <boost/preprocessor/arithmetic/dec.hpp>
# include <boost/preprocessor/arithmetic/inc.hpp>
# include <boost/preprocessor/punctuation/comma_if.hpp>
# include <boost/preprocessor/comparison/equal.hpp>
# include <boost/preprocessor/control/iif.hpp>
# include <boost/preprocessor/repetition/repeat.hpp>
# include <boost/preprocessor/repetition/repeat_from_to.hpp>

# include <ket/gate/gate.hpp>
# include <ket/gate/hadamard.hpp>
# include <ket/gate/not_.hpp>
# include <ket/gate/pauli_x.hpp>
# include <ket/gate/pauli_y.hpp>
# include <ket/gate/pauli_z.hpp>
# include <ket/gate/swap.hpp>
# include <ket/gate/phase_shift.hpp>
# include <ket/gate/x_rotation_half_pi.hpp>
# include <ket/gate/y_rotation_half_pi.hpp>
# include <ket/gate/controlled_phase_shift.hpp>
# include <ket/gate/controlled_v.hpp>
# include <ket/gate/exponential_pauli_x.hpp>
# include <ket/gate/exponential_pauli_y.hpp>
# include <ket/gate/exponential_pauli_z.hpp>
# include <ket/gate/exponential_swap.hpp>
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

# ifndef BRA_MAX_NUM_OPERATED_QUBITS
#   define BRA_MAX_NUM_OPERATED_QUBITS 6
# endif // BRA_MAX_NUM_OPERATED_QUBITS


namespace bra
{
# ifndef KET_USE_ON_CACHE_STATE_VECTOR
  nompi_state::nompi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const total_num_qubits,
    unsigned int num_threads, ::bra::state::seed_type const seed)
    : ::bra::state{total_num_qubits, seed},
      parallel_policy_{num_threads},
      data_{make_initial_data(initial_integer, total_num_qubits)}
  { }
# else // KET_USE_ON_CACHE_STATE_VECTOR
#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
  nompi_state::nompi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const total_num_qubits,
    unsigned int num_threads, ::bra::state::seed_type const seed)
    : ::bra::state{total_num_qubits, seed},
      parallel_policy_{num_threads},
      data_{make_initial_data(initial_integer, total_num_qubits)},
      on_cache_data_{::ket::utility::integer_exp2< ::bra::state_integer_type >(KET_DEFAULT_NUM_ON_CACHE_QUBITS)}
  { }
# endif // KET_USE_ON_CACHE_STATE_VECTOR

  void nompi_state::do_i_gate(qubit_type const)
  { }

  void nompi_state::do_adj_i_gate(qubit_type const)
  { }

  void nompi_state::do_ii_gate(qubit_type const, qubit_type const)
  { }

  void nompi_state::do_adj_ii_gate(qubit_type const, qubit_type const)
  { }

  void nompi_state::do_in_gate(std::vector<qubit_type> const&)
  { }

  void nompi_state::do_adj_in_gate(std::vector<qubit_type> const&)
  { }

  void nompi_state::do_hadamard(qubit_type const qubit)
  { ket::gate::ranges::hadamard(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_hadamard(qubit_type const qubit)
  { ket::gate::ranges::adj_hadamard(parallel_policy_, data_, qubit); }

  void nompi_state::do_not_(qubit_type const qubit)
  { ket::gate::ranges::not_(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_not_(qubit_type const qubit)
  { ket::gate::ranges::adj_not_(parallel_policy_, data_, qubit); }

  void nompi_state::do_pauli_x(qubit_type const qubit)
  { ket::gate::ranges::pauli_x(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_pauli_x(qubit_type const qubit)
  { ket::gate::ranges::adj_pauli_x(parallel_policy_, data_, qubit); }

  void nompi_state::do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::pauli_x(parallel_policy_, data_, qubit1, qubit2); }

  void nompi_state::do_adj_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::adj_pauli_x(parallel_policy_, data_, qubit1, qubit2); }

  void nompi_state::do_pauli_xn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::pauli_x(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_adj_pauli_xn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_pauli_x(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_pauli_y(qubit_type const qubit)
  { ket::gate::ranges::pauli_y(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_pauli_y(qubit_type const qubit)
  { ket::gate::ranges::adj_pauli_y(parallel_policy_, data_, qubit); }

  void nompi_state::do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::pauli_y(parallel_policy_, data_, qubit1, qubit2); }

  void nompi_state::do_adj_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::adj_pauli_y(parallel_policy_, data_, qubit1, qubit2); }

  void nompi_state::do_pauli_yn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::pauli_y(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_adj_pauli_yn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_pauli_y(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_pauli_z(qubit_type const qubit)
  { ket::gate::ranges::pauli_z(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_pauli_z(qubit_type const qubit)
  { ket::gate::ranges::adj_pauli_z(parallel_policy_, data_, qubit); }

  void nompi_state::do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::pauli_z(parallel_policy_, data_, qubit1, qubit2); }

  void nompi_state::do_adj_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::adj_pauli_z(parallel_policy_, data_, qubit1, qubit2); }

  void nompi_state::do_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::pauli_z(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_adj_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_pauli_z(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_swap(qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::swap(parallel_policy_, data_, qubit1, qubit2); }

  void nompi_state::do_adj_swap(qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::adj_swap(parallel_policy_, data_, qubit1, qubit2); }

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
    complex_type const& phase_coefficient, qubit_type const qubit)
  { ket::gate::ranges::phase_shift_coeff(parallel_policy_, data_, phase_coefficient, qubit); }

  void nompi_state::do_adj_phase_shift(
    complex_type const& phase_coefficient, qubit_type const qubit)
  { ket::gate::ranges::adj_phase_shift_coeff(parallel_policy_, data_, phase_coefficient, qubit); }

  void nompi_state::do_x_rotation_half_pi(qubit_type const qubit)
  { ket::gate::ranges::x_rotation_half_pi(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_x_rotation_half_pi(qubit_type const qubit)
  { ket::gate::ranges::adj_x_rotation_half_pi(parallel_policy_, data_, qubit); }

  void nompi_state::do_y_rotation_half_pi(qubit_type const qubit)
  { ket::gate::ranges::y_rotation_half_pi(parallel_policy_, data_, qubit); }

  void nompi_state::do_adj_y_rotation_half_pi(qubit_type const qubit)
  { ket::gate::ranges::adj_y_rotation_half_pi(parallel_policy_, data_, qubit); }

  void nompi_state::do_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::gate::ranges::controlled_v_coeff(
      parallel_policy_,
      data_, phase_coefficient, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::gate::ranges::adj_controlled_v_coeff(
      parallel_policy_,
      data_, phase_coefficient, target_qubit, control_qubit);
  }

  void nompi_state::do_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  { ket::gate::ranges::exponential_pauli_x(parallel_policy_, data_, phase, qubit); }

  void nompi_state::do_adj_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  { ket::gate::ranges::adj_exponential_pauli_x(parallel_policy_, data_, phase, qubit); }

  void nompi_state::do_exponential_pauli_xx(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::exponential_pauli_x(parallel_policy_, data_, phase, qubit1, qubit2); }

  void nompi_state::do_adj_exponential_pauli_xx(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::adj_exponential_pauli_x(parallel_policy_, data_, phase, qubit1, qubit2); }

  void nompi_state::do_exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::exponential_pauli_x(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_adj_exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_exponential_pauli_x(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  { ket::gate::ranges::exponential_pauli_y(parallel_policy_, data_, phase, qubit); }

  void nompi_state::do_adj_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  { ket::gate::ranges::adj_exponential_pauli_y(parallel_policy_, data_, phase, qubit); }

  void nompi_state::do_exponential_pauli_yy(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::exponential_pauli_y(parallel_policy_, data_, phase, qubit1, qubit2); }

  void nompi_state::do_adj_exponential_pauli_yy(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::adj_exponential_pauli_y(parallel_policy_, data_, phase, qubit1, qubit2); }

  void nompi_state::do_exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::exponential_pauli_y(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_adj_exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_exponential_pauli_y(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  { ket::gate::ranges::exponential_pauli_z(parallel_policy_, data_, phase, qubit); }

  void nompi_state::do_adj_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  { ket::gate::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase, qubit); }

  void nompi_state::do_exponential_pauli_zz(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::exponential_pauli_z(parallel_policy_, data_, phase, qubit1, qubit2); }

  void nompi_state::do_adj_exponential_pauli_zz(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase, qubit1, qubit2); }

  void nompi_state::do_exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::exponential_pauli_z(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_adj_exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_exponential_swap(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::exponential_swap(parallel_policy_, data_, phase, qubit1, qubit2); }

  void nompi_state::do_adj_exponential_swap(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  { ket::gate::ranges::adj_exponential_swap(parallel_policy_, data_, phase, qubit1, qubit2); }

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

  ket::gate::outcome nompi_state::do_projective_measurement(qubit_type const qubit)
  {
    return ket::gate::ranges::projective_measurement(
      parallel_policy_, data_, random_number_generator_, qubit);
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
      = ket::ranges::measure(
          ket::utility::policy::make_sequential(), // parallel_policy_,
          data_, random_number_generator_);
  }

  void nompi_state::do_generate_events(int const num_events, int const seed)
  {
    if (seed < 0)
      ket::ranges::generate_events(
        ket::utility::policy::make_sequential(), // parallel_policy_,
        generated_events_, data_, num_events, random_number_generator_);
    else
      ket::ranges::generate_events(
        ket::utility::policy::make_sequential(), // parallel_policy_,
        generated_events_, data_, num_events, random_number_generator_, static_cast<seed_type>(seed));
  }

  void nompi_state::do_shor_box(
    state_integer_type const divisor, state_integer_type const base,
    std::vector<qubit_type> const& exponent_qubits,
    std::vector<qubit_type> const& modular_exponentiation_qubits)
  {
    ket::ranges::shor_box(
      parallel_policy_,
      data_, base, divisor, exponent_qubits, modular_exponentiation_qubits);
  }

# ifndef BRA_MAX_NUM_FUSED_QUBITS
#   ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
#   else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS 10
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
# endif // BRA_MAX_NUM_FUSED_QUBITS
  void nompi_state::do_begin_fusion()
  {
    constexpr auto max_num_fused_qubits = decltype(fused_qubits_.size()){BRA_MAX_NUM_FUSED_QUBITS};
    if (fused_qubits_.size() > max_num_fused_qubits)
      throw ::bra::too_many_operated_qubits_error{fused_qubits_.size(), max_num_fused_qubits};
  }

  void nompi_state::do_end_fusion()
  {
    switch (fused_qubits_.size())
    {
# define QUBITS(z, n, qubits) BOOST_PP_COMMA_IF(n) qubits[n]
# ifndef KET_USE_ON_CACHE_STATE_VECTOR
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_N(z, num_fused_qubits, _) \
     case num_fused_qubits:\
      ket::gate::ranges::gate(\
        parallel_policy_, data_,\
        [this](\
          auto const first, ::bra::state_integer_type const index_wo_qubits,\
          std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
          std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
          int const)\
        {\
          for (auto const& gate_ptr: this->fused_gates_)\
            gate_ptr->call(std::addressof(*first), index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
        }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
      break;\

#   else // KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_N(z, num_fused_qubits, _) \
     case num_fused_qubits:\
      ket::gate::ranges::gate(\
        parallel_policy_, data_,\
        [this](\
          auto const first, ::bra::state_integer_type const index_wo_qubits,\
          std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
          std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
          int const)\
        {\
          for (auto const& gate_ptr: this->fused_gates_)\
            gate_ptr->call(std::addressof(*first), index_wo_qubits, qubit_masks, index_masks);\
        }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
      break;\

#   endif // KET_USE_BIT_MASKS_EXPLICITLY
# else // KET_USE_ON_CACHE_STATE_VECTOR
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_N(z, num_fused_qubits, _) \
     case num_fused_qubits:\
      ket::gate::cache::ranges::gate(\
        parallel_policy_, data_, on_cache_data_,\
        [this](\
          auto const first, ::bra::state_integer_type const index_wo_qubits,\
          std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
          std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
          int const)\
        {\
          for (auto const& gate_ptr: this->fused_gates_)\
            gate_ptr->call(std::addressof(*first), index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
        }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
      break;\

#   else // KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_N(z, num_fused_qubits, _) \
     case num_fused_qubits:\
      ket::gate::cache::ranges::gate(\
        parallel_policy_, data_, on_cache_data_,\
        [this](\
          auto const first, ::bra::state_integer_type const index_wo_qubits,\
          std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
          std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
          int const)\
        {\
          for (auto const& gate_ptr: this->fused_gates_)\
            gate_ptr->call(std::addressof(*first), index_wo_qubits, qubit_masks, index_masks);\
        }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
      break;\

#   endif // KET_USE_BIT_MASKS_EXPLICITLY
# endif // KET_USE_ON_CACHE_STATE_VECTOR
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), CASE_N, nil)
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_clear(qubit_type const qubit)
  { ket::gate::ranges::clear(parallel_policy_, data_, qubit); }

  void nompi_state::do_set(qubit_type const qubit)
  { ket::gate::ranges::set(parallel_policy_, data_, qubit); }

  void nompi_state::do_controlled_i_gate(qubit_type const, control_qubit_type const)
  { }

  void nompi_state::do_adj_controlled_i_gate(qubit_type const, control_qubit_type const)
  { }

  void nompi_state::do_multi_controlled_in_gate(
    std::vector<qubit_type> const&, std::vector<control_qubit_type> const&)
  { }

  void nompi_state::do_adj_multi_controlled_in_gate(
    std::vector<qubit_type> const&, std::vector<control_qubit_type> const&)
  { }

  void nompi_state::do_controlled_hadamard(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::hadamard(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_hadamard(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_hadamard(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_hadamard(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::hadamard(parallel_policy_, data_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_hadamard(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_hadamard(parallel_policy_, data_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::not_(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_not_(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_not(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::not_(parallel_policy_, data_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_not(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_not_(parallel_policy_, data_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::pauli_x(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::gate::ranges::adj_pauli_x(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_pauli_xn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::pauli_x(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_pauli_xn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::adj_pauli_x(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_controlled_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::pauli_y(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_pauli_y(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_pauli_yn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::pauli_y(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_pauli_yn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::adj_pauli_y(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_controlled_pauli_z(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::pauli_z(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_pauli_z(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_pauli_z(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::pauli_z(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::adj_pauli_z(parallel_policy_, data_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_multi_controlled_swap(
    qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::swap(parallel_policy_, data_, target_qubit1, target_qubit2, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_DEC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{2u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_swap(
    qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_swap(parallel_policy_, data_, target_qubit1, target_qubit2, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_DEC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{2u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::phase_shift_coeff(parallel_policy_, data_, phase_coefficient, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_phase_shift_coeff(parallel_policy_, data_, phase_coefficient, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::phase_shift_coeff(parallel_policy_, data_, phase_coefficient, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_phase_shift_coeff(parallel_policy_, data_, phase_coefficient, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_u1(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::phase_shift(parallel_policy_, data_, phase, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_u1(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_phase_shift(parallel_policy_, data_, phase, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_u1(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::phase_shift(parallel_policy_, data_, phase, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_u1(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_phase_shift(parallel_policy_, data_, phase, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::phase_shift2(parallel_policy_, data_, phase1, phase2, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_phase_shift2(parallel_policy_, data_, phase1, phase2, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::phase_shift2(parallel_policy_, data_, phase1, phase2, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_phase_shift2(parallel_policy_, data_, phase1, phase2, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::x_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_x_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::x_rotation_half_pi(parallel_policy_, data_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_x_rotation_half_pi(parallel_policy_, data_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::y_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_y_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::y_rotation_half_pi(parallel_policy_, data_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_y_rotation_half_pi(parallel_policy_, data_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_multi_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::controlled_v_coeff(parallel_policy_, data_, phase_coefficient, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_controlled_v_coeff(parallel_policy_, data_, phase_coefficient, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_exponential_pauli_x(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::exponential_pauli_x(parallel_policy_, data_, phase, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_exponential_pauli_x(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_exponential_pauli_x(parallel_policy_, data_, phase, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::exponential_pauli_x(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::adj_exponential_pauli_x(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_controlled_exponential_pauli_y(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::exponential_pauli_y(parallel_policy_, data_, phase, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_exponential_pauli_y(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_exponential_pauli_y(parallel_policy_, data_, phase, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::exponential_pauli_y(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::adj_exponential_pauli_y(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::exponential_pauli_z(parallel_policy_, data_, phase, target_qubit, control_qubit); }

  void nompi_state::do_adj_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  { ket::gate::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase, target_qubit, control_qubit); }

  void nompi_state::do_multi_controlled_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::exponential_pauli_z(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_multi_controlled_exponential_swap(
    real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::exponential_swap(parallel_policy_, data_, phase, target_qubit1, target_qubit2, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_DEC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{2u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_exponential_swap(
    real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_exponential_swap(parallel_policy_, data_, phase, target_qubit1, target_qubit2, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_DEC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{2u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }
} // namespace bra


#endif // BRA_NO_MPI
