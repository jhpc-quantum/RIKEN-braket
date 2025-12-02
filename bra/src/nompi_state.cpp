#ifdef BRA_NO_MPI
# include <iostream>
# include <sstream>
# include <vector>
# include <array>
# include <iterator>
# include <algorithm>
# include <numeric>
# include <utility>

# include <boost/preprocessor/arithmetic/dec.hpp>
# include <boost/preprocessor/arithmetic/inc.hpp>
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
# include <ket/gate/sqrt_pauli_x.hpp>
# include <ket/gate/sqrt_pauli_y.hpp>
# include <ket/gate/sqrt_pauli_z.hpp>
# include <ket/gate/phase_shift.hpp>
# include <ket/gate/x_rotation_half_pi.hpp>
# include <ket/gate/y_rotation_half_pi.hpp>
# include <ket/gate/controlled_phase_shift.hpp>
# include <ket/gate/exponential_pauli_x.hpp>
# include <ket/gate/exponential_pauli_y.hpp>
# include <ket/gate/exponential_pauli_z.hpp>
# include <ket/gate/exponential_swap.hpp>
# include <ket/gate/toffoli.hpp>
# include <ket/gate/projective_measurement.hpp>
# include <ket/gate/clear.hpp>
# include <ket/gate/set.hpp>
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   include <ket/gate/utility/cache_aware_iterator.hpp>
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/pauli_index_coeff.hpp>
# include <ket/all_spin_expectation_values.hpp>
# include <ket/print_amplitudes.hpp>
# include <ket/measure.hpp>
# include <ket/generate_events.hpp>
# include <ket/expectation_value.hpp>
# include <ket/shor_box.hpp>
# include <ket/utility/all_in_state_vector.hpp>
# include <ket/utility/none_in_state_vector.hpp>

# include <bra/nompi_state.hpp>
# include <bra/state.hpp>
# include <bra/types.hpp>
# include <bra/fused_gate.hpp>

# ifndef BRA_MAX_NUM_OPERATED_QUBITS
#   define BRA_MAX_NUM_OPERATED_QUBITS 6
# endif // BRA_MAX_NUM_OPERATED_QUBITS


namespace bra
{
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION)
  nompi_state::nompi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const total_num_qubits,
    unsigned int num_threads, ::bra::state::seed_type const seed)
    : ::bra::state{total_num_qubits, seed},
      parallel_policy_{num_threads},
      data_{make_initial_data(initial_integer, total_num_qubits)},
      fused_gates_{}
  { }
# elif !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  nompi_state::nompi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const total_num_qubits,
    unsigned int num_threads, ::bra::state::seed_type const seed)
    : ::bra::state{total_num_qubits, seed},
      parallel_policy_{num_threads},
      data_{make_initial_data(initial_integer, total_num_qubits)},
      fused_gates_{},
      cache_aware_fused_gates_{}
  { }
# else
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
      on_cache_data_{::ket::utility::integer_exp2< ::bra::state_integer_type >(KET_DEFAULT_NUM_ON_CACHE_QUBITS)},
      fused_gates_{}
  { }
# endif

  void nompi_state::do_i_gate(qubit_type const qubit)
  { }

  void nompi_state::do_ic_gate(control_qubit_type const control_qubit)
  { }

  void nompi_state::do_ii_gate(qubit_type const qubit1, qubit_type const qubit2)
  { }

  void nompi_state::do_in_gate(std::vector<qubit_type> const& qubits)
  { }

  void nompi_state::do_hadamard(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_hadamard<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_hadamard<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::hadamard(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_not_(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_not_<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_not_<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::not_(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_x<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_x<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::pauli_x(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xx<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xx<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::pauli_x(parallel_policy_, data_, qubit1, qubit2);
  }

  void nompi_state::do_pauli_xn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xn<fused_gate_iterator> >(qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xn<cache_aware_fused_gate_iterator> >(qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::pauli_x(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_y<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_y<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::pauli_y(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yy<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yy<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::pauli_y(parallel_policy_, data_, qubit1, qubit2);
  }

  void nompi_state::do_pauli_yn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yn<fused_gate_iterator> >(qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yn<cache_aware_fused_gate_iterator> >(qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::pauli_y(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_z<fused_gate_iterator> >(control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_z<cache_aware_fused_gate_iterator> >(control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::pauli_z(parallel_policy_, data_, control_qubit);
  }

  void nompi_state::do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zz<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zz<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::pauli_z(parallel_policy_, data_, qubit1, qubit2);
  }

  void nompi_state::do_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zn<fused_gate_iterator> >(qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zn<cache_aware_fused_gate_iterator> >(qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::pauli_z(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_swap(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_swap<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_swap<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::swap(parallel_policy_, data_, qubit1, qubit2);
  }

  void nompi_state::do_sqrt_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_x<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::sqrt_pauli_x(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_adj_sqrt_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_x<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_sqrt_pauli_x(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_sqrt_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_y<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::sqrt_pauli_y(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_adj_sqrt_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_y<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_sqrt_pauli_y(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_sqrt_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_z<fused_gate_iterator> >(control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::sqrt_pauli_z(parallel_policy_, data_, control_qubit);
  }

  void nompi_state::do_adj_sqrt_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_z<fused_gate_iterator> >(control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_sqrt_pauli_z(parallel_policy_, data_, control_qubit);
  }

  void nompi_state::do_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zz<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zz<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::sqrt_pauli_z(parallel_policy_, data_, qubit1, qubit2);
  }

  void nompi_state::do_adj_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zz<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zz<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_sqrt_pauli_z(parallel_policy_, data_, qubit1, qubit2);
  }

  void nompi_state::do_sqrt_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zn<fused_gate_iterator> >(qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::sqrt_pauli_z(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_adj_sqrt_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zn<fused_gate_iterator> >(qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_sqrt_pauli_z(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_u1(real_type const phase, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u1<fused_gate_iterator> >(phase, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u1<cache_aware_fused_gate_iterator> >(phase, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::phase_shift(parallel_policy_, data_, phase, control_qubit);
  }

  void nompi_state::do_adj_u1(real_type const phase, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u1<fused_gate_iterator> >(phase, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u1<cache_aware_fused_gate_iterator> >(phase, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_phase_shift(parallel_policy_, data_, phase, control_qubit);
  }

  void nompi_state::do_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u2<fused_gate_iterator> >(phase1, phase2, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::phase_shift2(parallel_policy_, data_, phase1, phase2, qubit);
  }

  void nompi_state::do_adj_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u2<fused_gate_iterator> >(phase1, phase2, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_phase_shift2(parallel_policy_, data_, phase1, phase2, qubit);
  }

  void nompi_state::do_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u3<fused_gate_iterator> >(phase1, phase2, phase3, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, qubit);
  }

  void nompi_state::do_adj_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u3<fused_gate_iterator> >(phase1, phase2, phase3, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, qubit);
  }

  void nompi_state::do_phase_shift(
    complex_type const& phase_coefficient, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_phase_shift<fused_gate_iterator> >(phase_coefficient, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::phase_shift_coeff(parallel_policy_, data_, phase_coefficient, control_qubit);
  }

  void nompi_state::do_adj_phase_shift(
    complex_type const& phase_coefficient, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_phase_shift<fused_gate_iterator> >(phase_coefficient, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_phase_shift_coeff(parallel_policy_, data_, phase_coefficient, control_qubit);
  }

  void nompi_state::do_x_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_x_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::x_rotation_half_pi(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_adj_x_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_x_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_x_rotation_half_pi(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_y_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_y_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::y_rotation_half_pi(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_adj_y_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_y_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_y_rotation_half_pi(parallel_policy_, data_, qubit);
  }

  void nompi_state::do_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_x<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_x<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::exponential_pauli_x(parallel_policy_, data_, phase, qubit);
  }

  void nompi_state::do_adj_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_x<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_x<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_exponential_pauli_x(parallel_policy_, data_, phase, qubit);
  }

  void nompi_state::do_exponential_pauli_xx(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xx<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xx<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::exponential_pauli_x(parallel_policy_, data_, phase, qubit1, qubit2);
  }

  void nompi_state::do_adj_exponential_pauli_xx(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xx<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xx<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_exponential_pauli_x(parallel_policy_, data_, phase, qubit1, qubit2);
  }

  void nompi_state::do_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::exponential_pauli_x(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_adj_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_exponential_pauli_x(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_y<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_y<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::exponential_pauli_y(parallel_policy_, data_, phase, qubit);
  }

  void nompi_state::do_adj_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_y<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_y<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_exponential_pauli_y(parallel_policy_, data_, phase, qubit);
  }

  void nompi_state::do_exponential_pauli_yy(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yy<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yy<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::exponential_pauli_y(parallel_policy_, data_, phase, qubit1, qubit2);
  }

  void nompi_state::do_adj_exponential_pauli_yy(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yy<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yy<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_exponential_pauli_y(parallel_policy_, data_, phase, qubit1, qubit2);
  }

  void nompi_state::do_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::exponential_pauli_y(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_adj_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_exponential_pauli_y(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_z<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::exponential_pauli_z(parallel_policy_, data_, phase, qubit);
  }

  void nompi_state::do_adj_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_z<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase, qubit);
  }

  void nompi_state::do_exponential_pauli_zz(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zz<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zz<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::exponential_pauli_z(parallel_policy_, data_, phase, qubit1, qubit2);
  }

  void nompi_state::do_adj_exponential_pauli_zz(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zz<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zz<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase, qubit1, qubit2);
  }

  void nompi_state::do_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::exponential_pauli_z(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_adj_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void nompi_state::do_exponential_swap(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_swap<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_swap<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::exponential_swap(parallel_policy_, data_, phase, qubit1, qubit2);
  }

  void nompi_state::do_adj_exponential_swap(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_swap<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_swap<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_exponential_swap(parallel_policy_, data_, phase, qubit1, qubit2);
  }

  void nompi_state::do_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_toffoli<fused_gate_iterator> >(
          target_qubit, control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_toffoli<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::toffoli(parallel_policy_, data_, target_qubit, control_qubit1, control_qubit2);
  }

  ::ket::gate::outcome nompi_state::do_projective_measurement(qubit_type const qubit)
  { return ket::gate::ranges::projective_measurement(parallel_policy_, data_, random_number_generator_, qubit); }

  void nompi_state::do_expectation_values()
  { maybe_expectation_values_ = ket::ranges::all_spin_expectation_values<qubit_type>(parallel_policy_, data_); }

  void nompi_state::do_amplitudes()
  {
    std::ostringstream oss;

    ket::println_amplitudes(
      oss, data_,
      [this](::bra::state_integer_type const qubit_value, ::bra::complex_type const& amplitude)
      {
        std::ostringstream oss;
        using std::real;
        using std::imag;
        oss << ::bra::state_detail::integer_to_bits_string(qubit_value, this->total_num_qubits_) << " => " << real(amplitude) << " + " << imag(amplitude) << " i";
        return oss.str();
      }, std::string{"\n"});

    std::cout << oss.str() << std::flush;
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

  void nompi_state::do_expectation_value(std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits)
  {
    auto const num_operated_qubits = operated_qubits.size();
    auto const pauli_string_space_element = to_pauli_string_space(operator_literal_or_variable_name);

    if (num_operated_qubits != pauli_string_space_element.num_qubits())
      throw ::bra::wrong_pauli_string_length_error{num_operated_qubits, pauli_string_space_element.num_qubits()};

    switch (num_operated_qubits)
    {
# define OPERATED_QUBITS(z, n, _) , operated_qubits[n]
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   define CASE_N(z, num_operated_qubits_, _) \
     case num_operated_qubits_:\
      result_\
        = ket::ranges::expectation_value(\
            parallel_policy_, data_,\
            [&pauli_string_space_element](\
              auto const first, state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, num_operated_qubits_ > const& unsorted_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_INC(num_operated_qubits_) > const& sorted_qubits_with_sentinel)\
            {\
              auto result = ::bra::complex_type{};\
\
              auto const last_index = (::bra::state_integer_type{1u} << num_operated_qubits_);\
              for (auto index = ::bra::state_integer_type{0u}; index < last_index; ++index)\
              {\
                using std::begin;\
                using std::end;\
                auto const iter\
                  = first\
                    + ket::gate::utility::index_with_qubits(\
                        index_wo_qubits, index,\
                        begin(unsorted_qubits), end(unsorted_qubits), begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));\
\
                for (auto const& basis_scalar: pauli_string_space_element)\
                {\
                  auto const other_index_coeff = ket::gate::utility::pauli_index_coeff< ::bra::complex_type >(basis_scalar.first, index);\
                  auto const other_iter\
                    = first\
                      + ket::gate::utility::index_with_qubits(\
                          index_wo_qubits, other_index_coeff.first,\
                          begin(unsorted_qubits), end(unsorted_qubits), begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel));\
\
                  using std::conj;\
                  result += basis_scalar.second * (conj(*iter) * (other_index_coeff.second * *other_iter));\
                }\
              }\
\
              return result;\
            } BOOST_PP_REPEAT_ ## z(num_operated_qubits_, OPERATED_QUBITS, nil));\
      break;\

# else // KET_USE_BIT_MASKS_EXPLICITLY
#   define CASE_N(z, num_operated_qubits_, _) \
     case num_operated_qubits_:\
      result_\
        = ket::ranges::expectation_value(\
            parallel_policy_, data_,\
            [&pauli_string_space_element](\
              auto const first, state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, num_operated_qubits_ > const& qubit_masks,\
              std::array< ::bra::state_integer_type, BOOST_PP_INC(num_operated_qubits_) > const& index_masks)\
            {\
              auto result = ::bra::complex_type{};\
\
              auto const last_index = (::bra::state_integer_type{1u} << num_operated_qubits_);\
              for (auto index = ::bra::state_integer_type{0u}; index < last_index; ++index)\
              {\
                using std::begin;\
                using std::end;\
                auto const iter\
                  = first\
                    + ket::gate::utility::index_with_qubits(\
                        index_wo_qubits, index,\
                        begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));\
\
                for (auto const& basis_scalar: pauli_string_space_element)\
                {\
                  auto const other_index_coeff = ket::gate::utility::pauli_index_coeff< ::bra::complex_type >(basis_scalar.first, index);\
                  auto const other_iter\
                    = first\
                      + ket::gate::utility::index_with_qubits(\
                          index_wo_qubits, other_index_coeff.first,\
                          begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks));\
\
                  using std::conj;\
                  result += basis_scalar.second * (conj(*iter) * (other_index_coeff.second * *other_iter));\
                }\
              }\
\
              return result;\
            } BOOST_PP_REPEAT_ ## z(num_operated_qubits_, OPERATED_QUBITS, nil));\
      break;\

# endif // KET_USE_BIT_MASKS_EXPLICITLY

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef OPERATED_QUBITS
    }
  }

  void nompi_state::do_shor_box(
    state_integer_type const divisor, state_integer_type const base,
    std::vector<qubit_type> const& exponent_qubits,
    std::vector<qubit_type> const& modular_exponentiation_qubits)
  { ket::ranges::shor_box(parallel_policy_, data_, base, divisor, exponent_qubits, modular_exponentiation_qubits); }

  void nompi_state::do_begin_fusion()
  { }

  void nompi_state::do_end_fusion()
  {
# if !(!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    assert(fused_gates_.size() == cache_aware_fused_gates_.size());
# endif // !(!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && defined(KET_USE_ON_CACHE_STATE_VECTOR)))

    // generate fused_control_qubits and fused_qubits from found_qubits_
    auto fused_control_qubits = std::vector< ::bra::control_qubit_type >{};
    fused_control_qubits.reserve(total_num_qubits_);
    auto fused_qubits = std::vector< ::bra::qubit_type >{};
    fused_qubits.reserve(total_num_qubits_);
    for (auto index = ::bra::bit_integer_type{0}; index < total_num_qubits_; ++index)
      switch (found_qubits_[index])
      {
       case ::bra::found_qubit::control_qubit:
        fused_control_qubits.push_back(ket::make_control(ket::make_qubit< ::bra::state_integer_type >(index)));
        break;

       case ::bra::found_qubit::ez_qubit:
       case ::bra::found_qubit::cez_qubit:
       case ::bra::found_qubit::qubit:
        fused_qubits.push_back(ket::make_qubit< ::bra::state_integer_type >(index));
        break;

       case ::bra::found_qubit::not_found:
        break;
      }

    // generate to_qubit_index_in_fused_gates
    auto to_qubit_index_in_fused_gates = std::vector< ::bra::bit_integer_type >(total_num_qubits_);
    using std::begin;
    using std::end;
    std::iota(begin(to_qubit_index_in_fused_gates), end(to_qubit_index_in_fused_gates), ::bra::bit_integer_type{0u});
    auto present_qubit_index = ::bra::bit_integer_type{0u};
    for (auto const fused_qubit: fused_qubits)
      to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(fused_qubit)] = present_qubit_index++;
    for (auto const fused_control_qubit: fused_control_qubits)
      to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(fused_control_qubit.qubit())] = present_qubit_index++;

#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
    constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
    constexpr auto cache_size = ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);

    switch (fused_qubits.size())
    {
# define FUSED_QUBITS(z, n, _) , fused_qubits[n]
# define FUSED_CONTROL_QUBITS(z, n, _) , fused_control_qubits[n]
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION)
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::gate(\
          parallel_policy_, data_,\
          [this, &to_qubit_index_in_fused_gates](\
            auto const first, ::bra::state_integer_type const index_wo_qubits,\
            std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& unsorted_fused_qubits,\
            std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
            int const)\
          {\
            for (auto const& gate_ptr: this->fused_gates_)\
              gate_ptr->call(\
                first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                to_qubit_index_in_fused_gates);\
          } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        break;\

#   else // KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::gate(\
          parallel_policy_, data_,\
          [this, &to_qubit_index_in_fused_gates](\
            auto const first, ::bra::state_integer_type const index_wo_qubits,\
            std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& qubit_masks,\
            std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& index_masks,\
            int const)\
          {\
            for (auto const& gate_ptr: this->fused_gates_)\
              gate_ptr->call(\
                first, index_wo_qubits, qubit_masks, index_masks,\
                to_qubit_index_in_fused_gates);\
          } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        break;\

#   endif // KET_USE_BIT_MASKS_EXPLICITLY
# elif !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        if (data_.size() <= cache_size)\
          ket::gate::nocache::ranges::gate(\
            parallel_policy_, data_,\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else if (ket::utility::all_in_state_vector(num_on_cache_qubits BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil)))\
          ket::gate::cache::all_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else if (ket::utility::none_in_state_vector(num_on_cache_qubits BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil)))\
          ket::gate::cache::none_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else\
          ket::gate::cache::some_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        break;\

#   else // KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        if (data_.size() <= cache_size)\
          ket::gate::nocache::ranges::gate(\
            parallel_policy_, data_,\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& qubit_masks,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, qubit_masks, index_masks,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else if (ket::utility::all_in_state_vector(num_on_cache_qubits BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil)))\
          ket::gate::cache::all_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& qubit_masks,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, qubit_masks, index_masks,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else if (ket::utility::none_in_state_vector(num_on_cache_qubits BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil)))\
          ket::gate::cache::none_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& qubit_masks,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, qubit_masks, index_masks,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else\
          ket::gate::cache::some_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& qubit_masks,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, qubit_masks, index_masks,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        break;\

#   endif // KET_USE_BIT_MASKS_EXPLICITLY
# else
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        if (data_.size() <= cache_size)\
          ket::gate::nocache::ranges::gate(\
            parallel_policy_, data_,\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else if (ket::utility::all_in_state_vector(num_on_cache_qubits BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil)))\
          ket::gate::cache::all_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else if (ket::utility::none_in_state_vector(num_on_cache_qubits BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil)))\
          ket::gate::cache::none_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else\
          ket::gate::cache::some_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        break;\

#   else // KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        if (data_.size() <= cache_size)\
          ket::gate::nocache::ranges::gate(\
            parallel_policy_, data_,\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& qubit_masks,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, qubit_masks, index_masks,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else if (ket::utility::all_in_state_vector(num_on_cache_qubits BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil)))\
          ket::gate::cache::all_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& qubit_masks,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, qubit_masks, index_masks,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else if (ket::utility::none_in_state_vector(num_on_cache_qubits BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil)))\
          ket::gate::cache::none_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& qubit_masks,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, qubit_masks, index_masks,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        else\
          ket::gate::cache::some_on_cache::gate(\
            parallel_policy_, begin(data_), end(data_),\
            [this, &to_qubit_index_in_fused_gates](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) > const& qubit_masks,\
              std::array< ::bra::state_integer_type, BOOST_PP_ADD(num_target_qubits, num_control_qubits) + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->fused_gates_)\
                gate_ptr->call(\
                  first, index_wo_qubits, qubit_masks, index_masks,\
                  to_qubit_index_in_fused_gates);\
            } BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
        break;\

#   endif // KET_USE_BIT_MASKS_EXPLICITLY
# endif
# ifndef BRA_MAX_NUM_FUSED_QUBITS
#   ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
#   else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS 10
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
# endif // BRA_MAX_NUM_FUSED_QUBITS
# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (fused_control_qubits.size())\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(0, BOOST_PP_INC(BOOST_PP_SUB(BRA_MAX_NUM_FUSED_QUBITS, num_target_qubits)), CASE_CN, num_target_qubits)\
      }\
      break;\

     case 0:
      switch (fused_control_qubits.size())
      {
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), CASE_CN, 0)
      }
      break;

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), CASE_N, nil)
# undef CASE_N
# undef CASE_CN
# undef FUSED_CONTROL_QUBITS
# undef FUSED_QUBITS
    }

    fused_gates_.clear();
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    cache_aware_fused_gates_.clear();
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  }

  void nompi_state::do_clear(qubit_type const qubit)
  { ket::gate::ranges::clear(parallel_policy_, data_, qubit); }

  void nompi_state::do_set(qubit_type const qubit)
  { ket::gate::ranges::set(parallel_policy_, data_, qubit); }

  void nompi_state::do_controlled_i_gate(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  { }

  void nompi_state::do_controlled_ic_gate(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  { }

  void nompi_state::do_multi_controlled_in_gate(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  { }

  void nompi_state::do_multi_controlled_ic_gate(std::vector<control_qubit_type> const& control_qubits)
  { }

  void nompi_state::do_controlled_hadamard(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_hadamard<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_hadamard<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::hadamard(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_hadamard(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_hadamard<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_hadamard<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::hadamard(parallel_policy_, data_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_not<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_not<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::not_(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_not(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_not<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_not<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::not_(parallel_policy_, data_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_x<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_x<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::pauli_x(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_pauli_xn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_xn<fused_gate_iterator> >(target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_xn<cache_aware_fused_gate_iterator> >(target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::pauli_x(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_y<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_y<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::pauli_y(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_pauli_yn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_yn<fused_gate_iterator> >(target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_yn<cache_aware_fused_gate_iterator> >(target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::pauli_y(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_z<fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_z<cache_aware_fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::pauli_z(parallel_policy_, data_, control_qubit1, control_qubit2);
  }

  void nompi_state::do_multi_controlled_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_z<fused_gate_iterator> >(control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_z<cache_aware_fused_gate_iterator> >(control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, num_target_qubits) \
       case num_operated_qubits:\
        ket::gate::ranges::pauli_z(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_multi_controlled_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_zn<fused_gate_iterator> >(target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_zn<cache_aware_fused_gate_iterator> >(target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::pauli_z(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_swap<fused_gate_iterator> >(target_qubit1, target_qubit2, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_swap<cache_aware_fused_gate_iterator> >(target_qubit1, target_qubit2, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::swap(parallel_policy_, data_, target_qubit1, target_qubit2 BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_DEC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{2u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_x<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::sqrt_pauli_x(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_x<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_sqrt_pauli_x(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_x<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::sqrt_pauli_x(parallel_policy_, data_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_x<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_sqrt_pauli_x(parallel_policy_, data_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_y<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::sqrt_pauli_y(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_y<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_sqrt_pauli_y(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_y<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::sqrt_pauli_y(parallel_policy_, data_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_y<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_sqrt_pauli_y(parallel_policy_, data_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_sqrt_pauli_z(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_z<fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::sqrt_pauli_z(parallel_policy_, data_, control_qubit1, control_qubit2);
  }

  void nompi_state::do_adj_controlled_sqrt_pauli_z(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_z<fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_sqrt_pauli_z(parallel_policy_, data_, control_qubit1, control_qubit2);
  }

  void nompi_state::do_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_z<fused_gate_iterator> >(control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
       case num_operated_qubits:\
        ket::gate::ranges::sqrt_pauli_z(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_z<fused_gate_iterator> >(control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
       case num_operated_qubits:\
        ket::gate::ranges::adj_sqrt_pauli_z(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_multi_controlled_sqrt_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_zn<fused_gate_iterator> >(target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::sqrt_pauli_z(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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

  void nompi_state::do_adj_multi_controlled_sqrt_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_zn<fused_gate_iterator> >(target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::adj_sqrt_pauli_z(parallel_policy_, data_ BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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

  void nompi_state::do_controlled_phase_shift(
    complex_type const& phase_coefficient,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_phase_shift<fused_gate_iterator> >(
          phase_coefficient, control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_phase_shift<cache_aware_fused_gate_iterator> >(
          phase_coefficient, control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::phase_shift_coeff(parallel_policy_, data_, phase_coefficient, control_qubit1, control_qubit2);
  }

  void nompi_state::do_adj_controlled_phase_shift(
    complex_type const& phase_coefficient,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_phase_shift<fused_gate_iterator> >(
          phase_coefficient, control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_phase_shift<cache_aware_fused_gate_iterator> >(
          phase_coefficient, control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_phase_shift_coeff(parallel_policy_, data_, phase_coefficient, control_qubit1, control_qubit2);
  }

  void nompi_state::do_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_phase_shift<fused_gate_iterator> >(phase_coefficient, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::phase_shift_coeff(parallel_policy_, data_, phase_coefficient BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_phase_shift<fused_gate_iterator> >(phase_coefficient, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 1u);

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_phase_shift_coeff(parallel_policy_, data_, phase_coefficient BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_u1(
    real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u1<fused_gate_iterator> >(
          phase, control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u1<cache_aware_fused_gate_iterator> >(
          phase, control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::phase_shift(parallel_policy_, data_, phase, control_qubit1, control_qubit2);
  }

  void nompi_state::do_adj_controlled_u1(
    real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u1<fused_gate_iterator> >(
          phase, control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u1<cache_aware_fused_gate_iterator> >(
          phase, control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_phase_shift(parallel_policy_, data_, phase, control_qubit1, control_qubit2);
  }

  void nompi_state::do_multi_controlled_u1(
    real_type const phase, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u1<fused_gate_iterator> >(phase, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u1<cache_aware_fused_gate_iterator> >(phase, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::phase_shift(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_u1(
    real_type const phase, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u1<fused_gate_iterator> >(phase, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u1<cache_aware_fused_gate_iterator> >(phase, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::gate::ranges::adj_phase_shift(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void nompi_state::do_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u2<fused_gate_iterator> >(
          phase1, phase2, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u2<cache_aware_fused_gate_iterator> >(
          phase1, phase2, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::phase_shift2(parallel_policy_, data_, phase1, phase2, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u2<fused_gate_iterator> >(
          phase1, phase2, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u2<cache_aware_fused_gate_iterator> >(
          phase1, phase2, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_phase_shift2(parallel_policy_, data_, phase1, phase2, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u2<fused_gate_iterator> >(phase1, phase2, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::phase_shift2(parallel_policy_, data_, phase1, phase2, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u2<fused_gate_iterator> >(phase1, phase2, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_phase_shift2(parallel_policy_, data_, phase1, phase2, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u3<fused_gate_iterator> >(
          phase1, phase2, phase3, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u3<cache_aware_fused_gate_iterator> >(
          phase1, phase2, phase3, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u3<fused_gate_iterator> >(
          phase1, phase2, phase3, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u3<cache_aware_fused_gate_iterator> >(
          phase1, phase2, phase3, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u3<fused_gate_iterator> >(phase1, phase2, phase3, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u3<fused_gate_iterator> >(phase1, phase2, phase3, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_phase_shift3(parallel_policy_, data_, phase1, phase2, phase3, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_x_rotation_half_pi<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::x_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_x_rotation_half_pi<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_x_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_x_rotation_half_pi<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::x_rotation_half_pi(parallel_policy_, data_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_x_rotation_half_pi<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_x_rotation_half_pi(parallel_policy_, data_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_y_rotation_half_pi<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::y_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_y_rotation_half_pi<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_y_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_y_rotation_half_pi<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::y_rotation_half_pi(parallel_policy_, data_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_y_rotation_half_pi<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_y_rotation_half_pi(parallel_policy_, data_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_x<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_x<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::exponential_pauli_x(parallel_policy_, data_, phase, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_exponential_pauli_x(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_x<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_x<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_exponential_pauli_x(parallel_policy_, data_, phase, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_xn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::exponential_pauli_x(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_xn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::adj_exponential_pauli_x(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_y<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_y<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::exponential_pauli_y(parallel_policy_, data_, phase, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_exponential_pauli_y(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_y<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_y<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_exponential_pauli_y(parallel_policy_, data_, phase, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_yn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::exponential_pauli_y(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_yn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::adj_exponential_pauli_y(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_z<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::exponential_pauli_z(parallel_policy_, data_, phase, target_qubit, control_qubit);
  }

  void nompi_state::do_adj_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_z<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::gate::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase, target_qubit, control_qubit);
  }

  void nompi_state::do_multi_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_z<fused_gate_iterator> >(phase, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    constexpr auto num_target_qubits = 1u;
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_operated_qubits > 2u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
       case num_control_qubits:\
        ket::gate::ranges::exponential_pauli_z(parallel_policy_, data_, phase, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_adj_multi_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_z<fused_gate_iterator> >(phase, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    constexpr auto num_target_qubits = 1u;
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_operated_qubits > 2u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
       case num_control_qubits:\
        ket::gate::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void nompi_state::do_multi_controlled_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_zn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::exponential_pauli_z(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_zn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::gate::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_swap<fused_gate_iterator> >(phase, target_qubit1, target_qubit2, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_swap<cache_aware_fused_gate_iterator> >(phase, target_qubit1, target_qubit2, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::exponential_swap(parallel_policy_, data_, phase, target_qubit1, target_qubit2 BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_swap<fused_gate_iterator> >(phase, target_qubit1, target_qubit2, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_swap<cache_aware_fused_gate_iterator> >(phase, target_qubit1, target_qubit2, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::gate::ranges::adj_exponential_swap(parallel_policy_, data_, phase, target_qubit1, target_qubit2 BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
