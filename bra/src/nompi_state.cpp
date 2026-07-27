#ifdef BRA_NO_MPI
# include <cmath>
# include <iostream>
# include <sstream>
# include <vector>
# include <iterator>
# include <algorithm>
# include <numeric>
# include <utility>
# include <type_traits>

# include <boost/algorithm/string/case_conv.hpp>

# include <ket/inner_product.hpp>
# include <ket/fidelity.hpp>
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
# include <bra/utility/closest_floating_point_of.hpp>

namespace bra
{
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  template <typename Iterator, typename CacheAwareIterator>
  struct nompi_fused_gate_caller
  {
    std::vector<std::unique_ptr< ::bra::fused_gate::fused_gate<Iterator> >> const& fused_gates_;
    std::vector<std::unique_ptr< ::bra::fused_gate::fused_gate<CacheAwareIterator> >> const& cache_aware_fused_gates_;
    std::vector< ::bra::bit_integer_type > const& to_qubit_index_in_fused_gates_;

    template <typename First, typename UnsortedFusedQubitsOrMasks, typename SortedFusedQubitsWithSentinelOrIndexMasks>
    auto operator()(
      First const first, ::bra::state_integer_type const index_wo_qubits,
      UnsortedFusedQubitsOrMasks const& unsorted_fused_qubits_or_masks,
      SortedFusedQubitsWithSentinelOrIndexMasks const& sorted_fused_qubits_with_sentinel_or_index_masks,
      int const) const
    -> typename std::enable_if<
         std::is_same<typename std::decay<First>::type, Iterator>::value>::type
    {
      for (auto const& gate_ptr: fused_gates_)
        gate_ptr->call(
          first, index_wo_qubits,
          unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
          to_qubit_index_in_fused_gates_);
    }

    template <typename First, typename UnsortedFusedQubitsOrMasks, typename SortedFusedQubitsWithSentinelOrIndexMasks>
    auto operator()(
      First const first, ::bra::state_integer_type const index_wo_qubits,
      UnsortedFusedQubitsOrMasks const& unsorted_fused_qubits_or_masks,
      SortedFusedQubitsWithSentinelOrIndexMasks const& sorted_fused_qubits_with_sentinel_or_index_masks,
      int const) const
    -> typename std::enable_if<
         std::is_same<typename std::decay<First>::type, CacheAwareIterator>::value>::type
    {
      for (auto const& gate_ptr: cache_aware_fused_gates_)
        gate_ptr->call(
          first, index_wo_qubits,
          unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
          to_qubit_index_in_fused_gates_);
    }
  };
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION)
  nompi_state::nompi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const total_num_qubits,
    unsigned int num_threads, ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    int const circuit_index)
    : ::bra::state{total_num_qubits, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, circuit_index},
      parallel_policy_{num_threads},
      data_{make_initial_data(initial_integer, total_num_qubits)},
      fused_gates_{},
      is_waiting_{false}
  { }
# elif !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  nompi_state::nompi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const total_num_qubits,
    unsigned int num_threads, ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    int const circuit_index)
    : ::bra::state{total_num_qubits, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, circuit_index},
      parallel_policy_{num_threads},
      data_{make_initial_data(initial_integer, total_num_qubits)},
      fused_gates_{},
      cache_aware_fused_gates_{},
      is_waiting_{false}
  { }
# else
#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
  nompi_state::nompi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const total_num_qubits,
    unsigned int num_threads, ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    int const circuit_index)
    : ::bra::state{total_num_qubits, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, circuit_index},
      parallel_policy_{num_threads},
      data_{make_initial_data(initial_integer, total_num_qubits)},
      on_cache_data_{::ket::utility::integer_exp2< ::bra::state_integer_type >(KET_DEFAULT_NUM_ON_CACHE_QUBITS)},
      fused_gates_{},
      is_waiting_{false}
  { }
# endif

  auto nompi_state::do_is_waiting() const -> bool
  { return is_waiting_; }

  auto nompi_state::do_cancel_waiting() -> void
  { is_waiting_ = false; }

  auto nompi_state::generate_probability() -> real_type
  {
    using floating_point_type = typename ::bra::utility::closest_floating_point_of<real_type>::type;
    auto distribution = std::uniform_real_distribution<floating_point_type>{0.0, 1.0};

    return
      uses_depolarizing_seed_
      ? static_cast<real_type>(distribution(depolarizing_random_number_generator_))
      : static_cast<real_type>(distribution(random_number_generator_));
  }

  auto nompi_state::do_send_real_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void
  {
    if (destination_circuit_index < 0 or destination_circuit_index == circuit_index_)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::send_real_variable_t{}, destination_circuit_index, variable_name, num_elements};
  }

  auto nompi_state::do_send_complex_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void
  {
    if (destination_circuit_index < 0 or destination_circuit_index == circuit_index_)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::send_complex_variable_t{}, destination_circuit_index, variable_name, num_elements};
  }

  auto nompi_state::do_send_int_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void
  {
    if (destination_circuit_index < 0 or destination_circuit_index == circuit_index_)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::send_int_variable_t{}, destination_circuit_index, variable_name, num_elements};
  }

  auto nompi_state::do_receive_real_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (destination_circuit_index < 0 or destination_circuit_index == circuit_index_)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::receive_real_variable_t{}, destination_circuit_index, variable_name, num_elements};
  }

  auto nompi_state::do_receive_complex_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (destination_circuit_index < 0 or destination_circuit_index == circuit_index_)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::receive_complex_variable_t{}, destination_circuit_index, variable_name, num_elements};
  }

  auto nompi_state::do_receive_int_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (destination_circuit_index < 0 or destination_circuit_index == circuit_index_)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::receive_int_variable_t{}, destination_circuit_index, variable_name, num_elements};
  }

  auto nompi_state::do_broadcast_real_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (root_circuit_index < 0)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::broadcast_real_variable_t{}, root_circuit_index, variable_name, num_elements};
  }

  auto nompi_state::do_broadcast_complex_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (root_circuit_index < 0)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::broadcast_complex_variable_t{}, root_circuit_index, variable_name, num_elements};
  }

  auto nompi_state::do_broadcast_int_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (root_circuit_index < 0)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::broadcast_int_variable_t{}, root_circuit_index, variable_name, num_elements};
  }

  auto nompi_state::do_gather_real_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& destination_variable_name) -> void
  {
    if (root_circuit_index < 0)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::gather_real_variable_t{}, root_circuit_index, variable_name, num_elements, destination_variable_name};
  }

  auto nompi_state::do_gather_complex_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& destination_variable_name) -> void
  {
    if (root_circuit_index < 0)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::gather_complex_variable_t{}, root_circuit_index, variable_name, num_elements, destination_variable_name};
  }

  auto nompi_state::do_gather_int_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& destination_variable_name) -> void
  {
    if (root_circuit_index < 0)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::gather_int_variable_t{}, root_circuit_index, variable_name, num_elements, destination_variable_name};
  }

  auto nompi_state::do_scatter_real_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& source_variable_name) -> void
  {
    if (root_circuit_index < 0)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::scatter_real_variable_t{}, root_circuit_index, variable_name, num_elements, source_variable_name};
  }

  auto nompi_state::do_scatter_complex_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& source_variable_name) -> void
  {
    if (root_circuit_index < 0)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::scatter_complex_variable_t{}, root_circuit_index, variable_name, num_elements, source_variable_name};
  }

  auto nompi_state::do_scatter_int_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& source_variable_name) -> void
  {
    if (root_circuit_index < 0)
      return;

    is_waiting_ = true;
    wait_reason_ = ::bra::wait_reason{::bra::wait_reason::scatter_int_variable_t{}, root_circuit_index, variable_name, num_elements, source_variable_name};
  }

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

    assert(qubits.size() > 2u);

    ket::gate::runtime::ranges::pauli_x(parallel_policy_, data_, qubits);
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

    assert(qubits.size() > 2u);

    ket::gate::runtime::ranges::pauli_y(parallel_policy_, data_, qubits);
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

    assert(qubits.size() > 2u);

    ket::gate::runtime::ranges::pauli_z(parallel_policy_, data_, qubits);
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

    assert(qubits.size() > 2u);

    ket::gate::runtime::ranges::sqrt_pauli_z(parallel_policy_, data_, qubits);
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

    assert(qubits.size() > 2u);

    ket::gate::runtime::ranges::adj_sqrt_pauli_z(parallel_policy_, data_, qubits);
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

    assert(qubits.size() > 2u);

    ket::gate::runtime::ranges::exponential_pauli_x(parallel_policy_, data_, phase, qubits);
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

    assert(qubits.size() > 2u);

    ket::gate::runtime::ranges::adj_exponential_pauli_x(parallel_policy_, data_, phase, qubits);
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

    assert(qubits.size() > 2u);

    ket::gate::runtime::ranges::exponential_pauli_y(parallel_policy_, data_, phase, qubits);
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

    assert(qubits.size() > 2u);

    ket::gate::runtime::ranges::adj_exponential_pauli_y(parallel_policy_, data_, phase, qubits);
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

    assert(qubits.size() > 2u);

    ket::gate::runtime::ranges::exponential_pauli_z(parallel_policy_, data_, phase, qubits);
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

    assert(qubits.size() > 2u);

    ket::gate::runtime::ranges::adj_exponential_pauli_z(parallel_policy_, data_, phase, qubits);
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

  void nompi_state::do_amplitudes(std::vector< ::bra::state_integer_type > const& amplitude_indices)
  {
    std::ostringstream oss;

    if (amplitude_indices.empty())
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
    else
      for (auto const amplitude_index: amplitude_indices)
      {
        auto const& amplitude = data_[amplitude_index];
        using std::real;
        using std::imag;
        oss << ::bra::state_detail::integer_to_bits_string(amplitude_index, total_num_qubits_) << " => " << real(amplitude) << " + " << imag(amplitude) << " i\n";
      }

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

    result_
      = ket::runtime::ranges::expectation_value(
          parallel_policy_, data_,
          [&pauli_string_space_element, num_operated_qubits](
            auto const first, auto const index_wo_qubits,
            auto const& unsorted_qubits_or_masks, auto const& sorted_qubits_or_index_masks)
          {
            auto result = ::bra::complex_type{};

            auto const last_index = (::bra::state_integer_type{1u} << num_operated_qubits);
            for (auto index = ::bra::state_integer_type{0u}; index < last_index; ++index)
            {
              using std::begin;
              using std::end;
              auto const iter
                = first
                  + ket::gate::utility::index_with_qubits(
                      index_wo_qubits, index,
                      begin(unsorted_qubits_or_masks), end(unsorted_qubits_or_masks),
                      begin(sorted_qubits_or_index_masks), end(sorted_qubits_or_index_masks));

              for (auto const& basis_scalar: pauli_string_space_element)
              {
                auto const other_index_coeff = ket::gate::utility::pauli_index_coeff< ::bra::complex_type >(basis_scalar.first, index);
                auto const other_iter
                  = first
                    + ket::gate::utility::index_with_qubits(
                        index_wo_qubits, other_index_coeff.first,
                        begin(unsorted_qubits_or_masks), end(unsorted_qubits_or_masks),
                        begin(sorted_qubits_or_index_masks), end(sorted_qubits_or_index_masks));

                using std::conj;
                result += basis_scalar.second * (conj(*iter) * (other_index_coeff.second * *other_iter));
              }
            }

            return result;
          },
          operated_qubits);
  }

  void nompi_state::do_inner_product(std::string const& remote_circuit_index_or_all)
  {
    if (std::isdigit(static_cast<unsigned char>(remote_circuit_index_or_all.front())))
    {
      auto const remote_circuit_index = boost::lexical_cast<int>(remote_circuit_index_or_all);
      if (remote_circuit_index < 0 or remote_circuit_index == circuit_index_)
        return;

      is_waiting_ = true;
      wait_reason_ = ::bra::wait_reason{::bra::wait_reason::inner_product_t{}, remote_circuit_index};
    }
    else if (boost::algorithm::to_upper_copy(remote_circuit_index_or_all) == "ALL")
    {
      is_waiting_ = true;
      wait_reason_ = ::bra::wait_reason{::bra::wait_reason::inner_product_all_t{}};
    }
  }

  void nompi_state::do_inner_product(std::string const& remote_circuit_index_or_all, std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits)
  {
    if (std::isdigit(static_cast<unsigned char>(remote_circuit_index_or_all.front())))
    {
      auto const remote_circuit_index = boost::lexical_cast<int>(remote_circuit_index_or_all);
      if (remote_circuit_index < 0 or remote_circuit_index == circuit_index_)
        return;

      is_waiting_ = true;
      wait_reason_ = ::bra::wait_reason{::bra::wait_reason::inner_product_op_t{}, remote_circuit_index, operator_literal_or_variable_name, operated_qubits};
    }
    else if (boost::algorithm::to_upper_copy(remote_circuit_index_or_all) == "ALL")
    {
      is_waiting_ = true;
      wait_reason_ = ::bra::wait_reason{::bra::wait_reason::inner_product_all_op_t{}, operator_literal_or_variable_name, operated_qubits};
    }
  }

  void nompi_state::do_fidelity(std::string const& remote_circuit_index_or_all)
  {
    if (std::isdigit(static_cast<unsigned char>(remote_circuit_index_or_all.front())))
    {
      auto const remote_circuit_index = boost::lexical_cast<int>(remote_circuit_index_or_all);
      if (remote_circuit_index < 0 or remote_circuit_index == circuit_index_)
        return;

      is_waiting_ = true;
      wait_reason_ = ::bra::wait_reason{::bra::wait_reason::fidelity_t{}, remote_circuit_index};
    }
    else if (boost::algorithm::to_upper_copy(remote_circuit_index_or_all) == "ALL")
    {
      is_waiting_ = true;
      wait_reason_ = ::bra::wait_reason{::bra::wait_reason::fidelity_all_t{}};
    }
  }

  void nompi_state::do_fidelity(std::string const& remote_circuit_index_or_all, std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits)
  {
    if (std::isdigit(static_cast<unsigned char>(remote_circuit_index_or_all.front())))
    {
      auto const remote_circuit_index = boost::lexical_cast<int>(remote_circuit_index_or_all);
      if (remote_circuit_index < 0 or remote_circuit_index == circuit_index_)
        return;

      is_waiting_ = true;
      wait_reason_ = ::bra::wait_reason{::bra::wait_reason::fidelity_op_t{}, remote_circuit_index, operator_literal_or_variable_name, operated_qubits};
    }
    else if (boost::algorithm::to_upper_copy(remote_circuit_index_or_all) == "ALL")
    {
      is_waiting_ = true;
      wait_reason_ = ::bra::wait_reason{::bra::wait_reason::fidelity_all_op_t{}, operator_literal_or_variable_name, operated_qubits};
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

    auto operated_qubits = std::vector< ::bra::qubit_type >{};
    operated_qubits.reserve(fused_qubits.size() + fused_control_qubits.size());
    operated_qubits.insert(end(operated_qubits), begin(fused_qubits), end(fused_qubits));
    for (auto const fused_control_qubit: fused_control_qubits)
      operated_qubits.push_back(fused_control_qubit.qubit());

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || defined(KET_USE_ON_CACHE_STATE_VECTOR)
    auto const call_fused_gates
      = [this, &to_qubit_index_in_fused_gates](
          auto const first, ::bra::state_integer_type const index_wo_qubits,
          auto const& unsorted_fused_qubits_or_masks,
          auto const& sorted_fused_qubits_with_sentinel_or_index_masks,
          int const)
        {
          for (auto const& gate_ptr: this->fused_gates_)
            gate_ptr->call(
              first, index_wo_qubits,
              unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
              to_qubit_index_in_fused_gates);
        };

# else // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    auto const call_fused_gates
      = nompi_fused_gate_caller<fused_gate_iterator, cache_aware_fused_gate_iterator>{
          fused_gates_, cache_aware_fused_gates_, to_qubit_index_in_fused_gates};
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

    ket::gate::runtime::ranges::gate(parallel_policy_, data_, call_fused_gates, operated_qubits);

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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::hadamard(parallel_policy_, data_, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::not_(parallel_policy_, data_, target_qubit, control_qubits);
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

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::pauli_x(parallel_policy_, data_, target_qubits, control_qubits);
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

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::pauli_y(parallel_policy_, data_, target_qubits, control_qubits);
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

    assert(control_qubits.size() > 2u);

    ket::gate::runtime::ranges::pauli_z(parallel_policy_, data_, control_qubits);
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

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::pauli_z(parallel_policy_, data_, target_qubits, control_qubits);
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

    assert(control_qubits.size() > 0u);

    ket::gate::runtime::ranges::swap(parallel_policy_, data_, target_qubit1, target_qubit2, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::sqrt_pauli_x(parallel_policy_, data_, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::adj_sqrt_pauli_x(parallel_policy_, data_, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::sqrt_pauli_y(parallel_policy_, data_, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::adj_sqrt_pauli_y(parallel_policy_, data_, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 2u);

    ket::gate::runtime::ranges::sqrt_pauli_z(parallel_policy_, data_, control_qubits);
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

    assert(control_qubits.size() > 2u);

    ket::gate::runtime::ranges::adj_sqrt_pauli_z(parallel_policy_, data_, control_qubits);
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

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::sqrt_pauli_z(parallel_policy_, data_, target_qubits, control_qubits);
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

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::adj_sqrt_pauli_z(parallel_policy_, data_, target_qubits, control_qubits);
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

    assert(control_qubits.size() > 2u);

    ket::gate::runtime::ranges::phase_shift_coeff(
      parallel_policy_, data_, phase_coefficient, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::adj_phase_shift_coeff(
      parallel_policy_, data_, phase_coefficient, control_qubits);
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

    assert(control_qubits.size() > 2u);

    ket::gate::runtime::ranges::phase_shift(
      parallel_policy_, data_, phase, control_qubits);
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

    assert(control_qubits.size() > 2u);

    ket::gate::runtime::ranges::adj_phase_shift(
      parallel_policy_, data_, phase, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::phase_shift2(
      parallel_policy_, data_, phase1, phase2, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::adj_phase_shift2(
      parallel_policy_, data_, phase1, phase2, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::phase_shift3(
      parallel_policy_, data_, phase1, phase2, phase3, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::adj_phase_shift3(
      parallel_policy_, data_, phase1, phase2, phase3, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::x_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::adj_x_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::y_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubits);
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

    assert(control_qubits.size() > 1u);

    ket::gate::runtime::ranges::adj_y_rotation_half_pi(parallel_policy_, data_, target_qubit, control_qubits);
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

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::exponential_pauli_x(
      parallel_policy_, data_, phase, target_qubits, control_qubits);
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

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::adj_exponential_pauli_x(
      parallel_policy_, data_, phase, target_qubits, control_qubits);
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

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::exponential_pauli_y(
      parallel_policy_, data_, phase, target_qubits, control_qubits);
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

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::adj_exponential_pauli_y(
      parallel_policy_, data_, phase, target_qubits, control_qubits);
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

    assert(1u + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::exponential_pauli_z(
      parallel_policy_, data_, phase,
      boost::make_iterator_range(&target_qubit, &target_qubit + 1), control_qubits);
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

    assert(1u + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::adj_exponential_pauli_z(
      parallel_policy_, data_, phase,
      boost::make_iterator_range(&target_qubit, &target_qubit + 1), control_qubits);
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

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::exponential_pauli_z(
      parallel_policy_, data_, phase, target_qubits, control_qubits);
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

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    ket::gate::runtime::ranges::adj_exponential_pauli_z(
      parallel_policy_, data_, phase, target_qubits, control_qubits);
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

    assert(control_qubits.size() > 0u);

    ket::gate::runtime::ranges::exponential_swap(
      parallel_policy_, data_, phase, target_qubit1, target_qubit2, control_qubits);
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

    assert(control_qubits.size() > 0u);

    ket::gate::runtime::ranges::adj_exponential_swap(
      parallel_policy_, data_, phase, target_qubit1, target_qubit2, control_qubits);
  }

  void inner_product(::bra::nompi_state& state1, ::bra::nompi_state& state2)
  {
    auto const result
      = ket::ranges::inner_product(state1.parallel_policy_, state1.data_, state2.data_);
    state1.result_ = result;
    using std::conj;
    state2.result_ = conj(result);
  }

  void inner_product_all(std::vector< ::bra::nompi_state >& states)
  {
    using std::begin;
    using std::end;
    auto const state_first = begin(states);
    auto const state_last = end(states);
    for (auto iter = state_first; iter != state_last; ++iter)
      iter->result_
        = ket::ranges::inner_product(state_first->parallel_policy_, state_first->data_, iter->data_);
  }

  void inner_product_op(
    ::bra::nompi_state& state1, ::bra::nompi_state& state2,
    std::string const& operator_literal_or_variable_name,
    std::vector< ::bra::qubit_type > const& operated_qubits)
  {
    auto const num_operated_qubits = operated_qubits.size();
    if (num_operated_qubits > state1.total_num_qubits_)
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, state1.total_num_qubits_};

    auto const pauli_string_space_element = state1.to_pauli_string_space(operator_literal_or_variable_name);

    if (num_operated_qubits != pauli_string_space_element.num_qubits())
      throw ::bra::wrong_pauli_string_length_error{num_operated_qubits, pauli_string_space_element.num_qubits()};

    auto const result
      = ket::runtime::ranges::inner_product(
          state1.parallel_policy_, state1.data_, state2.data_,
          [&pauli_string_space_element, num_operated_qubits](
            auto const ket_first, auto const bra_first, auto const index_wo_qubits,
            auto const& unsorted_qubits_or_masks, auto const& sorted_qubits_or_index_masks)
          {
            auto result = ::bra::complex_type{};

            auto const last_index = (::bra::state_integer_type{1u} << num_operated_qubits);
            for (auto bra_index = ::bra::state_integer_type{0u}; bra_index < last_index; ++bra_index)
            {
              using std::begin;
              using std::end;
              auto const bra_iter
                = bra_first
                  + ket::gate::utility::index_with_qubits(
                      index_wo_qubits, bra_index,
                      begin(unsorted_qubits_or_masks), end(unsorted_qubits_or_masks),
                      begin(sorted_qubits_or_index_masks), end(sorted_qubits_or_index_masks));

              for (auto const& basis_scalar: pauli_string_space_element)
              {
                auto const ket_index_coeff = ket::gate::utility::pauli_index_coeff< ::bra::complex_type >(basis_scalar.first, bra_index);
                auto const ket_iter
                  = ket_first
                    + ket::gate::utility::index_with_qubits(
                        index_wo_qubits, ket_index_coeff.first,
                        begin(unsorted_qubits_or_masks), end(unsorted_qubits_or_masks),
                        begin(sorted_qubits_or_index_masks), end(sorted_qubits_or_index_masks));

                using std::conj;
                result += basis_scalar.second * (conj(*bra_iter) * (ket_index_coeff.second * *ket_iter));
              }
            }

            return result;
          },
          operated_qubits);
    state1.result_ = result;
    using std::conj;
    state2.result_ = conj(result);
  }

  void inner_product_all_op(
    std::vector< ::bra::nompi_state >& states,
    std::string const& operator_literal_or_variable_name,
    std::vector< ::bra::qubit_type > const& operated_qubits)
  {
    using std::begin;
    using std::end;
    auto const state_first = begin(states);
    auto const state_last = end(states);

    auto const num_operated_qubits = operated_qubits.size();
    if (num_operated_qubits > state_first->total_num_qubits_)
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, state_first->total_num_qubits_};

    auto const pauli_string_space_element = state_first->to_pauli_string_space(operator_literal_or_variable_name);

    if (num_operated_qubits != pauli_string_space_element.num_qubits())
      throw ::bra::wrong_pauli_string_length_error{num_operated_qubits, pauli_string_space_element.num_qubits()};

    for (auto iter = state_first; iter != state_last; ++iter)
      iter->result_
        = ket::runtime::ranges::inner_product(
            state_first->parallel_policy_, state_first->data_, iter->data_,
            [&pauli_string_space_element, num_operated_qubits](
              auto const ket_first, auto const bra_first, auto const index_wo_qubits,
              auto const& unsorted_qubits_or_masks, auto const& sorted_qubits_or_index_masks)
            {
              auto result = ::bra::complex_type{};

              auto const last_index = (::bra::state_integer_type{1u} << num_operated_qubits);
              for (auto bra_index = ::bra::state_integer_type{0u}; bra_index < last_index; ++bra_index)
              {
                using std::begin;
                using std::end;
                auto const bra_iter
                  = bra_first
                    + ket::gate::utility::index_with_qubits(
                        index_wo_qubits, bra_index,
                        begin(unsorted_qubits_or_masks), end(unsorted_qubits_or_masks),
                        begin(sorted_qubits_or_index_masks), end(sorted_qubits_or_index_masks));

                for (auto const& basis_scalar: pauli_string_space_element)
                {
                  auto const ket_index_coeff = ket::gate::utility::pauli_index_coeff< ::bra::complex_type >(basis_scalar.first, bra_index);
                  auto const ket_iter
                    = ket_first
                      + ket::gate::utility::index_with_qubits(
                          index_wo_qubits, ket_index_coeff.first,
                          begin(unsorted_qubits_or_masks), end(unsorted_qubits_or_masks),
                          begin(sorted_qubits_or_index_masks), end(sorted_qubits_or_index_masks));

                  using std::conj;
                  result += basis_scalar.second * (conj(*bra_iter) * (ket_index_coeff.second * *ket_iter));
                }
              }

              return result;
            },
            operated_qubits);
  }

  void fidelity(::bra::nompi_state& state1, ::bra::nompi_state& state2)
  {
    auto const result
      = ket::ranges::fidelity(state1.parallel_policy_, state1.data_, state2.data_);
    state1.result_ = result;
    using std::conj;
    state2.result_ = conj(result);
  }

  void fidelity_all(std::vector< ::bra::nompi_state >& states)
  {
    using std::begin;
    using std::end;
    auto const state_first = begin(states);
    auto const state_last = end(states);
    for (auto iter = state_first; iter != state_last; ++iter)
      iter->result_
        = ket::ranges::fidelity(state_first->parallel_policy_, state_first->data_, iter->data_);
  }

  void fidelity_op(
    ::bra::nompi_state& state1, ::bra::nompi_state& state2,
    std::string const& operator_literal_or_variable_name,
    std::vector< ::bra::qubit_type > const& operated_qubits)
  {
    auto const num_operated_qubits = operated_qubits.size();
    if (num_operated_qubits > state1.total_num_qubits_)
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, state1.total_num_qubits_};

    auto const pauli_string_space_element = state1.to_pauli_string_space(operator_literal_or_variable_name);

    if (num_operated_qubits != pauli_string_space_element.num_qubits())
      throw ::bra::wrong_pauli_string_length_error{num_operated_qubits, pauli_string_space_element.num_qubits()};

    auto const result
      = ket::runtime::ranges::fidelity(
          state1.parallel_policy_, state1.data_, state2.data_,
          [&pauli_string_space_element, num_operated_qubits](
            auto const ket_first, auto const bra_first, auto const index_wo_qubits,
            auto const& unsorted_qubits_or_masks, auto const& sorted_qubits_or_index_masks)
          {
            auto result = ::bra::complex_type{};

            auto const last_index = (::bra::state_integer_type{1u} << num_operated_qubits);
            for (auto bra_index = ::bra::state_integer_type{0u}; bra_index < last_index; ++bra_index)
            {
              using std::begin;
              using std::end;
              auto const bra_iter
                = bra_first
                  + ket::gate::utility::index_with_qubits(
                      index_wo_qubits, bra_index,
                      begin(unsorted_qubits_or_masks), end(unsorted_qubits_or_masks),
                      begin(sorted_qubits_or_index_masks), end(sorted_qubits_or_index_masks));

              for (auto const& basis_scalar: pauli_string_space_element)
              {
                auto const ket_index_coeff = ket::gate::utility::pauli_index_coeff< ::bra::complex_type >(basis_scalar.first, bra_index);
                auto const ket_iter
                  = ket_first
                    + ket::gate::utility::index_with_qubits(
                        index_wo_qubits, ket_index_coeff.first,
                        begin(unsorted_qubits_or_masks), end(unsorted_qubits_or_masks),
                        begin(sorted_qubits_or_index_masks), end(sorted_qubits_or_index_masks));

                using std::conj;
                result += basis_scalar.second * (conj(*bra_iter) * (ket_index_coeff.second * *ket_iter));
              }
            }

            return result;
          },
          operated_qubits);
    state1.result_ = result;
    using std::conj;
    state2.result_ = conj(result);
  }

  void fidelity_all_op(
    std::vector< ::bra::nompi_state >& states,
    std::string const& operator_literal_or_variable_name,
    std::vector< ::bra::qubit_type > const& operated_qubits)
  {
    using std::begin;
    using std::end;
    auto const state_first = begin(states);
    auto const state_last = end(states);

    auto const num_operated_qubits = operated_qubits.size();
    if (num_operated_qubits > state_first->total_num_qubits_)
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, state_first->total_num_qubits_};

    auto const pauli_string_space_element = state_first->to_pauli_string_space(operator_literal_or_variable_name);

    if (num_operated_qubits != pauli_string_space_element.num_qubits())
      throw ::bra::wrong_pauli_string_length_error{num_operated_qubits, pauli_string_space_element.num_qubits()};

    for (auto iter = state_first; iter != state_last; ++iter)
      iter->result_
        = ket::runtime::ranges::fidelity(
            state_first->parallel_policy_, state_first->data_, iter->data_,
            [&pauli_string_space_element, num_operated_qubits](
              auto const ket_first, auto const bra_first, auto const index_wo_qubits,
              auto const& unsorted_qubits_or_masks, auto const& sorted_qubits_or_index_masks)
            {
              auto result = ::bra::complex_type{};

              auto const last_index = (::bra::state_integer_type{1u} << num_operated_qubits);
              for (auto bra_index = ::bra::state_integer_type{0u}; bra_index < last_index; ++bra_index)
              {
                using std::begin;
                using std::end;
                auto const bra_iter
                  = bra_first
                    + ket::gate::utility::index_with_qubits(
                        index_wo_qubits, bra_index,
                        begin(unsorted_qubits_or_masks), end(unsorted_qubits_or_masks),
                        begin(sorted_qubits_or_index_masks), end(sorted_qubits_or_index_masks));

                for (auto const& basis_scalar: pauli_string_space_element)
                {
                  auto const ket_index_coeff = ket::gate::utility::pauli_index_coeff< ::bra::complex_type >(basis_scalar.first, bra_index);
                  auto const ket_iter
                    = ket_first
                      + ket::gate::utility::index_with_qubits(
                          index_wo_qubits, ket_index_coeff.first,
                          begin(unsorted_qubits_or_masks), end(unsorted_qubits_or_masks),
                          begin(sorted_qubits_or_index_masks), end(sorted_qubits_or_index_masks));

                  using std::conj;
                  result += basis_scalar.second * (conj(*bra_iter) * (ket_index_coeff.second * *ket_iter));
                }
              }

              return result;
            },
            operated_qubits);
  }

  void send_real_variable(
    nompi_state const& source_state, std::string const& source_variable_name,
    nompi_state& destination_state, std::string const& destination_variable_name,
    int const num_elements)
  {
    auto const& source_real_variable = source_state.to_real_variable(source_variable_name);
    auto& destination_real_variable = destination_state.to_real_variable(destination_variable_name);

    std::copy(
      std::addressof(source_real_variable), std::addressof(source_real_variable) + num_elements,
      std::addressof(destination_real_variable));
  }

  void send_complex_variable(
    nompi_state const& source_state, std::string const& source_variable_name,
    nompi_state& destination_state, std::string const& destination_variable_name,
    int const num_elements)
  {
    auto const& source_complex_variable = source_state.to_complex_variable(source_variable_name);
    auto& destination_complex_variable = destination_state.to_complex_variable(destination_variable_name);

    std::copy(
      std::addressof(source_complex_variable), std::addressof(source_complex_variable) + num_elements,
      std::addressof(destination_complex_variable));
  }

  void send_int_variable(
    nompi_state const& source_state, std::string const& source_variable_name,
    nompi_state& destination_state, std::string const& destination_variable_name,
    int const num_elements)
  {
    auto const& source_int_variable = source_state.to_int_variable(source_variable_name);
    auto& destination_int_variable = destination_state.to_int_variable(destination_variable_name);

    std::copy(
      std::addressof(source_int_variable), std::addressof(source_int_variable) + num_elements,
      std::addressof(destination_int_variable));
  }

  void broadcast_real_variable(
    std::vector< ::bra::nompi_state >& states,
    std::vector<std::string> const& variable_names,
    int const root_circuit_index, int const num_elements)
  {
    auto const& root_real_variable = states[root_circuit_index].to_real_variable(variable_names[root_circuit_index]);

    auto const num_circuits = static_cast<int>(states.size());
    for (auto circuit_index = 0; circuit_index < num_circuits; ++circuit_index)
    {
      if (circuit_index == root_circuit_index)
        continue;

      auto& destination_real_variable = states[circuit_index].to_real_variable(variable_names[circuit_index]);
      std::copy(
        std::addressof(root_real_variable), std::addressof(root_real_variable) + num_elements,
        std::addressof(destination_real_variable));
    }
  }

  void broadcast_complex_variable(
    std::vector< ::bra::nompi_state >& states,
    std::vector<std::string> const& variable_names,
    int const root_circuit_index, int const num_elements)
  {
    auto const& root_complex_variable = states[root_circuit_index].to_complex_variable(variable_names[root_circuit_index]);

    auto const num_circuits = static_cast<int>(states.size());
    for (auto circuit_index = 0; circuit_index < num_circuits; ++circuit_index)
    {
      if (circuit_index == root_circuit_index)
        continue;

      auto& destination_complex_variable = states[circuit_index].to_complex_variable(variable_names[circuit_index]);
      std::copy(
        std::addressof(root_complex_variable), std::addressof(root_complex_variable) + num_elements,
        std::addressof(destination_complex_variable));
    }
  }

  void broadcast_int_variable(
    std::vector< ::bra::nompi_state >& states,
    std::vector<std::string> const& variable_names,
    int const root_circuit_index, int const num_elements)
  {
    auto const& root_int_variable = states[root_circuit_index].to_int_variable(variable_names[root_circuit_index]);

    auto const num_circuits = static_cast<int>(states.size());
    for (auto circuit_index = 0; circuit_index < num_circuits; ++circuit_index)
    {
      if (circuit_index == root_circuit_index)
        continue;

      auto& destination_int_variable = states[circuit_index].to_int_variable(variable_names[circuit_index]);
      std::copy(
        std::addressof(root_int_variable), std::addressof(root_int_variable) + num_elements,
        std::addressof(destination_int_variable));
    }
  }

  void gather_real_variable(
    std::vector< ::bra::nompi_state >& states,
    std::vector<std::string> const& variable_names,
    std::string const& destination_variable_name,
    int const root_circuit_index, int const num_elements)
  {
    auto is_destination_variable_name_specified = destination_variable_name != "";
    auto& destination_real_variable
      = is_destination_variable_name_specified
        ? states[root_circuit_index].to_real_variable(destination_variable_name)
        : states[root_circuit_index].to_real_variable(variable_names[root_circuit_index]);

    auto const num_circuits = static_cast<int>(states.size());
    for (auto circuit_index = 0; circuit_index < num_circuits; ++circuit_index)
    {
      if (circuit_index == root_circuit_index and not is_destination_variable_name_specified)
        continue;

      auto const& source_real_variable = states[circuit_index].to_real_variable(variable_names[circuit_index]);
      std::copy(
        std::addressof(source_real_variable), std::addressof(source_real_variable) + num_elements,
        std::addressof(destination_real_variable) + num_elements * circuit_index);
    }
  }

  void gather_complex_variable(
    std::vector< ::bra::nompi_state >& states,
    std::vector<std::string> const& variable_names,
    std::string const& destination_variable_name,
    int const root_circuit_index, int const num_elements)
  {
    auto is_destination_variable_name_specified = destination_variable_name != "";
    auto& destination_complex_variable
      = is_destination_variable_name_specified
        ? states[root_circuit_index].to_complex_variable(destination_variable_name)
        : states[root_circuit_index].to_complex_variable(variable_names[root_circuit_index]);

    auto const num_circuits = static_cast<int>(states.size());
    for (auto circuit_index = 0; circuit_index < num_circuits; ++circuit_index)
    {
      if (circuit_index == root_circuit_index and not is_destination_variable_name_specified)
        continue;

      auto const& source_complex_variable = states[circuit_index].to_complex_variable(variable_names[circuit_index]);
      std::copy(
        std::addressof(source_complex_variable), std::addressof(source_complex_variable) + num_elements,
        std::addressof(destination_complex_variable) + num_elements * circuit_index);
    }
  }

  void gather_int_variable(
    std::vector< ::bra::nompi_state >& states,
    std::vector<std::string> const& variable_names,
    std::string const& destination_variable_name,
    int const root_circuit_index, int const num_elements)
  {
    auto is_destination_variable_name_specified = destination_variable_name != "";
    auto& destination_int_variable
      = is_destination_variable_name_specified
        ? states[root_circuit_index].to_int_variable(destination_variable_name)
        : states[root_circuit_index].to_int_variable(variable_names[root_circuit_index]);

    auto const num_circuits = static_cast<int>(states.size());
    for (auto circuit_index = 0; circuit_index < num_circuits; ++circuit_index)
    {
      if (circuit_index == root_circuit_index and not is_destination_variable_name_specified)
        continue;

      auto const& source_int_variable = states[circuit_index].to_int_variable(variable_names[circuit_index]);
      std::copy(
        std::addressof(source_int_variable), std::addressof(source_int_variable) + num_elements,
        std::addressof(destination_int_variable) + num_elements * circuit_index);
    }
  }

  void scatter_real_variable(
    std::vector< ::bra::nompi_state >& states,
    std::vector<std::string> const& variable_names,
    std::string const& source_variable_name,
    int const root_circuit_index, int const num_elements)
  {
    auto is_source_variable_name_specified = source_variable_name != "";
    auto const& source_real_variable
      = is_source_variable_name_specified
        ? states[root_circuit_index].to_real_variable(source_variable_name)
        : states[root_circuit_index].to_real_variable(variable_names[root_circuit_index]);

    auto const num_circuits = static_cast<int>(states.size());
    for (auto circuit_index = 0; circuit_index < num_circuits; ++circuit_index)
    {
      if (circuit_index == root_circuit_index and not is_source_variable_name_specified)
        continue;

      auto& destination_real_variable = states[circuit_index].to_real_variable(variable_names[circuit_index]);
      std::copy(
        std::addressof(source_real_variable) + num_elements * circuit_index,
        std::addressof(source_real_variable) + num_elements * circuit_index + num_elements,
        std::addressof(destination_real_variable));
    }
  }

  void scatter_complex_variable(
    std::vector< ::bra::nompi_state >& states,
    std::vector<std::string> const& variable_names,
    std::string const& source_variable_name,
    int const root_circuit_index, int const num_elements)
  {
    auto is_source_variable_name_specified = source_variable_name != "";
    auto const& source_complex_variable
      = is_source_variable_name_specified
        ? states[root_circuit_index].to_complex_variable(source_variable_name)
        : states[root_circuit_index].to_complex_variable(variable_names[root_circuit_index]);

    auto const num_circuits = static_cast<int>(states.size());
    for (auto circuit_index = 0; circuit_index < num_circuits; ++circuit_index)
    {
      if (circuit_index == root_circuit_index and not is_source_variable_name_specified)
        continue;

      auto& destination_complex_variable = states[circuit_index].to_complex_variable(variable_names[circuit_index]);
      std::copy(
        std::addressof(source_complex_variable) + num_elements * circuit_index,
        std::addressof(source_complex_variable) + num_elements * circuit_index + num_elements,
        std::addressof(destination_complex_variable));
    }
  }

  void scatter_int_variable(
    std::vector< ::bra::nompi_state >& states,
    std::vector<std::string> const& variable_names,
    std::string const& source_variable_name,
    int const root_circuit_index, int const num_elements)
  {
    auto is_source_variable_name_specified = source_variable_name != "";
    auto const& source_int_variable
      = is_source_variable_name_specified
        ? states[root_circuit_index].to_int_variable(source_variable_name)
        : states[root_circuit_index].to_int_variable(variable_names[root_circuit_index]);

    auto const num_circuits = static_cast<int>(states.size());
    for (auto circuit_index = 0; circuit_index < num_circuits; ++circuit_index)
    {
      if (circuit_index == root_circuit_index and not is_source_variable_name_specified)
        continue;

      auto& destination_int_variable = states[circuit_index].to_int_variable(variable_names[circuit_index]);
      std::copy(
        std::addressof(source_int_variable) + num_elements * circuit_index,
        std::addressof(source_int_variable) + num_elements * circuit_index + num_elements,
        std::addressof(destination_int_variable));
    }
  }
} // namespace bra


#endif // BRA_NO_MPI
