#ifndef BRA_NO_MPI
# include <cmath>
# include <iostream>
# include <sstream>
# include <vector>
# include <array>
# include <iterator>
# include <algorithm>
# include <numeric>
# include <random>
# include <utility>
# include <type_traits>

# include <boost/algorithm/string/case_conv.hpp>
# include <boost/range/iterator_range.hpp>


# include <yampi/buffer.hpp>
# include <yampi/tag.hpp>
# include <yampi/rank.hpp>
# include <yampi/send.hpp>
# include <yampi/receive.hpp>
# include <yampi/broadcast.hpp>
# include <yampi/gather.hpp>
# include <yampi/scatter.hpp>
# include <yampi/communicator.hpp>
# include <yampi/intercommunicator.hpp>
# include <yampi/environment.hpp>

# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
#   include <ket/control_io.hpp>
# endif // KET_PRINT_LOG
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   include <ket/gate/utility/cache_aware_iterator.hpp>
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/pauli_index_coeff.hpp>
# include <ket/utility/all_in_state_vector.hpp>
# include <ket/utility/none_in_state_vector.hpp>
# include <ket/mpi/gate/gate.hpp>
# include <ket/mpi/gate/identity.hpp>
# include <ket/mpi/gate/hadamard.hpp>
# include <ket/mpi/gate/not_.hpp>
# include <ket/mpi/gate/pauli_x.hpp>
# include <ket/mpi/gate/pauli_y.hpp>
# include <ket/mpi/gate/pauli_z.hpp>
# include <ket/mpi/gate/swap.hpp>
# include <ket/mpi/gate/sqrt_pauli_x.hpp>
# include <ket/mpi/gate/sqrt_pauli_y.hpp>
# include <ket/mpi/gate/sqrt_pauli_z.hpp>
# include <ket/mpi/gate/phase_shift.hpp>
# include <ket/mpi/gate/x_rotation_half_pi.hpp>
# include <ket/mpi/gate/y_rotation_half_pi.hpp>
# include <ket/mpi/gate/exponential_pauli_x.hpp>
# include <ket/mpi/gate/exponential_pauli_y.hpp>
# include <ket/mpi/gate/exponential_pauli_z.hpp>
# include <ket/mpi/gate/exponential_swap.hpp>
# include <ket/mpi/gate/toffoli.hpp>
# include <ket/mpi/gate/projective_measurement.hpp>
# include <ket/mpi/gate/clear.hpp>
# include <ket/mpi/gate/set.hpp>
# include <ket/mpi/all_spin_expectation_values.hpp>
# include <ket/mpi/print_amplitudes.hpp>
# include <ket/mpi/measure.hpp>
# include <ket/mpi/generate_events.hpp>
# include <ket/mpi/expectation_value.hpp>
# include <ket/mpi/inner_product.hpp>
# include <ket/mpi/fidelity.hpp>
# include <ket/mpi/shor_box.hpp>
# include <ket/mpi/page/page_size.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/apply_local_gate.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>

# include <bra/simple_mpi_state.hpp>
# include <bra/state.hpp>
# include <bra/types.hpp>
# include <bra/fused_gate.hpp>
# include <bra/utility/closest_floating_point_of.hpp>


namespace bra
{
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  template <typename Iterator, typename CacheAwareIterator>
  struct simple_mpi_fused_gate_caller
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

  template <typename MpiPolicy, typename LocalState, typename Communicator, typename Environment>
  auto throw_if_too_many_operated_qubits(
    std::size_t const num_operated_qubits,
    MpiPolicy const& mpi_policy, LocalState const& local_state,
    Communicator const& communicator, Environment const& environment) -> void
  {
    auto const num_local_qubits
      = ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment);
    if (num_operated_qubits > num_local_qubits)
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, num_local_qubits};
  }

  unsigned int simple_mpi_state::do_num_page_qubits() const
  { return 0u; }

  unsigned int simple_mpi_state::do_num_pages() const
  { return 1u; }

# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, circuit_communicator, environment)},
      fused_gates_{},
      cache_aware_fused_gates_{}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, circuit_communicator, environment)},
      fused_gates_{},
      cache_aware_fused_gates_{}
  { }
#   else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, num_elements_in_buffer, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, circuit_communicator, environment)},
      fused_gates_{},
      cache_aware_fused_gates_{}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, num_elements_in_buffer, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, circuit_communicator, environment)},
      fused_gates_{},
      cache_aware_fused_gates_{}
  { }
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
# else // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, circuit_communicator, environment)},
      fused_gates_{}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, circuit_communicator, environment)},
      fused_gates_{}
  { }
#   else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, num_elements_in_buffer, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, circuit_communicator, environment)},
      fused_gates_{}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    bool const is_depolarizing_channel,
    ::bra::real_type const depolarizing_px,
    ::bra::real_type const depolarizing_py,
    ::bra::real_type const depolarizing_pz,
    bool const uses_depolarizing_seed,
    ::bra::state::seed_type const depolarizing_seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& circuit_communicator,
    yampi::communicator const& intercircuit_communicator,
    int const circuit_index,
    std::vector<yampi::intercommunicator> const& intercommunicators,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, is_depolarizing_channel, depolarizing_px, depolarizing_py, depolarizing_pz, uses_depolarizing_seed, depolarizing_seed, num_elements_in_buffer, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, circuit_communicator, environment)},
      fused_gates_{}
  { }
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

  simple_mpi_state::data_type simple_mpi_state::generate_initial_data(
    unsigned int const num_local_qubits,
    ::bra::state::state_integer_type const initial_integer,
    yampi::communicator const& circuit_communicator, yampi::environment const& environment) const
  {
    auto result
      = data_type(
          ket::utility::integer_exp2<std::size_t>(num_local_qubits)
            * ket::mpi::utility::policy::num_data_blocks(mpi_policy_, circuit_communicator, environment),
          complex_type{0});

    auto const rank_index
      = ket::mpi::utility::qubit_value_to_rank_index(
          mpi_policy_, data_, ket::mpi::permutate_bits(permutation_, initial_integer),
          circuit_communicator, environment);
    if (circuit_communicator.rank(environment) == rank_index.first)
      result[rank_index.second] = complex_type{1};

    return result;
  }

  auto simple_mpi_state::generate_probability() -> real_type
  {
    auto result = real_type{0};

    using namespace yampi::literals::rank_literals;
    if (circuit_communicator_.rank(environment_) == 0_r)
    {
      using floating_point_type = typename ::bra::utility::closest_floating_point_of<real_type>::type;
      auto distribution = std::uniform_real_distribution<floating_point_type>{0.0, 1.0};

      result
        = uses_depolarizing_seed_
          ? static_cast<real_type>(distribution(depolarizing_random_number_generator_))
          : static_cast<real_type>(distribution(random_number_generator_));
    }

    yampi::broadcast(yampi::make_buffer(result), 0_r, circuit_communicator_, environment_);

    return result;
  }

  auto simple_mpi_state::do_send_real_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void
  {
    if (destination_circuit_index == circuit_index_)
      return;
    if (is_real_symbol(variable_name))
      return;

    auto const& real_variable = to_real_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    // soruce_circuit_index + num_circuits * destination_circuit_index + num_circuits * num_circuits * rank_in_circuit
    auto const tag
      = yampi::tag{circuit_index_ + num_circuits * destination_circuit_index + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::send(
      yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
      yampi::rank{destination_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_send_complex_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void
  {
    if (destination_circuit_index == circuit_index_)
      return;
    if (is_complex_symbol(variable_name))
      return;

    auto const& complex_variable = to_complex_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    // soruce_circuit_index + num_circuits * destination_circuit_index + num_circuits * num_circuits * rank_in_circuit
    auto const tag
      = yampi::tag{circuit_index_ + num_circuits * destination_circuit_index + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::send(
      yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
      yampi::rank{destination_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_send_int_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void
  {
    if (destination_circuit_index == circuit_index_)
      return;
    if (is_int_symbol(variable_name))
      return;

    auto const& int_variable = to_int_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    // soruce_circuit_index + num_circuits * destination_circuit_index + num_circuits * num_circuits * rank_in_circuit
    auto const tag
      = yampi::tag{circuit_index_ + num_circuits * destination_circuit_index + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::send(
      yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
      yampi::rank{destination_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_receive_real_variable(int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (source_circuit_index == circuit_index_)
      return;
    if (is_real_symbol(variable_name))
      return;

    auto& real_variable = to_real_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    // soruce_circuit_index + num_circuits * destination_circuit_index + num_circuits * num_circuits * rank_in_circuit
    auto const tag
      = yampi::tag{source_circuit_index + num_circuits * circuit_index_ + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::receive(
      yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
      yampi::rank{source_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_receive_complex_variable(int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (source_circuit_index == circuit_index_)
      return;
    if (is_complex_symbol(variable_name))
      return;

    auto& complex_variable = to_complex_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    // soruce_circuit_index + num_circuits * destination_circuit_index + num_circuits * num_circuits * rank_in_circuit
    auto const tag
      = yampi::tag{source_circuit_index + num_circuits * circuit_index_ + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::receive(
      yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
      yampi::rank{source_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_receive_int_variable(int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (source_circuit_index == circuit_index_)
      return;
    if (is_int_symbol(variable_name))
      return;

    auto& int_variable = to_int_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    // soruce_circuit_index + num_circuits * destination_circuit_index + num_circuits * num_circuits * rank_in_circuit
    auto const tag
      = yampi::tag{source_circuit_index + num_circuits * circuit_index_ + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::receive(
      yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
      yampi::rank{source_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_broadcast_real_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (is_real_symbol(variable_name))
      return;

    auto& real_variable = to_real_variable(variable_name);

    yampi::broadcast(
      yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
      yampi::rank{root_circuit_index}, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_broadcast_complex_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (is_complex_symbol(variable_name))
      return;

    auto& complex_variable = to_complex_variable(variable_name);

    yampi::broadcast(
      yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
      yampi::rank{root_circuit_index}, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_broadcast_int_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (is_int_symbol(variable_name))
      return;

    auto& int_variable = to_int_variable(variable_name);

    yampi::broadcast(
      yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
      yampi::rank{root_circuit_index}, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_gather_real_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& destination_variable_name) -> void
  {
    if (is_real_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (destination_variable_name == "")
      {
        auto& real_variable = to_real_variable(variable_name);
        yampi::gather(
          yampi::in_place,
          yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);

        return;
      }

      auto const& real_variable = to_real_variable(variable_name);
      auto& destination_real_variable = to_real_variable(destination_variable_name);
      yampi::gather(
        yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
        std::addressof(destination_real_variable),
        intercircuit_root, intercircuit_communicator_, environment_);

      return;
    }

    auto const& real_variable = to_real_variable(variable_name);
    yampi::gather(
      yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_gather_complex_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& destination_variable_name) -> void
  {
    if (is_complex_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (destination_variable_name == "")
      {
        auto& complex_variable = to_complex_variable(variable_name);
        yampi::gather(
          yampi::in_place,
          yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);

        return;
      }

      auto const& complex_variable = to_complex_variable(variable_name);
      auto& destination_complex_variable = to_complex_variable(destination_variable_name);
      yampi::gather(
        yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
        std::addressof(destination_complex_variable),
        intercircuit_root, intercircuit_communicator_, environment_);

      return;
    }

    auto const& complex_variable = to_complex_variable(variable_name);
    yampi::gather(
      yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_gather_int_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& destination_variable_name) -> void
  {
    if (is_int_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (destination_variable_name == "")
      {
        auto& int_variable = to_int_variable(variable_name);
        yampi::gather(
          yampi::in_place,
          yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);

        return;
      }

      auto const& int_variable = to_int_variable(variable_name);
      auto& destination_int_variable = to_int_variable(destination_variable_name);
      yampi::gather(
        yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
        std::addressof(destination_int_variable),
        intercircuit_root, intercircuit_communicator_, environment_);

      return;
    }

    auto const& int_variable = to_int_variable(variable_name);
    yampi::gather(
      yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_scatter_real_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& source_variable_name) -> void
  {
    if (is_real_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (source_variable_name == "")
      {
        auto const& real_variable = to_real_variable(variable_name);
        yampi::scatter(
          yampi::in_place,
          yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);

        return;
      }

      auto& real_variable = to_real_variable(variable_name);
      auto const& source_real_variable = to_real_variable(source_variable_name);
      yampi::scatter(
        std::addressof(source_real_variable),
        yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
        intercircuit_root, intercircuit_communicator_, environment_);

      return;
    }

    auto& real_variable = to_real_variable(variable_name);
    yampi::scatter(
      yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_scatter_complex_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& source_variable_name) -> void
  {
    if (is_complex_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (source_variable_name == "")
      {
        auto const& complex_variable = to_complex_variable(variable_name);
        yampi::scatter(
          yampi::in_place,
          yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);

        return;
      }

      auto& complex_variable = to_complex_variable(variable_name);
      auto const& source_complex_variable = to_complex_variable(source_variable_name);
      yampi::scatter(
        std::addressof(source_complex_variable),
        yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
        intercircuit_root, intercircuit_communicator_, environment_);

      return;
    }

    auto& complex_variable = to_complex_variable(variable_name);
    yampi::scatter(
      yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }

  auto simple_mpi_state::do_scatter_int_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& source_variable_name) -> void
  {
    if (is_int_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (source_variable_name == "")
      {
        auto const& int_variable = to_int_variable(variable_name);
        yampi::scatter(
          yampi::in_place,
          yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);

        return;
      }

      auto& int_variable = to_int_variable(variable_name);
      auto const& source_int_variable = to_int_variable(source_variable_name);
      yampi::scatter(
        std::addressof(source_int_variable),
        yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
        intercircuit_root, intercircuit_communicator_, environment_);

      return;
    }

    auto& int_variable = to_int_variable(variable_name);
    yampi::scatter(
      yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }

  void simple_mpi_state::do_i_gate(qubit_type const qubit)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_ic_gate(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, control_qubit);
  }

  void simple_mpi_state::do_ii_gate(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_in_gate(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      return;

    assert(qubits.size() > 2u);
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubits);
  }

  void simple_mpi_state::do_hadamard(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_hadamard<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_hadamard<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_not_(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_not_<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_not_<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_x<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_x<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xx<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xx<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_pauli_xn(std::vector<qubit_type> const& qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubits);
  }

  void simple_mpi_state::do_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_y<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_y<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yy<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yy<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_pauli_yn(std::vector<qubit_type> const& qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubits);
  }

  void simple_mpi_state::do_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_z<fused_gate_iterator> >(control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_z<cache_aware_fused_gate_iterator> >(control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, control_qubit);
  }

  void simple_mpi_state::do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zz<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zz<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_pauli_zn(std::vector<qubit_type> const& qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubits);
  }

  void simple_mpi_state::do_swap(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_swap<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_swap<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_sqrt_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_x<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_adj_sqrt_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_x<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_sqrt_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_y<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_adj_sqrt_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_y<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_sqrt_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_z<fused_gate_iterator> >(control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, control_qubit);
  }

  void simple_mpi_state::do_adj_sqrt_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_z<fused_gate_iterator> >(control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, control_qubit);
  }

  void simple_mpi_state::do_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zz<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zz<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_adj_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zz<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zz<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_sqrt_pauli_zn(std::vector<qubit_type> const& qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubits);
  }

  void simple_mpi_state::do_adj_sqrt_pauli_zn(std::vector<qubit_type> const& qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubits);
  }

  void simple_mpi_state::do_u1(real_type const phase, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u1<fused_gate_iterator> >(phase, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u1<cache_aware_fused_gate_iterator> >(phase, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, control_qubit);
  }

  void simple_mpi_state::do_adj_u1(real_type const phase, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u1<fused_gate_iterator> >(phase, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u1<cache_aware_fused_gate_iterator> >(phase, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, control_qubit);
  }

  void simple_mpi_state::do_u2(
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
      ket::mpi::gate::phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, qubit);
  }

  void simple_mpi_state::do_adj_u2(
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
      ket::mpi::gate::adj_phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, qubit);
  }

  void simple_mpi_state::do_u3(
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
      ket::mpi::gate::phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, phase3, qubit);
  }

  void simple_mpi_state::do_adj_u3(
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
      ket::mpi::gate::adj_phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, phase3, qubit);
  }

  void simple_mpi_state::do_phase_shift(
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
      ket::mpi::gate::phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase_coefficient, control_qubit);
  }

  void simple_mpi_state::do_adj_phase_shift(
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
      ket::mpi::gate::adj_phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase_coefficient, control_qubit);
  }

  void simple_mpi_state::do_x_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_x_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_adj_x_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_x_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_y_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_y_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_adj_y_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_y_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_x<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_x<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_adj_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_x<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_x<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_exponential_pauli_xx(
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
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_adj_exponential_pauli_xx(
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
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_exponential_pauli_xn(
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
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubits);
  }

  void simple_mpi_state::do_adj_exponential_pauli_xn(
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
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubits);
  }

  void simple_mpi_state::do_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_y<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_y<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_adj_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_y<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_y<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_exponential_pauli_yy(
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
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_adj_exponential_pauli_yy(
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
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_exponential_pauli_yn(
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
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubits);
  }

  void simple_mpi_state::do_adj_exponential_pauli_yn(
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
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubits);
  }

  void simple_mpi_state::do_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_z<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_adj_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_z<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_exponential_pauli_zz(
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
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_adj_exponential_pauli_zz(
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
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_exponential_pauli_zn(
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
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubits);
  }

  void simple_mpi_state::do_adj_exponential_pauli_zn(
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
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubits);
  }

  void simple_mpi_state::do_exponential_swap(
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
      ket::mpi::gate::exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_adj_exponential_swap(
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
      ket::mpi::gate::adj_exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_toffoli(
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
      ket::mpi::gate::toffoli(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit1, control_qubit2);
  }

  ket::gate::outcome simple_mpi_state::do_projective_measurement(
    qubit_type const qubit, yampi::rank const root)
  {
    return ket::mpi::gate::projective_measurement(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, root, circuit_communicator_, environment_, random_number_generator_, qubit);
  }

  void simple_mpi_state::do_expectation_values(yampi::rank const root)
  {
    maybe_expectation_values_
      = ket::mpi::all_spin_expectation_values<typename spins_type::allocator_type>(
          mpi_policy_, parallel_policy_,
          data_, permutation_, total_num_qubits_, buffer_, root, circuit_communicator_, environment_);
  }

  void simple_mpi_state::do_amplitudes(yampi::rank const root, std::vector< ::bra::state_integer_type > const& amplitude_indices)
  {
    std::ostringstream oss;

    if (amplitude_indices.empty())
      ket::mpi::println_amplitudes(
        mpi_policy_, oss, data_, permutation_, root, circuit_communicator_, environment_,
        [this](::bra::state_integer_type const qubit_value, ::bra::complex_type const& amplitude)
        {
          std::ostringstream oss;
          using std::real;
          using std::imag;
          oss << ::bra::state_detail::integer_to_bits_string(qubit_value, this->total_num_qubits_) << " => " << real(amplitude) << " + " << imag(amplitude) << " i";
          return oss.str();
        }, std::string{"\n"});
    else
    {
      auto const present_rank = circuit_communicator_.rank(environment_);

      for (auto const amplitude_index: amplitude_indices)
      {
        auto const rank_index
          = ::ket::mpi::utility::qubit_value_to_rank_index(
              mpi_policy_, data_, ::ket::mpi::permutate_bits(permutation_, amplitude_index), circuit_communicator_, environment_);

        if (present_rank == root)
        {
          auto amplitude = ::bra::complex_type{};

          if (present_rank == rank_index.first)
            amplitude = data_[rank_index.second];
          else
            yampi::receive(yampi::ignore_status, yampi::make_buffer(amplitude), rank_index.first, yampi::tag{static_cast<int>(rank_index.second)}, circuit_communicator_, environment_);

          using std::real;
          using std::imag;
          oss << ::bra::state_detail::integer_to_bits_string(amplitude_index, total_num_qubits_) << " => " << real(amplitude) << " + " << imag(amplitude) << " i\n";
        }
        else if (present_rank == rank_index.first)
          yampi::send(yampi::make_buffer(data_[rank_index.second]), root, yampi::tag{static_cast<int>(rank_index.second)}, circuit_communicator_, environment_);
      }
    }

    if (circuit_communicator_.rank(environment_) == root)
      std::cout << oss.str() << std::flush;
  }

  void simple_mpi_state::do_measure(yampi::rank const root)
  {
    measured_value_
      = ket::mpi::measure(
          mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
          data_, random_number_generator_, permutation_, circuit_communicator_, environment_);
  }

  void simple_mpi_state::do_generate_events(yampi::rank const root, int const num_events, int const seed)
  {
    if (seed < 0)
      ket::mpi::generate_events(
        mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
        generated_events_, data_, num_events, random_number_generator_, permutation_,
        circuit_communicator_, environment_);
    else
      ket::mpi::generate_events(
        mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
        generated_events_, data_, num_events, random_number_generator_, static_cast<seed_type>(seed), permutation_,
        circuit_communicator_, environment_);
  }

  void simple_mpi_state::do_expectation_value(std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits)
  {
    auto const num_operated_qubits = operated_qubits.size();
    auto const pauli_string_space_element = to_pauli_string_space(operator_literal_or_variable_name);

    if (num_operated_qubits != pauli_string_space_element.num_qubits())
      throw ::bra::wrong_pauli_string_length_error{num_operated_qubits, pauli_string_space_element.num_qubits()};

    result_
      = ket::mpi::runtime::ranges::expectation_value(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, circuit_communicator_, environment_,
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

  void simple_mpi_state::do_inner_product(std::string const& remote_circuit_index_or_all)
  {
    auto remote_circuit_index = -1;
    auto is_all = false;

    if (std::isdigit(static_cast<unsigned char>(remote_circuit_index_or_all.front())))
    {
      remote_circuit_index = boost::lexical_cast<int>(remote_circuit_index_or_all);
      if (remote_circuit_index < 0 or remote_circuit_index == circuit_index_)
        return;
    }
    else if (boost::algorithm::to_upper_copy(remote_circuit_index_or_all) == "ALL")
      is_all = true;
    else
      return;

    if (is_all)
    {
      using namespace yampi::literals::rank_literals;
      result_
        = ket::mpi::inner_product(
            mpi_policy_, parallel_policy_,
            data_, buffer_, circuit_communicator_, 0_r, intercircuit_communicator_, environment_);
    }
    else
    {
      auto const index = remote_circuit_index < circuit_index_ ? remote_circuit_index : remote_circuit_index - 1;
      result_
        = ket::mpi::inner_product(
            mpi_policy_, parallel_policy_,
            data_, buffer_, circuit_communicator_, intercommunicators_[index], environment_);
    }
  }

  void simple_mpi_state::do_inner_product(std::string const& remote_circuit_index_or_all, std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits)
  {
    auto remote_circuit_index = -1;
    auto is_all = false;

    auto const num_operated_qubits = operated_qubits.size();
    auto const pauli_string_space_element = to_pauli_string_space(operator_literal_or_variable_name);

    if (num_operated_qubits != pauli_string_space_element.num_qubits())
      throw ::bra::wrong_pauli_string_length_error{num_operated_qubits, pauli_string_space_element.num_qubits()};

    if (std::isdigit(static_cast<unsigned char>(remote_circuit_index_or_all.front())))
    {
      remote_circuit_index = boost::lexical_cast<int>(remote_circuit_index_or_all);
      if (remote_circuit_index < 0 or remote_circuit_index == circuit_index_)
        return;
    }
    else if (boost::algorithm::to_upper_copy(remote_circuit_index_or_all) == "ALL")
      is_all = true;
    else
      return;

    if (is_all)
    {
      using namespace yampi::literals::rank_literals;
      result_
        = ket::mpi::runtime::ranges::inner_product(
            mpi_policy_, parallel_policy_,
            data_, permutation_, buffer_, circuit_communicator_, 0_r, intercircuit_communicator_, environment_,
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
    else
    {
      auto const index = remote_circuit_index < circuit_index_ ? remote_circuit_index : remote_circuit_index - 1;
      result_
        = ket::mpi::runtime::ranges::inner_product(
            mpi_policy_, parallel_policy_,
            data_, permutation_, buffer_, circuit_communicator_, intercommunicators_[index], environment_,
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
  }

  void simple_mpi_state::do_fidelity(std::string const& remote_circuit_index_or_all)
  {
    auto remote_circuit_index = -1;
    auto is_all = false;

    if (std::isdigit(static_cast<unsigned char>(remote_circuit_index_or_all.front())))
    {
      remote_circuit_index = boost::lexical_cast<int>(remote_circuit_index_or_all);
      if (remote_circuit_index < 0 or remote_circuit_index == circuit_index_)
        return;
    }
    else if (boost::algorithm::to_upper_copy(remote_circuit_index_or_all) == "ALL")
      is_all = true;
    else
      return;

    if (is_all)
    {
      using namespace yampi::literals::rank_literals;
      result_
        = ket::mpi::fidelity(
            mpi_policy_, parallel_policy_,
            data_, buffer_, circuit_communicator_, 0_r, intercircuit_communicator_, environment_);
    }
    else
    {
      auto const index = remote_circuit_index < circuit_index_ ? remote_circuit_index : remote_circuit_index - 1;
      result_
        = ket::mpi::fidelity(
            mpi_policy_, parallel_policy_,
            data_, buffer_, circuit_communicator_, intercommunicators_[index], environment_);
    }
  }

  void simple_mpi_state::do_fidelity(std::string const& remote_circuit_index_or_all, std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits)
  {
    auto remote_circuit_index = -1;
    auto is_all = false;

    auto const num_operated_qubits = operated_qubits.size();
    auto const pauli_string_space_element = to_pauli_string_space(operator_literal_or_variable_name);

    if (num_operated_qubits != pauli_string_space_element.num_qubits())
      throw ::bra::wrong_pauli_string_length_error{num_operated_qubits, pauli_string_space_element.num_qubits()};

    if (std::isdigit(static_cast<unsigned char>(remote_circuit_index_or_all.front())))
    {
      remote_circuit_index = boost::lexical_cast<int>(remote_circuit_index_or_all);
      if (remote_circuit_index < 0 or remote_circuit_index == circuit_index_)
        return;
    }
    else if (boost::algorithm::to_upper_copy(remote_circuit_index_or_all) == "ALL")
      is_all = true;
    else
      return;

    if (is_all)
    {
      using namespace yampi::literals::rank_literals;
      result_
        = ket::mpi::runtime::ranges::fidelity(
            mpi_policy_, parallel_policy_,
            data_, permutation_, buffer_, circuit_communicator_, 0_r, intercircuit_communicator_, environment_,
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
    else
    {
      auto const index = remote_circuit_index < circuit_index_ ? remote_circuit_index : remote_circuit_index - 1;
      result_
        = ket::mpi::runtime::ranges::fidelity(
            mpi_policy_, parallel_policy_,
            data_, permutation_, buffer_, circuit_communicator_, intercommunicators_[index], environment_,
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
  }

  void simple_mpi_state::do_shor_box(
    state_integer_type const divisor, state_integer_type const base,
    std::vector<qubit_type> const& exponent_qubits,
    std::vector<qubit_type> const& modular_exponentiation_qubits)
  {
    ket::mpi::shor_box(
      mpi_policy_, parallel_policy_,
      data_, base, divisor, exponent_qubits, modular_exponentiation_qubits,
      permutation_, circuit_communicator_, environment_);
  }

  void simple_mpi_state::do_begin_fusion()
  { }

  void simple_mpi_state::do_end_fusion()
  {
# if !(!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    assert(fused_gates_.size() == cache_aware_fused_gates_.size());
# endif // !(!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && defined(KET_USE_ON_CACHE_STATE_VECTOR)))

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

    auto to_qubit_index_in_fused_gates = std::vector< ::bra::bit_integer_type >(total_num_qubits_);
    using std::begin;
    using std::end;
    std::iota(begin(to_qubit_index_in_fused_gates), end(to_qubit_index_in_fused_gates), ::bra::bit_integer_type{0u});
    auto present_qubit_index = ::bra::bit_integer_type{0u};
    for (auto const fused_qubit: fused_qubits)
      to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(fused_qubit)] = present_qubit_index++;
    for (auto const fused_control_qubit: fused_control_qubits)
      to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(fused_control_qubit.qubit())] = present_qubit_index++;

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
      = simple_mpi_fused_gate_caller<fused_gate_iterator, cache_aware_fused_gate_iterator>{
          fused_gates_, cache_aware_fused_gates_, to_qubit_index_in_fused_gates};
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

    ket::mpi::gate::runtime::ranges::gate(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_,
      call_fused_gates, fused_qubits, fused_control_qubits);

    fused_gates_.clear();
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    cache_aware_fused_gates_.clear();
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  }

  void simple_mpi_state::do_clear(qubit_type const qubit)
  {
    ket::mpi::gate::clear(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_set(qubit_type const qubit)
  {
    ket::mpi::gate::set(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_controlled_i_gate(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_controlled_ic_gate(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_multi_controlled_in_gate(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      return;

    assert(target_qubits.size() > 0u);
    assert(control_qubits.size() > 0u);
    assert(target_qubits.size() + control_qubits.size() > 2u);

    auto qubits = target_qubits;
    qubits.reserve(target_qubits.size() + control_qubits.size());
    for (auto const control_qubit: control_qubits)
      qubits.push_back(control_qubit.qubit());
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubits);
  }

  void simple_mpi_state::do_multi_controlled_ic_gate(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      return;

    assert(control_qubits.size() > 2u);

    auto qubits = std::vector<qubit_type>{};
    qubits.reserve(control_qubits.size());
    for (auto const control_qubit: control_qubits)
      qubits.push_back(control_qubit.qubit());
    ::bra::throw_if_too_many_operated_qubits(
      qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, qubits);
  }

  void simple_mpi_state::do_controlled_hadamard(
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
      ket::mpi::gate::hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_hadamard(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::hadamard(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_controlled_not(
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
      ket::mpi::gate::not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_not(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::not_(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_controlled_pauli_x(
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
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_pauli_xn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      target_qubits.size() + control_qubits.size(),
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_controlled_pauli_y(
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
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_pauli_yn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      target_qubits.size() + control_qubits.size(),
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_controlled_pauli_z(
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
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_multi_controlled_pauli_z(std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, control_qubits);
  }

  void simple_mpi_state::do_multi_controlled_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      target_qubits.size() + control_qubits.size(),
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_multi_controlled_swap(qubit_type const target_qubit1, qubit_type const target_qubit2,
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{2u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::swap(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit1, target_qubit2, control_qubits);
  }

  void simple_mpi_state::do_controlled_sqrt_pauli_x(
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
      ket::mpi::gate::sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_sqrt_pauli_x(
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
      ket::mpi::gate::adj_sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_sqrt_pauli_x(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::sqrt_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_sqrt_pauli_x(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_controlled_sqrt_pauli_y(
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
      ket::mpi::gate::sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_sqrt_pauli_y(
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
      ket::mpi::gate::adj_sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_sqrt_pauli_y(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::sqrt_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_sqrt_pauli_y(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_controlled_sqrt_pauli_z(
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
      ket::mpi::gate::sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_adj_controlled_sqrt_pauli_z(
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
      ket::mpi::gate::adj_sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, control_qubits);
  }

  void simple_mpi_state::do_multi_controlled_sqrt_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      target_qubits.size() + control_qubits.size(),
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_sqrt_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      target_qubits.size() + control_qubits.size(),
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_controlled_phase_shift(
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
      ket::mpi::gate::phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase_coefficient, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_adj_controlled_phase_shift(
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
      ket::mpi::gate::adj_phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase_coefficient, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_multi_controlled_phase_shift(complex_type const& phase_coefficient,
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase_coefficient, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_phase_shift(complex_type const& phase_coefficient,
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase_coefficient, control_qubits);
  }

  void simple_mpi_state::do_controlled_u1(
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
      ket::mpi::gate::phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_adj_controlled_u1(
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
      ket::mpi::gate::adj_phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_multi_controlled_u1(real_type const phase, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::phase_shift(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_u1(real_type const phase, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size(), mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_phase_shift(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, control_qubits);
  }

  void simple_mpi_state::do_controlled_u2(
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
      ket::mpi::gate::phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_u2(
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
      ket::mpi::gate::adj_phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_u2(real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_u2(real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_controlled_u3(
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
      ket::mpi::gate::phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_u3(
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
      ket::mpi::gate::adj_phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_u3(real_type const phase1, real_type const phase2, real_type const phase3,
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_u3(real_type const phase1, real_type const phase2, real_type const phase3,
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_controlled_x_rotation_half_pi(
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
      ket::mpi::gate::x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_x_rotation_half_pi(
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
      ket::mpi::gate::adj_x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_x_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_x_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_controlled_y_rotation_half_pi(
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
      ket::mpi::gate::y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_y_rotation_half_pi(
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
      ket::mpi::gate::adj_y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_y_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_y_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, target_qubit, control_qubits);
  }

  void simple_mpi_state::do_controlled_exponential_pauli_x(
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
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_exponential_pauli_x(
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
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_exponential_pauli_xn(real_type const phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      target_qubits.size() + control_qubits.size(),
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_exponential_pauli_xn(real_type const phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      target_qubits.size() + control_qubits.size(),
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_controlled_exponential_pauli_y(
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
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_exponential_pauli_y(
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
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_exponential_pauli_yn(real_type const phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      target_qubits.size() + control_qubits.size(),
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_exponential_pauli_yn(real_type const phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      target_qubits.size() + control_qubits.size(),
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_controlled_exponential_pauli_z(
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
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_exponential_pauli_z(
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
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_exponential_pauli_z(real_type const phase, qubit_type const target_qubit,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_z<fused_gate_iterator> >(phase, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    assert(control_qubits.size() > 1u);
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    auto const target_qubits = boost::make_iterator_range(&target_qubit, &target_qubit + 1);
    ket::mpi::gate::runtime::ranges::exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_exponential_pauli_z(real_type const phase, qubit_type const target_qubit,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_z<fused_gate_iterator> >(phase, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    assert(control_qubits.size() > 1u);
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{1u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    auto const target_qubits = boost::make_iterator_range(&target_qubit, &target_qubit + 1);
    ket::mpi::gate::runtime::ranges::adj_exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_multi_controlled_exponential_pauli_zn(real_type const phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      target_qubits.size() + control_qubits.size(),
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_exponential_pauli_zn(real_type const phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
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
    ::bra::throw_if_too_many_operated_qubits(
      target_qubits.size() + control_qubits.size(),
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubits, control_qubits);
  }

  void simple_mpi_state::do_multi_controlled_exponential_swap(real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{2u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::exponential_swap(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubit1, target_qubit2, control_qubits);
  }

  void simple_mpi_state::do_adj_multi_controlled_exponential_swap(real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
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
    ::bra::throw_if_too_many_operated_qubits(
      control_qubits.size() + std::size_t{2u},
      mpi_policy_, data_, circuit_communicator_, environment_);

    ket::mpi::gate::runtime::ranges::adj_exponential_swap(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, circuit_communicator_, environment_, phase, target_qubit1, target_qubit2, control_qubits);
  }
} // namespace bra


#endif // BRA_NO_MPI
