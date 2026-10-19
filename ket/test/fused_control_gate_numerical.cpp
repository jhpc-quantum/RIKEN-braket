// python3 ket/test/fused_control_gate_numerical.py
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <ket/gate/fused/sqrt_pauli_x.hpp>
#include <ket/gate/fused/controlled_not.hpp>
#include <ket/gate/fused/sqrt_pauli_y.hpp>
#include <ket/gate/fused/pauli_z.hpp>
#include <ket/gate/fused/exponential_pauli_z.hpp>
#include <ket/gate/fused/hadamard.hpp>
#include <ket/gate/fused/pauli_y.hpp>
#include <ket/gate/fused/not_.hpp>
#include <ket/gate/fused/pauli_x.hpp>
#include <ket/gate/fused/swap.hpp>
#include <ket/gate/fused/controlled_phase_shift.hpp>
#include <ket/gate/fused/x_rotation_half_pi.hpp>
#include <ket/gate/fused/toffoli.hpp>
#include <ket/gate/fused/exponential_swap.hpp>
#include <ket/gate/fused/y_rotation_half_pi.hpp>
#include <ket/gate/fused/controlled_v.hpp>
#include <ket/gate/fused/exponential_pauli_y.hpp>
#include <ket/gate/fused/exponential_pauli_x.hpp>
#include <ket/gate/fused/sqrt_pauli_z.hpp>
#include <ket/gate/fused/phase_shift.hpp>
#include <ket/gate/sqrt_pauli_x.hpp>
#include <ket/gate/controlled_not.hpp>
#include <ket/gate/sqrt_pauli_y.hpp>
#include <ket/gate/pauli_z.hpp>
#include <ket/gate/exponential_pauli_z.hpp>
#include <ket/gate/hadamard.hpp>
#include <ket/gate/pauli_y.hpp>
#include <ket/gate/not_.hpp>
#include <ket/gate/pauli_x.hpp>
#include <ket/gate/swap.hpp>
#include <ket/gate/controlled_phase_shift.hpp>
#include <ket/gate/x_rotation_half_pi.hpp>
#include <ket/gate/toffoli.hpp>
#include <ket/gate/exponential_swap.hpp>
#include <ket/gate/y_rotation_half_pi.hpp>
#include <ket/gate/controlled_v.hpp>
#include <ket/gate/exponential_pauli_y.hpp>
#include <ket/gate/exponential_pauli_x.hpp>
#include <ket/gate/sqrt_pauli_z.hpp>
#include <ket/gate/phase_shift.hpp>
#include <ket/utility/parallel/loop_n.hpp>

#ifndef KET_TEST_GROUP
# define KET_TEST_GROUP 1
#endif

namespace
{
  using qubit_type = ket::qubit<std::uint64_t, unsigned int>;
  using complex_type = std::complex<double>;
  using state_type = std::vector<complex_type>;
  using runtime_control = ket::runtime_control<qubit_type>;
  constexpr auto num_fused_qubits = std::size_t{7u};
  constexpr auto total_qubits = std::size_t{8u};
  constexpr auto state_size = std::size_t{1u} << total_qubits;
  constexpr std::array<unsigned int, 3u> target_positions{{3u, 0u, 5u}};
  constexpr std::array<unsigned int, 4u> control_positions{{4u, 1u, 6u, 2u}};
  auto checked = std::size_t{0u};

  using parallel_policy_type = decltype(ket::utility::policy::make_parallel(3u));

  // Keep the threading machinery independent of each gate/polarity template.
  auto execute_parallel(std::function<void(parallel_policy_type, int)> const& operation) -> void
  {
    auto const policy = ket::utility::policy::make_parallel(3u);
    ket::utility::execute(policy, [&](int const thread_index, auto&) { operation(policy, thread_index); });
  }


  struct context
  {
    std::array<unsigned int, num_fused_qubits> physical;
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    std::array<qubit_type, num_fused_qubits> unsorted;
    std::array<qubit_type, num_fused_qubits + 1u> sorted;
#else
    std::array<std::uint64_t, num_fused_qubits> unsorted;
    std::array<std::uint64_t, num_fused_qubits + 1u> sorted;
#endif
    std::uint64_t block = 0u;
    bool parallel = false;
    unsigned int spectator = 0u;

    explicit context(std::array<unsigned int, num_fused_qubits> const& positions) : physical{positions}
    {
      auto wires = std::vector<qubit_type>{};
      for (auto position : positions) wires.emplace_back(position);
      while (std::find(positions.begin(), positions.end(), spectator) != positions.end()) ++spectator;
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
      std::copy(wires.begin(), wires.end(), unsorted.begin());
      std::sort(wires.begin(), wires.end());
      std::copy(wires.begin(), wires.end(), sorted.begin());
      sorted.back() = qubit_type{static_cast<unsigned int>(total_qubits)};
#else
      auto masks = std::vector<std::uint64_t>{};
      ket::gate::gate_detail::runtime::ranges::make_qubit_masks(wires, std::back_inserter(masks));
      std::copy(masks.begin(), masks.end(), unsorted.begin());
      masks.clear();
      ket::gate::gate_detail::runtime::ranges::make_index_masks(wires, std::back_inserter(masks));
      std::copy(masks.begin(), masks.end(), sorted.begin());
#endif
    }
  };

  auto flip_controls(state_type& state, std::size_t const bits) -> void
  {
    for (auto i = std::size_t{0u}; i < state.size(); ++i)
      if (i < (i xor bits)) std::swap(state[i], state[i xor bits]);
  }

  auto compare(std::string const& name, state_type const& actual, state_type const& expected) -> void
  {
    for (auto i = std::size_t{0u}; i < actual.size(); ++i)
      if (std::abs(actual[i] - expected[i]) > 1e-11)
      {
        std::cerr << name << ": mismatch at " << i << ": " << actual[i] << " != " << expected[i] << '\n';
        std::exit(1);
      }
    ++checked;
  }

  template <typename Gate>
  auto apply_blocks(Gate const& gate, state_type& state, context& ctx) -> void
  {
    for (ctx.block = 0u; ctx.block < 2u; ++ctx.block) gate(state, ctx);
  }

  template <typename RangeGate>
  auto check_homogeneous(RangeGate const& gate, state_type const& initial, state_type const& expected,
                         context& ctx, std::size_t const count, std::true_type) -> void
  {
    auto controls = std::vector<ket::control<qubit_type, 0u>>{};
    for (auto i = std::size_t{0u}; i < count; ++i) controls.emplace_back(qubit_type{control_positions[i]});
    auto actual = initial;
    apply_blocks([&](state_type& state, context const& c) { gate(state, c, controls); }, actual, ctx);
    compare("homogeneous negative range", actual, expected);
  }
  template <typename RangeGate>
  auto check_homogeneous(RangeGate const&, state_type const&, state_type const&, context&, std::size_t, std::false_type) -> void { }

  template <typename Native, typename Positive, typename RangeGate, typename IteratorGate, typename AllNegative>
  auto check(std::string const& name, std::vector<unsigned int> const& values, Native const& native,
             Positive const& positive, RangeGate const& range_gate, IteratorGate const& iterator_gate,
             AllNegative const all_negative, bool const can_parallel) -> void
  {
    auto const controls = [&] {
      auto result = std::vector<runtime_control>{};
      for (auto i = std::size_t{0u}; i < values.size(); ++i) result.emplace_back(qubit_type{control_positions[i]}, values[i]);
      return result;
    }();
    // Identity fast path, permuted contiguous block, then a noncontiguous block
    // with its spectator in the middle of the global state index.
    for (auto const positions : {
      std::array<unsigned int, 7u>{{0u, 1u, 2u, 3u, 4u, 5u, 6u}},
      std::array<unsigned int, 7u>{{2u, 0u, 5u, 1u, 6u, 3u, 4u}},
      std::array<unsigned int, 7u>{{6u, 0u, 7u, 3u, 1u, 5u, 2u}}})
    {
      auto ctx = context{positions};
      auto negative_bits = std::size_t{0u};
      for (auto i = std::size_t{0u}; i < values.size(); ++i)
        if (values[i] == 0u) negative_bits |= std::size_t{1u} << positions[control_positions[i]];
      for (auto basis = std::size_t{0u}; basis <= state_size; ++basis)
      {
        auto initial = state_type(state_size);
        if (basis < state_size) initial[basis] = complex_type{1.0};
        else
          for (auto i = std::size_t{0u}; i < state_size; ++i)
            initial[i] = complex_type{0.01 * double(i + 1u), -0.02 * double((i * 3u) % 17u)};
        auto expected = initial;
        flip_controls(expected, negative_bits);
        positive(expected, ctx);
        flip_controls(expected, negative_bits);
        auto actual = initial;
        apply_blocks(native, actual, ctx);
        compare(name + "/variadic", actual, expected);
        actual = initial;
        apply_blocks([&](state_type& state, context const& c) { range_gate(state, c, controls); }, actual, ctx);
        compare(name + "/range", actual, expected);
        actual = initial;
        apply_blocks([&](state_type& state, context const& c) { iterator_gate(state, c, controls); }, actual, ctx);
        compare(name + "/iterators", actual, expected);
        check_homogeneous(range_gate, initial, expected, ctx, values.size(), all_negative);
        if (basis == state_size)
        {
          if (can_parallel)
          {
            ctx.parallel = true;
            actual = initial;
            apply_blocks([&](state_type& state, context const& c) { range_gate(state, c, controls); }, actual, ctx);
            compare(name + "/parallel", actual, expected);
            ctx.parallel = false;
          }
          auto one_block = expected;
          for (auto i = std::size_t{0u}; i < state_size; ++i)
            if ((i bitand (std::size_t{1u} << ctx.spectator)) == 0u) one_block[i] = initial[i];
          ctx.block = 1u;
          actual = initial;
          native(actual, ctx);
          compare(name + "/single outer block", actual, one_block);
          actual = initial;
          range_gate(actual, ctx, controls);
          compare(name + "/single outer block range", actual, one_block);
        }
      }
    }
  }

  template <unsigned int... control_values, std::size_t... indices>
  auto run(std::integer_sequence<unsigned int, control_values...>, std::index_sequence<indices...>) -> void
  {
    auto const values = std::vector<unsigned int>{control_values...};
    constexpr auto all_negative = ket::gate::utility::negative_control_mask<unsigned int>(
      ket::control<qubit_type, control_values>{control_positions[indices]}...) == (1u << sizeof...(control_values)) - 1u;
#if KET_TEST_GROUP == 1
  check("hadamard/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::hadamard(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::hadamard(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::hadamard(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::hadamard(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::hadamard(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_hadamard/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_hadamard(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_hadamard(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_hadamard(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_hadamard(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("not_/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::not_(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::not_(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::not_(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::not_(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::not_(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_not_/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_not_(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_not_(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_not_(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_not_(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_not_(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("controlled_not/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::controlled_not(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::controlled_not(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::controlled_not(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::controlled_not(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("adj_controlled_not/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_controlled_not(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_controlled_not(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_controlled_not(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_controlled_not(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("pauli_x/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::pauli_x(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::pauli_x(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::pauli_x(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_pauli_x/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_pauli_x(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::adj_pauli_x(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("pauli_y/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::pauli_y(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::pauli_y(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::pauli_y(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_pauli_y/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_pauli_y(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::adj_pauli_y(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("pauli_z/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::pauli_z(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_pauli_z/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_pauli_z(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::adj_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("sqrt_pauli_x/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::sqrt_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::sqrt_pauli_x(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::sqrt_pauli_x(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::sqrt_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::sqrt_pauli_x(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_sqrt_pauli_x/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_sqrt_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_sqrt_pauli_x(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_sqrt_pauli_x(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_sqrt_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_sqrt_pauli_x(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("sqrt_pauli_y/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::sqrt_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::sqrt_pauli_y(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::sqrt_pauli_y(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::sqrt_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::sqrt_pauli_y(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_sqrt_pauli_y/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_sqrt_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_sqrt_pauli_y(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_sqrt_pauli_y(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_sqrt_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_sqrt_pauli_y(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("sqrt_pauli_z/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::sqrt_pauli_z(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::sqrt_pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_sqrt_pauli_z/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_sqrt_pauli_z(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_sqrt_pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::adj_sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("x_rotation_half_pi/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::x_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::x_rotation_half_pi(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::x_rotation_half_pi(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::x_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::x_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_x_rotation_half_pi/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_x_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_x_rotation_half_pi(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_x_rotation_half_pi(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_x_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_x_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("y_rotation_half_pi/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::y_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::y_rotation_half_pi(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::y_rotation_half_pi(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::y_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::y_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_y_rotation_half_pi/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_y_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_y_rotation_half_pi(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_y_rotation_half_pi(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_y_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_y_rotation_half_pi(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);
#endif

#if KET_TEST_GROUP == 2
  check("phase_shift/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::phase_shift(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::phase_shift(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::phase_shift(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_phase_shift/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_phase_shift(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_phase_shift(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_phase_shift(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("phase_shift_coeff/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
      {
        execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
          ket::gate::fused::runtime::ranges::phase_shift_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted,
            complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls);
        });
      }
      else ket::gate::fused::runtime::ranges::phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted,
        complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_phase_shift_coeff/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
      {
        execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
          ket::gate::fused::runtime::ranges::adj_phase_shift_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted,
            complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls);
        });
      }
      else ket::gate::fused::runtime::ranges::adj_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted,
        complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("phase_shift2/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::phase_shift2(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.21, -0.34, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::phase_shift2(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.21, -0.34, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::phase_shift2(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.21, -0.34, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::phase_shift2(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.21, -0.34, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_phase_shift2/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_phase_shift2(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.21, -0.34, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_phase_shift2(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.21, -0.34, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_phase_shift2(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.21, -0.34, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_phase_shift2(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.21, -0.34, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("phase_shift3/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::phase_shift3(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::phase_shift3(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::phase_shift3(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::phase_shift3(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_phase_shift3/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_phase_shift3(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_phase_shift3(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_phase_shift3(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_phase_shift3(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("exponential_pauli_x/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::exponential_pauli_x(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_exponential_pauli_x/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_exponential_pauli_x(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::adj_exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("exponential_pauli_x_coeff/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::exponential_pauli_x_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::exponential_pauli_x_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::exponential_pauli_x_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::exponential_pauli_x_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_exponential_pauli_x_coeff/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_exponential_pauli_x_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_exponential_pauli_x_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_exponential_pauli_x_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::adj_exponential_pauli_x_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("exponential_pauli_y/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::exponential_pauli_y(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_exponential_pauli_y/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_exponential_pauli_y(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::adj_exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("exponential_pauli_y_coeff/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::exponential_pauli_y_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::exponential_pauli_y_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::exponential_pauli_y_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::exponential_pauli_y_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_exponential_pauli_y_coeff/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_exponential_pauli_y_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_exponential_pauli_y_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_exponential_pauli_y_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::adj_exponential_pauli_y_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("exponential_pauli_z/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::exponential_pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_exponential_pauli_z/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_exponential_pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::adj_exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("exponential_pauli_z_coeff/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::exponential_pauli_z_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::exponential_pauli_z_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::exponential_pauli_z_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::exponential_pauli_z_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_exponential_pauli_z_coeff/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_exponential_pauli_z_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_exponential_pauli_z_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_exponential_pauli_z_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::fused::runtime::adj_exponential_pauli_z_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("controlled_v/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::controlled_v(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::controlled_v(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::controlled_v(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::controlled_v(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("adj_controlled_v/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_controlled_v(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_controlled_v(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_controlled_v(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_controlled_v(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("controlled_v_coeff/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::controlled_v_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::controlled_v_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::controlled_v_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("adj_controlled_v_coeff/1", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_controlled_v_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_controlled_v_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_controlled_v_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);
#endif

#if KET_TEST_GROUP == 3
  check("pauli_x/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::pauli_x(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::pauli_x(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::pauli_x(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_pauli_x/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_pauli_x(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::adj_pauli_x(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("pauli_y/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::pauli_y(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::pauli_y(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::pauli_y(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_pauli_y/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_pauli_y(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::adj_pauli_y(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("pauli_z/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::pauli_z(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_pauli_z/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_pauli_z(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::adj_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("exponential_pauli_x/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::exponential_pauli_x(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_exponential_pauli_x/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_exponential_pauli_x(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::adj_exponential_pauli_x(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("exponential_pauli_y/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::exponential_pauli_y(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_exponential_pauli_y/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_exponential_pauli_y(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::adj_exponential_pauli_y(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("exponential_pauli_z/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::exponential_pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_exponential_pauli_z/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_exponential_pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::adj_exponential_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("swap/2", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::swap(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::swap(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::swap(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::swap(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::swap(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_swap/2", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_swap(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_swap(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_swap(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_swap(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("exponential_swap/2", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::exponential_swap(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::exponential_swap(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::exponential_swap(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::exponential_swap(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::exponential_swap(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_exponential_swap/2", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_exponential_swap(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_exponential_swap(state.begin(), state.end(), 0.37, qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_exponential_swap(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_exponential_swap(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_exponential_swap(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("exponential_swap_coeff/2", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::exponential_swap_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::exponential_swap_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::exponential_swap_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::exponential_swap_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_exponential_swap_coeff/2", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_exponential_swap_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_exponential_swap_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_exponential_swap_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_exponential_swap_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("sqrt_pauli_z/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::sqrt_pauli_z(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::sqrt_pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_sqrt_pauli_z/3", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_sqrt_pauli_z(state.begin(), state.end(), qubit_type{ctx.physical[target_positions[0]]}, qubit_type{ctx.physical[target_positions[1]]}, qubit_type{ctx.physical[target_positions[2]]}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_sqrt_pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::fused::runtime::adj_sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);
#endif

#if KET_TEST_GROUP == 4
  check("phase_shift/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::phase_shift(state.begin(), state.end(), 0.37, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::phase_shift(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::phase_shift(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_phase_shift/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_phase_shift(state.begin(), state.end(), 0.37, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_phase_shift(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_phase_shift(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("phase_shift_coeff/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::phase_shift_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_phase_shift_coeff/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_phase_shift_coeff(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("controlled_phase_shift/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::controlled_phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::controlled_phase_shift(state.begin(), state.end(), 0.37, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::controlled_phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::controlled_phase_shift(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("adj_controlled_phase_shift/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_controlled_phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_controlled_phase_shift(state.begin(), state.end(), 0.37, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_controlled_phase_shift(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, 0.37, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_controlled_phase_shift(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), 0.37, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("controlled_phase_shift_coeff/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::controlled_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::controlled_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::controlled_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("adj_controlled_phase_shift_coeff/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_controlled_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_controlled_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, complex_type{0.8, 0.6}, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_controlled_phase_shift_coeff(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), complex_type{0.8, 0.6}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("pauli_z/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::pauli_z(state.begin(), state.end(), ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_pauli_z/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_pauli_z(state.begin(), state.end(), ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      ket::gate::fused::runtime::ranges::adj_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, false);

  check("sqrt_pauli_z/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::sqrt_pauli_z(state.begin(), state.end(), ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::sqrt_pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);

  check("adj_sqrt_pauli_z/0", values,
    [&](state_type& state, context const& ctx) { ket::gate::fused::adj_sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state, context const& ctx) { ket::gate::adj_sqrt_pauli_z(state.begin(), state.end(), ket::make_control(qubit_type{ctx.physical[control_positions[indices]]})...); },
    [&](state_type& state, context const& ctx, auto const& controls) {
      if (ctx.parallel)
        {
          execute_parallel([&](parallel_policy_type const policy, int const thread_index) {
            ket::gate::fused::runtime::ranges::adj_sqrt_pauli_z(policy, thread_index, state.begin(), ctx.block, ctx.unsorted, ctx.sorted, controls);
          });
        }
        else ket::gate::fused::runtime::ranges::adj_sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted, ctx.sorted, controls);
    },
    [&](state_type& state, context const& ctx, auto const& controls) { ket::gate::fused::runtime::adj_sqrt_pauli_z(state.begin(), ctx.block, ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{}, true);
#endif

  }

  template <unsigned int... control_values>
  auto run_pattern() -> void
  { run(std::integer_sequence<unsigned int, control_values...>{}, std::make_index_sequence<sizeof...(control_values)>{}); }
}

int main()
{
  run_pattern<0u>();
  run_pattern<0u, 1u>();
  run_pattern<1u, 0u>();
  run_pattern<0u, 0u>();
  run_pattern<0u, 1u, 0u>();
  run_pattern<1u, 0u, 1u>();
  run_pattern<0u, 0u, 0u>();
  run_pattern<1u, 1u, 1u>();
  run_pattern<0u, 1u, 0u, 1u>();
  std::cout << "Fused control numerical group " << KET_TEST_GROUP << ": " << checked << " state comparisons passed\n";
}
