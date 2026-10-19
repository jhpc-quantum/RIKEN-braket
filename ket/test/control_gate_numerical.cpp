// python3 ket/test/control_gate_numerical.py
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <ket/gate/sqrt_pauli_x.hpp>
#include <ket/gate/sqrt_pauli_y.hpp>
#include <ket/gate/exponential_pauli_z.hpp>
#include <ket/gate/not_.hpp>
#include <ket/gate/hadamard.hpp>
#include <ket/gate/pauli_y.hpp>
#include <ket/gate/pauli_z.hpp>
#include <ket/gate/phase_shift.hpp>
#include <ket/gate/exponential_pauli_x.hpp>
#include <ket/gate/sqrt_pauli_z.hpp>
#include <ket/gate/controlled_not.hpp>
#include <ket/gate/exponential_pauli_y.hpp>
#include <ket/gate/exponential_swap.hpp>
#include <ket/gate/y_rotation_half_pi.hpp>
#include <ket/gate/controlled_v.hpp>
#include <ket/gate/pauli_x.hpp>
#include <ket/gate/x_rotation_half_pi.hpp>
#include <ket/gate/toffoli.hpp>
#include <ket/gate/controlled_phase_shift.hpp>
#include <ket/gate/swap.hpp>
#include <ket/runtime_control.hpp>

#ifndef KET_TEST_GROUP
# define KET_TEST_GROUP 1
#endif

namespace
{
  using qubit_type = ket::qubit<std::uint64_t, unsigned int>;
  using complex_type = std::complex<double>;
  using state_type = std::vector<complex_type>;
  using runtime_control = ket::runtime_control<qubit_type>;
  constexpr auto state_size = std::size_t{128u};
  constexpr std::array<unsigned int, 3u> target_positions{{3u, 0u, 5u}};
  constexpr std::array<unsigned int, 4u> control_positions{{4u, 1u, 6u, 2u}};
  auto checked = std::size_t{0u};

  auto flip_controls(state_type& state, std::size_t const bits) -> void
  {
    for (auto i = std::size_t{0u}; i < state.size(); ++i)
      if (i < (i xor bits))
        std::swap(state[i], state[i xor bits]);
  }

  auto compare(std::string const& name, state_type const& actual, state_type const& expected) -> void
  {
    for (auto i = std::size_t{0u}; i < actual.size(); ++i)
      if (std::abs(actual[i] - expected[i]) > 1e-11)
      {
        std::cerr << name << ": amplitude mismatch at " << i << ": "
                  << actual[i] << " != " << expected[i] << '\n';
        std::exit(1);
      }
    ++checked;
  }

  template <typename RangeGate>
  auto check_homogeneous(RangeGate const& gate, state_type const& initial,
                        state_type const& expected, std::size_t const count, std::true_type) -> void
  {
    auto controls = std::vector<ket::control<qubit_type, 0u>>{};
    for (auto i = std::size_t{0u}; i < count; ++i)
      controls.emplace_back(qubit_type{control_positions[i]});
    auto actual = initial;
    gate(actual, controls);
    compare("homogeneous negative range", actual, expected);
  }

  template <typename RangeGate>
  auto check_homogeneous(RangeGate const&, state_type const&, state_type const&, std::size_t, std::false_type) -> void { }

  template <typename Native, typename Positive, typename RangeGate, typename IteratorGate, typename AllNegative>
  auto check(std::string const& name, std::vector<unsigned int> const& values, Native const& native,
             Positive const& positive, RangeGate const& range_gate, IteratorGate const& iterator_gate,
             AllNegative const all_negative) -> void
  {
    auto controls = std::vector<runtime_control>{};
    auto bits = std::size_t{0u};
    for (auto i = std::size_t{0u}; i < values.size(); ++i)
    {
      controls.emplace_back(qubit_type{control_positions[i]}, values[i]);
      if (values[i] == 0u) bits |= std::size_t{1u} << control_positions[i];
    }
    // Every basis state plus a complex superposition. This includes untouched
    // spectator bits, inactive controls, and relative phases (no phase alignment).
    for (auto basis = std::size_t{0u}; basis <= state_size; ++basis)
    {
      auto initial = state_type(state_size);
      if (basis < state_size) initial[basis] = complex_type{1.0};
      else
        for (auto i = std::size_t{0u}; i < state_size; ++i)
          initial[i] = complex_type{0.01 * double(i + 1u), -0.02 * double((i * 3u) % 17u)};
      auto expected = initial;
      flip_controls(expected, bits);
      positive(expected);
      flip_controls(expected, bits);
      auto actual = initial;
      native(actual);
      compare(name + "/variadic", actual, expected);
      actual = initial;
      range_gate(actual, controls);
      compare(name + "/range", actual, expected);
      actual = initial;
      iterator_gate(actual, controls);
      compare(name + "/iterators", actual, expected);
      check_homogeneous(range_gate, initial, expected, values.size(), all_negative);
    }
  }

  template <unsigned int... control_values, std::size_t... indices>
  auto run(std::integer_sequence<unsigned int, control_values...>, std::index_sequence<indices...>) -> void
  {
    auto const negative_bits = std::vector<unsigned int>{control_values...};
    constexpr auto all_negative = ket::gate::utility::negative_control_mask<unsigned int>(
      ket::control<qubit_type, control_values>{control_positions[indices]}...) == (1u << sizeof...(control_values)) - 1u;
#if KET_TEST_GROUP == 1
  check("hadamard/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::hadamard(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::hadamard(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::hadamard(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::hadamard(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_hadamard/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_hadamard(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_hadamard(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_hadamard(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_hadamard(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("not_/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::not_(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::not_(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::not_(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::not_(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_not_/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_not_(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_not_(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_not_(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_not_(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("controlled_not/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::controlled_not(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::controlled_not(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::controlled_not(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::controlled_not(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_controlled_not/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_controlled_not(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_controlled_not(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_controlled_not(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_controlled_not(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("pauli_x/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::pauli_x(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::pauli_x(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_pauli_x/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_pauli_x(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::adj_pauli_x(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("pauli_y/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::pauli_y(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::pauli_y(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_pauli_y/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_pauli_y(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::adj_pauli_y(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("pauli_z/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::pauli_z(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::pauli_z(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_pauli_z/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_pauli_z(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::adj_pauli_z(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("sqrt_pauli_x/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::sqrt_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::sqrt_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::sqrt_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::sqrt_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_sqrt_pauli_x/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_sqrt_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("sqrt_pauli_y/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::sqrt_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::sqrt_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::sqrt_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::sqrt_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_sqrt_pauli_y/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_sqrt_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("sqrt_pauli_z/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::sqrt_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::sqrt_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::sqrt_pauli_z(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::sqrt_pauli_z(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_sqrt_pauli_z/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_z(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::adj_sqrt_pauli_z(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("x_rotation_half_pi/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::x_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::x_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::x_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::x_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_x_rotation_half_pi/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_x_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_x_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_x_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_x_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("y_rotation_half_pi/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::y_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::y_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::y_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::y_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_y_rotation_half_pi/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_y_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_y_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_y_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_y_rotation_half_pi(state.begin(), state.end(), qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
#endif
#if KET_TEST_GROUP == 2
  check("phase_shift/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::phase_shift(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::phase_shift(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::phase_shift(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::phase_shift(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_phase_shift/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_phase_shift(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_phase_shift(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_phase_shift(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_phase_shift(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("phase_shift_coeff/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_phase_shift_coeff/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("phase_shift2/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_phase_shift2/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("phase_shift3/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_phase_shift3/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("exponential_pauli_x/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::exponential_pauli_x(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::exponential_pauli_x(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_exponential_pauli_x/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("exponential_pauli_x_coeff/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_exponential_pauli_x_coeff/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::adj_exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("exponential_pauli_y/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::exponential_pauli_y(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::exponential_pauli_y(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_exponential_pauli_y/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("exponential_pauli_y_coeff/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_exponential_pauli_y_coeff/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::adj_exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("exponential_pauli_z/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::exponential_pauli_z(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::exponential_pauli_z(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_exponential_pauli_z/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("exponential_pauli_z_coeff/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_exponential_pauli_z_coeff/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, std::vector<qubit_type>{qubit_type{target_positions[0]}}, controls); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}}; ket::gate::runtime::adj_exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("controlled_v/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::controlled_v(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::controlled_v(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::controlled_v(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::controlled_v(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_controlled_v/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_controlled_v(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_controlled_v(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_controlled_v(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_controlled_v(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("controlled_v_coeff/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_controlled_v_coeff/1 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
#endif
#if KET_TEST_GROUP == 3
  check("pauli_x/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::pauli_x(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::pauli_x(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("adj_pauli_x/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_pauli_x(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::adj_pauli_x(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_pauli_x(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("pauli_y/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::pauli_y(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::pauli_y(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("adj_pauli_y/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_pauli_y(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::adj_pauli_y(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_pauli_y(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("pauli_z/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::pauli_z(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::pauli_z(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("sqrt_pauli_z/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::sqrt_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::sqrt_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::sqrt_pauli_z(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::sqrt_pauli_z(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("adj_pauli_z/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::adj_pauli_z(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_pauli_z(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("adj_sqrt_pauli_z/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_z(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::adj_sqrt_pauli_z(state.begin(), state.end(), targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_z(state.begin(), state.end(), std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("exponential_pauli_x/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::exponential_pauli_x(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::exponential_pauli_x(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("adj_exponential_pauli_x/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("exponential_pauli_y/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::exponential_pauli_y(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::exponential_pauli_y(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("adj_exponential_pauli_y/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("exponential_pauli_z/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::exponential_pauli_z(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::exponential_pauli_z(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("adj_exponential_pauli_z/3 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { auto const targets = std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}; ket::gate::runtime::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, targets.begin(), targets.end(), controls.begin(), controls.end()); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, std::vector<qubit_type>{qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, qubit_type{target_positions[2]}}, controls); },
    std::integral_constant<bool, all_negative>{});
  check("swap/2 targets", negative_bits,
    [&](state_type& state) { ket::gate::swap(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::swap(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::swap(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::swap(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_swap/2 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_swap(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_swap(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_swap(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_swap(state.begin(), state.end(), qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("exponential_swap/2 targets", negative_bits,
    [&](state_type& state) { ket::gate::exponential_swap(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::exponential_swap(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::exponential_swap(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::exponential_swap(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_exponential_swap/2 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_exponential_swap(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_exponential_swap(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_exponential_swap(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_exponential_swap(state.begin(), state.end(), 0.37, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("exponential_swap_coeff/2 targets", negative_bits,
    [&](state_type& state) { ket::gate::exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_exponential_swap_coeff/2 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit_type{target_positions[0]}, qubit_type{target_positions[1]}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
#endif
#if KET_TEST_GROUP == 4
  check("phase_shift/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::phase_shift(state.begin(), state.end(), 0.37, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::phase_shift(state.begin(), state.end(), 0.37, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::phase_shift(state.begin(), state.end(), 0.37, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::phase_shift(state.begin(), state.end(), 0.37, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_phase_shift/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_phase_shift(state.begin(), state.end(), 0.37, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_phase_shift(state.begin(), state.end(), 0.37, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_phase_shift(state.begin(), state.end(), 0.37, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_phase_shift(state.begin(), state.end(), 0.37, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("phase_shift_coeff/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_phase_shift_coeff/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("controlled_phase_shift/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::controlled_phase_shift(state.begin(), state.end(), 0.37, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::controlled_phase_shift(state.begin(), state.end(), 0.37, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::controlled_phase_shift(state.begin(), state.end(), 0.37, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::controlled_phase_shift(state.begin(), state.end(), 0.37, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_controlled_phase_shift/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_controlled_phase_shift(state.begin(), state.end(), 0.37, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_controlled_phase_shift(state.begin(), state.end(), 0.37, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_controlled_phase_shift(state.begin(), state.end(), 0.37, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_controlled_phase_shift(state.begin(), state.end(), 0.37, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("controlled_phase_shift_coeff/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_controlled_phase_shift_coeff/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("pauli_z/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::pauli_z(state.begin(), state.end(), ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::pauli_z(state.begin(), state.end(), ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::pauli_z(state.begin(), state.end(), controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::pauli_z(state.begin(), state.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_pauli_z/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_pauli_z(state.begin(), state.end(), ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_pauli_z(state.begin(), state.end(), ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_pauli_z(state.begin(), state.end(), controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_pauli_z(state.begin(), state.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("sqrt_pauli_z/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::sqrt_pauli_z(state.begin(), state.end(), ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::sqrt_pauli_z(state.begin(), state.end(), ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::sqrt_pauli_z(state.begin(), state.end(), controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::sqrt_pauli_z(state.begin(), state.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
  check("adj_sqrt_pauli_z/0 targets", negative_bits,
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_z(state.begin(), state.end(), ket::control<qubit_type, control_values>{control_positions[indices]}...); },
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_z(state.begin(), state.end(), ket::make_control(qubit_type{control_positions[indices]})...); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_z(state.begin(), state.end(), controls); },
    [&](state_type& state, auto const& controls) { ket::gate::runtime::adj_sqrt_pauli_z(state.begin(), state.end(), controls.begin(), controls.end()); },
    std::integral_constant<bool, all_negative>{});
#endif
#if KET_TEST_GROUP == 4
    // The dedicated Toffoli helper is variadic-only.
    auto actual = state_type(state_size, complex_type{0.3, -0.2});
    for (auto i = std::size_t{0u}; i < state_size; ++i) actual[i] *= double(i + 1u);
    auto expected = actual;
    ket::gate::not_(expected.begin(), expected.end(), qubit_type{3u},
                   ket::control<qubit_type, 0>{4u}, ket::control<qubit_type, 1>{1u});
    ket::gate::toffoli(actual.begin(), actual.end(), qubit_type{3u},
                      ket::control<qubit_type, 0>{4u}, ket::control<qubit_type, 1>{1u});
    compare("toffoli", actual, expected);
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
  std::cout << "Control numerical group " << KET_TEST_GROUP << ": " << checked << " comparisons passed\n";
}
