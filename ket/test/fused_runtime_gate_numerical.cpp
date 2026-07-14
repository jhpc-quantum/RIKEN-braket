#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <ket/control.hpp>
#include <ket/gate/controlled_v.hpp>
#include <ket/gate/exponential_pauli_x.hpp>
#include <ket/gate/exponential_pauli_y.hpp>
#include <ket/gate/exponential_pauli_z.hpp>
#include <ket/gate/exponential_swap.hpp>
#include <ket/gate/fused/controlled_v.hpp>
#include <ket/gate/fused/exponential_pauli_x.hpp>
#include <ket/gate/fused/exponential_pauli_y.hpp>
#include <ket/gate/fused/exponential_pauli_z.hpp>
#include <ket/gate/fused/exponential_swap.hpp>
#include <ket/gate/fused/hadamard.hpp>
#include <ket/gate/fused/pauli_x.hpp>
#include <ket/gate/fused/pauli_y.hpp>
#include <ket/gate/fused/pauli_z.hpp>
#include <ket/gate/fused/phase_shift.hpp>
#include <ket/gate/fused/sqrt_pauli_z.hpp>
#include <ket/gate/fused/swap.hpp>
#include <ket/gate/fused/x_rotation_half_pi.hpp>
#include <ket/gate/fused/y_rotation_half_pi.hpp>
#include <ket/gate/hadamard.hpp>
#include <ket/gate/pauli_x.hpp>
#include <ket/gate/pauli_y.hpp>
#include <ket/gate/pauli_z.hpp>
#include <ket/gate/phase_shift.hpp>
#include <ket/gate/sqrt_pauli_z.hpp>
#include <ket/gate/swap.hpp>
#include <ket/gate/x_rotation_half_pi.hpp>
#include <ket/gate/y_rotation_half_pi.hpp>
#include <ket/qubit.hpp>
#include <ket/utility/exp_i.hpp>
#include <ket/utility/loop_n.hpp>

namespace
{
  using complex_type = std::complex<double>;
  using state_integer_type = std::uint64_t;
  using bit_integer_type = unsigned int;
  using qubit_type = ket::qubit<state_integer_type, bit_integer_type>;
  using control_qubit_type = ket::control<qubit_type>;

  using namespace ket::literals::control_literals;
  using namespace ket::literals::qubit_literals;

  constexpr auto num_fused_qubits = bit_integer_type{4u};
  constexpr auto state_size = std::size_t{1u} << num_fused_qubits;

  auto initial_state() -> std::vector<complex_type>
  {
    auto result = std::vector<complex_type>(state_size);
    for (auto index = std::size_t{0u}; index < result.size(); ++index)
      result[index] = complex_type{
        0.125 * static_cast<double>(index + 1u),
        -0.0625 * static_cast<double>((index * 3u + 1u) % 7u)};
    return result;
  }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
  using fused_qubits_type = std::vector<qubit_type>;

  auto fused_qubits() -> fused_qubits_type
  { return fused_qubits_type{0_q, 1_q, 2_q, 3_q}; }

  auto fused_qubits_with_sentinel() -> fused_qubits_type
  {
    auto result = fused_qubits();
    result.push_back(4_q);
    return result;
  }
#else
  using fused_qubits_type = std::vector<state_integer_type>;

  auto fused_qubits() -> fused_qubits_type
  {
    auto const qubits = std::vector<qubit_type>{0_q, 1_q, 2_q, 3_q};
    auto result = fused_qubits_type{};
    ket::gate::gate_detail::runtime::ranges::make_qubit_masks(qubits, std::back_inserter(result));
    return result;
  }

  auto fused_qubits_with_sentinel() -> fused_qubits_type
  {
    auto const qubits = std::vector<qubit_type>{0_q, 1_q, 2_q, 3_q};
    auto result = fused_qubits_type{};
    ket::gate::gate_detail::runtime::ranges::make_index_masks(qubits, std::back_inserter(result));
    return result;
  }
#endif

  auto make_controls(std::initializer_list<control_qubit_type> const control_qubits)
    -> std::vector<control_qubit_type>
  { return std::vector<control_qubit_type>{control_qubits}; }

  auto max_error(std::vector<complex_type> const& lhs, std::vector<complex_type> const& rhs) -> double
  {
    auto result = 0.0;
    for (auto index = std::size_t{0u}; index < lhs.size(); ++index)
      result = std::max(result, std::abs(lhs[index] - rhs[index]));
    return result;
  }

  template <typename FusedOperation, typename ReferenceOperation>
  auto run_case(std::string const& name, FusedOperation const& fused_operation, ReferenceOperation const& reference_operation) -> bool
  {
    auto state = initial_state();
    auto reference_state = state;
    auto const unsorted_fused_qubits = fused_qubits();
    auto const sorted_fused_qubits_with_sentinel = fused_qubits_with_sentinel();

    fused_operation(state, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
    reference_operation(reference_state);

    auto const error = max_error(state, reference_state);
    if (error < 1e-12)
      return true;

    std::cerr << name << " failed: max error = " << error << '\n';
    return false;
  }
}

int main()
{
  auto failed = false;
  auto const run = [&failed](bool const passed) { failed = failed or not passed; };

  run(run_case(
    "fused::runtime::ranges::hadamard",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::hadamard(state.begin(), state_integer_type{0u}, unsorted, sorted, 3_q, make_controls({0_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::hadamard(ket::utility::policy::make_sequential(), state, 3_q, make_controls({0_cq})); }));

  run(run_case(
    "fused::runtime::ranges::pauli_x",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::pauli_x(state.begin(), state_integer_type{0u}, unsorted, sorted, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::pauli_x(ket::utility::policy::make_sequential(), state, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); }));

  run(run_case(
    "fused::runtime::ranges::pauli_y",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::pauli_y(state.begin(), state_integer_type{0u}, unsorted, sorted, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::pauli_y(ket::utility::policy::make_sequential(), state, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); }));

  run(run_case(
    "fused::runtime::ranges::pauli_z",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::pauli_z(state.begin(), state_integer_type{0u}, unsorted, sorted, std::vector<qubit_type>{3_q}, make_controls({0_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::pauli_z(ket::utility::policy::make_sequential(), state, std::vector<qubit_type>{3_q}, make_controls({0_cq})); }));

  run(run_case(
    "fused::runtime::ranges::phase_shift_coeff",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::phase_shift_coeff(state.begin(), state_integer_type{0u}, unsorted, sorted, ket::utility::exp_i<complex_type>(0.375), 3_q, make_controls({0_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::phase_shift_coeff(ket::utility::policy::make_sequential(), state, ket::utility::exp_i<complex_type>(0.375), 3_q, make_controls({0_cq})); }));

  run(run_case(
    "fused::runtime::ranges::controlled_v",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::controlled_v(state.begin(), state_integer_type{0u}, unsorted, sorted, 0.25, 3_q, make_controls({0_cq, 1_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::controlled_v(ket::utility::policy::make_sequential(), state, 0.25, 3_q, make_controls({0_cq, 1_cq})); }));

  run(run_case(
    "fused::runtime::ranges::sqrt_pauli_z",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::sqrt_pauli_z(state.begin(), state_integer_type{0u}, unsorted, sorted, std::vector<qubit_type>{3_q}, make_controls({0_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::sqrt_pauli_z(ket::utility::policy::make_sequential(), state, std::vector<qubit_type>{3_q}, make_controls({0_cq})); }));

  run(run_case(
    "fused::runtime::ranges::swap",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::swap(state.begin(), state_integer_type{0u}, unsorted, sorted, 0_q, 3_q, make_controls({1_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::swap(ket::utility::policy::make_sequential(), state, 0_q, 3_q, make_controls({1_cq})); }));

  run(run_case(
    "fused::runtime::ranges::x_rotation_half_pi",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::x_rotation_half_pi(state.begin(), state_integer_type{0u}, unsorted, sorted, 3_q, make_controls({0_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::x_rotation_half_pi(ket::utility::policy::make_sequential(), state, 3_q, make_controls({0_cq})); }));

  run(run_case(
    "fused::runtime::ranges::y_rotation_half_pi",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::y_rotation_half_pi(state.begin(), state_integer_type{0u}, unsorted, sorted, 3_q, make_controls({0_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::y_rotation_half_pi(ket::utility::policy::make_sequential(), state, 3_q, make_controls({0_cq})); }));

  run(run_case(
    "fused::runtime::ranges::exponential_pauli_x",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::exponential_pauli_x(state.begin(), state_integer_type{0u}, unsorted, sorted, 0.25, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::exponential_pauli_x(ket::utility::policy::make_sequential(), state, 0.25, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); }));

  run(run_case(
    "fused::runtime::ranges::exponential_pauli_y",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::exponential_pauli_y(state.begin(), state_integer_type{0u}, unsorted, sorted, 0.25, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::exponential_pauli_y(ket::utility::policy::make_sequential(), state, 0.25, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); }));

  run(run_case(
    "fused::runtime::ranges::exponential_pauli_z",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::exponential_pauli_z(state.begin(), state_integer_type{0u}, unsorted, sorted, 0.25, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::exponential_pauli_z(ket::utility::policy::make_sequential(), state, 0.25, std::vector<qubit_type>{0_q, 3_q}, make_controls({1_cq})); }));

  run(run_case(
    "fused::runtime::ranges::exponential_swap",
    [](auto& state, auto const& unsorted, auto const& sorted)
    { ket::gate::fused::runtime::ranges::exponential_swap(state.begin(), state_integer_type{0u}, unsorted, sorted, 0.25, 0_q, 3_q, make_controls({1_cq})); },
    [](auto& state)
    { ket::gate::runtime::ranges::exponential_swap(ket::utility::policy::make_sequential(), state, 0.25, 0_q, 3_q, make_controls({1_cq})); }));

  if (not failed)
    std::cout << "runtime non-MPI fused gate numerical tests passed\n";

  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
