#include <algorithm>
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
#include <ket/gate/hadamard.hpp>
#include <ket/gate/pauli_x.hpp>
#include <ket/gate/pauli_y.hpp>
#include <ket/gate/pauli_z.hpp>
#include <ket/gate/phase_shift.hpp>
#include <ket/gate/sqrt_pauli_x.hpp>
#include <ket/gate/sqrt_pauli_y.hpp>
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

  constexpr auto total_qubits = bit_integer_type{4u};
  constexpr auto state_size = std::size_t{1u} << total_qubits;

  auto initial_state() -> std::vector<complex_type>
  {
    auto result = std::vector<complex_type>(state_size);
    for (auto index = std::size_t{0u}; index < result.size(); ++index)
      result[index] = complex_type{
        0.125 * static_cast<double>(index + 1u),
        -0.0625 * static_cast<double>((index * 3u + 1u) % 7u)};
    return result;
  }

  auto make_controls(std::initializer_list<control_qubit_type> const control_qubits)
    -> std::vector<control_qubit_type>
  { return std::vector<control_qubit_type>{control_qubits}; }

  auto max_error(
    std::vector<complex_type> const& lhs,
    std::vector<complex_type> const& rhs)
    -> double
  {
    auto result = 0.0;
    for (auto index = std::size_t{0u}; index < lhs.size(); ++index)
      result = std::max(result, std::abs(lhs[index] - rhs[index]));
    return result;
  }

  template <typename RuntimeOperation, typename ReferenceOperation>
  auto run_case(
    std::string const& name,
    RuntimeOperation const& runtime_operation,
    ReferenceOperation const& reference_operation)
    -> bool
  {
    auto state = initial_state();
    auto reference_state = state;

    runtime_operation(state);
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
  using namespace ket::literals::control_literals;
  using namespace ket::literals::qubit_literals;

  auto failed = false;
  auto const run = [&failed](bool const passed) { failed = failed or not passed; };

  run(run_case(
    "runtime::ranges::hadamard",
    [](auto& state)
    {
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::hadamard(
        ket::utility::policy::make_sequential(), state, 3_q, controls);
    },
    [](auto& state)
    { ket::gate::hadamard(ket::utility::policy::make_sequential(), state.begin(), state.end(), 3_q, 0_cq); }));

  run(run_case(
    "runtime::ranges::pauli_x",
    [](auto& state)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::pauli_x(
        ket::utility::policy::make_sequential(), state, target_qubits, controls);
    },
    [](auto& state)
    { ket::gate::pauli_x(ket::utility::policy::make_sequential(), state.begin(), state.end(), 0_q, 3_q, 1_cq); }));

  run(run_case(
    "runtime::ranges::pauli_y",
    [](auto& state)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::pauli_y(
        ket::utility::policy::make_sequential(), state, target_qubits, controls);
    },
    [](auto& state)
    { ket::gate::pauli_y(ket::utility::policy::make_sequential(), state.begin(), state.end(), 0_q, 3_q, 1_cq); }));

  run(run_case(
    "runtime::ranges::pauli_z",
    [](auto& state)
    {
      auto const target_qubits = std::vector<qubit_type>{3_q};
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::pauli_z(
        ket::utility::policy::make_sequential(), state, target_qubits, controls);
    },
    [](auto& state)
    { ket::gate::pauli_z(ket::utility::policy::make_sequential(), state.begin(), state.end(), 3_q, 0_cq); }));

  run(run_case(
    "runtime::ranges::phase_shift_coeff",
    [](auto& state)
    {
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::phase_shift_coeff(
        ket::utility::policy::make_sequential(), state,
        ket::utility::exp_i<complex_type>(0.375), 3_q, controls);
    },
    [](auto& state)
    { ket::gate::phase_shift_coeff(ket::utility::policy::make_sequential(), state.begin(), state.end(), ket::utility::exp_i<complex_type>(0.375), 3_q, 0_cq); }));

  run(run_case(
    "runtime::ranges::controlled_v",
    [](auto& state)
    {
      auto const controls = make_controls({0_cq, 1_cq});
      ket::gate::runtime::ranges::controlled_v(
        ket::utility::policy::make_sequential(), state, 0.25, 3_q, controls);
    },
    [](auto& state)
    { ket::gate::controlled_v(ket::utility::policy::make_sequential(), state.begin(), state.end(), 0.25, 3_q, 0_cq, 1_cq); }));

  run(run_case(
    "runtime::ranges::sqrt_pauli_z",
    [](auto& state)
    {
      auto const target_qubits = std::vector<qubit_type>{3_q};
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::sqrt_pauli_z(
        ket::utility::policy::make_sequential(), state, target_qubits, controls);
    },
    [](auto& state)
    { ket::gate::sqrt_pauli_z(ket::utility::policy::make_sequential(), state.begin(), state.end(), 3_q, 0_cq); }));

  run(run_case(
    "runtime::ranges::swap",
    [](auto& state)
    {
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::swap(
        ket::utility::policy::make_sequential(), state, 0_q, 3_q, controls);
    },
    [](auto& state)
    { ket::gate::swap(ket::utility::policy::make_sequential(), state.begin(), state.end(), 0_q, 3_q, 1_cq); }));

  run(run_case(
    "runtime::ranges::x_rotation_half_pi",
    [](auto& state)
    {
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::x_rotation_half_pi(
        ket::utility::policy::make_sequential(), state, 3_q, controls);
    },
    [](auto& state)
    { ket::gate::x_rotation_half_pi(ket::utility::policy::make_sequential(), state.begin(), state.end(), 3_q, 0_cq); }));

  run(run_case(
    "runtime::ranges::y_rotation_half_pi",
    [](auto& state)
    {
      auto const controls = make_controls({0_cq});
      ket::gate::runtime::ranges::y_rotation_half_pi(
        ket::utility::policy::make_sequential(), state, 3_q, controls);
    },
    [](auto& state)
    { ket::gate::y_rotation_half_pi(ket::utility::policy::make_sequential(), state.begin(), state.end(), 3_q, 0_cq); }));

  run(run_case(
    "runtime::ranges::exponential_pauli_x",
    [](auto& state)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::exponential_pauli_x(
        ket::utility::policy::make_sequential(), state, 0.25, target_qubits, controls);
    },
    [](auto& state)
    { ket::gate::exponential_pauli_x(ket::utility::policy::make_sequential(), state.begin(), state.end(), 0.25, 0_q, 3_q, 1_cq); }));

  run(run_case(
    "runtime::ranges::exponential_pauli_y",
    [](auto& state)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::exponential_pauli_y(
        ket::utility::policy::make_sequential(), state, 0.25, target_qubits, controls);
    },
    [](auto& state)
    { ket::gate::exponential_pauli_y(ket::utility::policy::make_sequential(), state.begin(), state.end(), 0.25, 0_q, 3_q, 1_cq); }));

  run(run_case(
    "runtime::ranges::exponential_pauli_z",
    [](auto& state)
    {
      auto const target_qubits = std::vector<qubit_type>{0_q, 3_q};
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::exponential_pauli_z(
        ket::utility::policy::make_sequential(), state, 0.25, target_qubits, controls);
    },
    [](auto& state)
    { ket::gate::exponential_pauli_z(ket::utility::policy::make_sequential(), state.begin(), state.end(), 0.25, 0_q, 3_q, 1_cq); }));

  run(run_case(
    "runtime::ranges::exponential_swap",
    [](auto& state)
    {
      auto const controls = make_controls({1_cq});
      ket::gate::runtime::ranges::exponential_swap(
        ket::utility::policy::make_sequential(), state, 0.25, 0_q, 3_q, controls);
    },
    [](auto& state)
    { ket::gate::exponential_swap(ket::utility::policy::make_sequential(), state.begin(), state.end(), 0.25, 0_q, 3_q, 1_cq); }));

  if (not failed)
    std::cout << "runtime non-MPI gate numerical tests passed\n";

  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
