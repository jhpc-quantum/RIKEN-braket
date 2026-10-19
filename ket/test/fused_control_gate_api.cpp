#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <vector>

#include <ket/gate/fused/gate.hpp>
#include <ket/gate/fused/hadamard.hpp>
#include <ket/gate/fused/not_.hpp>
#include <ket/gate/fused/toffoli.hpp>
#include <ket/gate/fused/phase_shift.hpp>
#include <ket/gate/fused/sqrt_pauli_z.hpp>
#include <ket/gate/fused/swap.hpp>
#include <ket/gate/global_phase.hpp>
#include <ket/gate/hadamard.hpp>
#include <ket/gate/phase_shift.hpp>

using qubit = ket::qubit<std::uint32_t, unsigned int>;
using state_type = std::vector<std::complex<double>>;
using iterator = state_type::iterator;
constexpr auto num_fused_qubits = std::size_t{3u};

struct context
{
  std::array<unsigned int, 3u> physical{{3u, 0u, 2u}};
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
  std::array<qubit, 3u> unsorted{{qubit{3u}, qubit{0u}, qubit{2u}}};
  std::array<qubit, 4u> sorted{{qubit{0u}, qubit{2u}, qubit{3u}, qubit{4u}}};
#else
  std::array<std::uint32_t, 3u> unsorted{{8u, 1u, 4u}};
  // Missing outer wire 1: packed outer bit 0 maps to global bit 1.
  std::array<std::uint32_t, 4u> sorted{{0u, 1u, 0u, ~std::uint32_t{1u}}};
#endif
};

template <std::uint32_t control_value1, std::uint32_t control_value2>
auto check_toffoli(context const& ctx) -> void
{
  for (auto target = 0u; target < 3u; ++target)
    for (auto control1 = 0u; control1 < 3u; ++control1)
    {
      if (target == control1) continue;
      auto const control2 = 3u - target - control1;
      for (auto basis = 0u; basis < 16u; ++basis)
      {
        auto state = state_type(16u);
        state[basis] = 1.0;
        auto const expected = (((basis >> ctx.physical[control1]) bitand 1u) == control_value1
                               and ((basis >> ctx.physical[control2]) bitand 1u) == control_value2)
          ? basis xor (1u << ctx.physical[target]) : basis;
        for (auto block = std::uint32_t{0u}; block < 2u; ++block)
          ket::gate::fused::toffoli(state.begin(), block, ctx.unsorted, ctx.sorted, qubit{target},
            ket::control<qubit, control_value1>{control1}, ket::control<qubit, control_value2>{control2});
        for (auto i = 0u; i < 16u; ++i) assert(state[i] == std::complex<double>{i == expected ? 1.0 : 0.0});
        for (auto block = std::uint32_t{0u}; block < 2u; ++block)
          ket::gate::fused::adj_toffoli(state.begin(), block, ctx.unsorted, ctx.sorted, qubit{target},
            ket::control<qubit, control_value1>{control1}, ket::control<qubit, control_value2>{control2});
        for (auto i = 0u; i < 16u; ++i) assert(state[i] == std::complex<double>{i == basis ? 1.0 : 0.0});
        for (auto block = std::uint32_t{0u}; block < 2u; ++block)
          ket::gate::fused::not_(state.begin(), block, ctx.unsorted, ctx.sorted, qubit{target},
            ket::control<qubit, control_value1>{control1}, ket::control<qubit, control_value2>{control2});
        for (auto i = 0u; i < 16u; ++i) assert(state[i] == std::complex<double>{i == expected ? 1.0 : 0.0});
      }
    }
}

auto compare(state_type const& actual, state_type const& expected) -> void
{ for (auto i = 0u; i < actual.size(); ++i) assert(std::abs(actual[i] - expected[i]) < 1e-12); }

int main()
{
  auto const ctx = context{};
  check_toffoli<0u, 0u>(ctx);
  check_toffoli<0u, 1u>(ctx);
  check_toffoli<1u, 0u>(ctx);
  check_toffoli<1u, 1u>(ctx);

  auto initial = state_type(16u);
  for (auto i = 0u; i < 16u; ++i) initial[i] = {0.01 * (i + 1u), -0.02 * ((i * 3u) % 7u)};
  auto actual = initial;
  auto expected = initial;
  ket::gate::hadamard(expected.begin(), expected.end(), qubit{ctx.physical[1u]},
                       ket::control<qubit, 0u>{ctx.physical[0u]});
  for (auto block = std::uint32_t{0u}; block < 2u; ++block)
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    ket::gate::fused::hadamard<iterator, std::uint32_t, unsigned int, num_fused_qubits>(
#else
    ket::gate::fused::hadamard<iterator, std::uint32_t, num_fused_qubits>(
#endif
      actual.begin(), block, ctx.unsorted, ctx.sorted, qubit{1u}, ket::control<qubit, 0u>{0u});
  compare(actual, expected); // Existing explicit template argument positions.

  auto const negative_controls = std::vector<ket::runtime_control<qubit>>{
    ket::runtime_control<qubit>{qubit{0u}, 0}};
  actual = initial; expected = initial;
  ket::gate::phase_shift2(expected.begin(), expected.end(), 0.21, -0.34,
    qubit{ctx.physical[1u]}, ket::control<qubit, 0u>{ctx.physical[0u]});
  for (auto block = std::uint32_t{0u}; block < 2u; ++block)
    ket::gate::fused::runtime::phase_shift2(actual.begin(), block,
      ctx.unsorted.begin(), ctx.unsorted.end(), ctx.sorted.begin(), ctx.sorted.end(),
      0.21, -0.34, qubit{1u}, negative_controls.begin(), negative_controls.end());
  compare(actual, expected);

  auto const empty = std::vector<ket::runtime_control<qubit>>{};
  auto const mixed = std::vector<ket::runtime_control<qubit>>{
    ket::runtime_control<qubit>{qubit{0u}, 0}, ket::runtime_control<qubit>{qubit{2u}, 1}};
  assert(ket::utility::runtime::ranges::all_in_state_vector(3u, mixed));
  assert(not ket::utility::runtime::ranges::all_in_state_vector(2u, mixed));
  assert(ket::utility::all_in_state_vector(3u, mixed[0], mixed[1]));
  assert(not ket::utility::all_in_state_vector(2u, mixed[0], mixed[1]));
  assert(ket::utility::all_in_state_vector(3u, 1u, 2u));
  assert(not ket::utility::all_in_state_vector(2u, 1u, 2u));
  assert(ket::utility::runtime::ranges::all_in_state_vector(3u, std::vector<unsigned int>{1u, 2u}));
  actual = initial; expected = initial;
  ket::gate::hadamard(expected.begin(), expected.end(), qubit{ctx.physical[1u]});
  for (auto block = std::uint32_t{0u}; block < 2u; ++block)
    ket::gate::fused::runtime::ranges::hadamard(actual.begin(), block, ctx.unsorted, ctx.sorted, qubit{1u}, empty);
  compare(actual, expected);
  actual = initial; expected = initial;
  ket::gate::global_phase(expected.begin(), expected.end(), 0.37);
  for (auto block = std::uint32_t{0u}; block < 2u; ++block)
    ket::gate::fused::runtime::ranges::phase_shift(actual.begin(), block, ctx.unsorted, ctx.sorted, 0.37, empty);
  compare(actual, expected);
  actual = initial; expected = initial;
  for (auto& amplitude : expected) amplitude *= std::complex<double>{0.0, 1.0};
  for (auto block = std::uint32_t{0u}; block < 2u; ++block)
    ket::gate::fused::runtime::ranges::sqrt_pauli_z(actual.begin(), block, ctx.unsorted, ctx.sorted, empty);
  compare(actual, expected);

  // The generic two-operand fused engine must accept heterogeneous operands.
  auto calls = 0u;
  ket::gate::fused::gate<3u>(actual.begin(),
    [&](auto, auto, auto const&, auto const&) { ++calls; },
    qubit{0u}, ket::control<qubit, 0u>{1u});
  assert(calls == 2u);
}
