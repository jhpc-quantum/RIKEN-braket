// Build with BRA_NO_MPI and the fused_gate, fused_controlled_hadamard,
// fused_multi_controlled_hadamard and fused_controlled_u1 sources.
// Exercise with/without KET_USE_BIT_MASKS_EXPLICITLY and with OpenMP enabled.
#include <algorithm>
#include <cmath>
#include <complex>
#include <iterator>
#include <stdexcept>
#include <vector>

#include <ket/gate/gate.hpp>
#include <ket/utility/parallel/loop_n.hpp>
#include <bra/types.hpp>
#include <bra/fused_gate/fused_controlled_hadamard.hpp>
#include <bra/fused_gate/fused_multi_controlled_hadamard.hpp>
#include <bra/fused_gate/fused_controlled_u1.hpp>

namespace
{
  using iterator = bra::data_type::iterator;
  auto const wires = std::vector<bra::qubit_type>{
    bra::qubit_type{3u}, bra::qubit_type{1u}, bra::qubit_type{4u}, bra::qubit_type{0u}, bra::qubit_type{2u}};
  auto const mapping = std::vector<bra::bit_integer_type>{3u, 1u, 4u, 0u, 2u};
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
  auto const unsorted = wires;
  auto const sorted = std::vector<bra::qubit_type>{
    bra::qubit_type{0u}, bra::qubit_type{1u}, bra::qubit_type{2u},
    bra::qubit_type{3u}, bra::qubit_type{4u}, bra::qubit_type{5u}};
#else
  auto masks(bool const indices) -> std::vector<bra::state_integer_type>
  {
    auto result = std::vector<bra::state_integer_type>{};
    if (indices)
      ket::gate::gate_detail::runtime::ranges::make_index_masks(wires, std::back_inserter(result));
    else
      ket::gate::gate_detail::runtime::ranges::make_qubit_masks(wires, std::back_inserter(result));
    return result;
  }
  auto const unsorted = masks(false);
  auto const sorted = masks(true);
#endif

  auto initial() -> bra::data_type
  {
    auto result = bra::data_type(32u);
    for (auto index = 0u; index < result.size(); ++index)
      result[index] = bra::complex_type{(index + 1) / 64.0, (index % 7) / 32.0};
    return result;
  }

  auto compare(bra::data_type const& actual, bra::data_type const& expected) -> void
  {
    for (auto index = 0u; index < actual.size(); ++index)
      if (std::abs(actual[index] - expected[index]) > 1e-12)
        throw std::runtime_error{"fused negative control remapping mismatch"};
  }

  template <typename Gate>
  auto check(Gate const& gate, bra::data_type const& expected) -> void
  {
    auto state = initial();
    gate.call(state.begin(), bra::state_integer_type{0u}, unsorted, sorted, mapping);
    compare(state, expected);
    state = initial();
    auto const policy = ket::utility::policy::make_parallel(3u);
    ket::utility::execute(policy, [&](int const thread_index, auto& executor)
    {
      gate.call_in_execute(policy, executor, thread_index,
        state.begin(), bra::state_integer_type{0u}, unsorted, sorted, mapping);
    });
    compare(state, expected);
  }

  auto reference(std::vector<bra::control_qubit_type> const& controls, bool const phase) -> bra::data_type
  {
    auto const input = initial();
    auto expected = input;
    for (auto index = 0u; index < expected.size(); ++index)
    {
      if (not std::all_of(controls.begin(), controls.end(), [index](auto const control)
          { return ((index >> static_cast<unsigned int>(control.qubit())) bitand 1u) == control.value(); }))
        continue;
      if (phase)
        expected[index] *= std::exp(bra::complex_type{0, 0.3});
      else if ((index bitand 4u) == 0u)
      {
        expected[index] = (input[index] + input[index | 4u]) / std::sqrt(2.0);
        expected[index | 4u] = (input[index] - input[index | 4u]) / std::sqrt(2.0);
      }
    }
    return expected;
  }
}

int main()
{
  for (auto first = 0u; first != 2u; ++first)
    for (auto second = 0u; second != 2u; ++second)
    {
      auto const controls = std::vector<bra::control_qubit_type>{
        {bra::qubit_type{4u}, first}, {bra::qubit_type{0u}, second}};
      bra::fused_gate::fused_controlled_hadamard<iterator> single{bra::qubit_type{2u}, controls[0]};
      bra::fused_gate::fused_multi_controlled_hadamard<iterator> multiple{bra::qubit_type{2u}, controls};
      bra::fused_gate::fused_controlled_u1<iterator> phase{0.3, controls[0], controls[1]};
      check(single, reference({controls[0]}, false));
      check(multiple, reference(controls, false));
      check(phase, reference(controls, true));
      auto terms = std::vector<bra::fused_gate::phase_shift_term>{};
      if (phase.append_phase_shift_term(terms, mapping, 0u) != (first == 1u and second == 1u))
        throw std::runtime_error{"negative controls used positive-only phase batching"};
    }
}
