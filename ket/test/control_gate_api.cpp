#include <cassert>
#include <complex>
#include <cstdint>
#include <deque>
#include <vector>

#include <ket/gate/meta/num_control_qubits.hpp>
#include <ket/gate/toffoli.hpp>
#include <ket/gate/not_.hpp>
#include <ket/gate/utility/control_mask.hpp>

using qubit = ket::qubit<std::uint32_t, unsigned int>;
using positive = ket::control<qubit, 1>;
using negative = ket::control<qubit, 0>;
using runtime = ket::runtime_control<qubit>;

static_assert(ket::gate::meta::num_control_qubits<unsigned int, qubit, qubit, negative, positive, negative>::value == 3u,
              "Targets precede controls, but control values are not grouped");
static_assert(ket::gate::meta::num_control_qubits<unsigned int, qubit, runtime, runtime>::value == 2u,
              "Runtime controls are recognized");
static_assert(ket::gate::utility::negative_control_mask<unsigned int>(qubit{0u}, negative{3u}, positive{2u}, negative{1u}) == 10u,
              "Packed bits follow operand order, not physical wire order");
static_assert(ket::gate::utility::negative_control_mask<unsigned int>() == 0u, "Empty operand list");
static_assert(ket::gate::utility::is_control_of<negative, qubit>::value, "Compile-time type match");
static_assert(ket::gate::utility::is_control_of<runtime, qubit>::value, "Runtime type match");
static_assert(not ket::gate::utility::is_control_of<qubit, qubit>::value, "Targets are not controls");
static_assert(not ket::gate::utility::is_control_of<runtime, ket::qubit<>>::value, "Reject mismatched quantum digit types");
static_assert(negative{1u} == positive{1u} and negative{1u} != positive{2u}, "Mixed-value comparison uses wire positions");
static_assert(negative{1u} < positive{2u} and positive{2u} > negative{1u}, "Mixed-value ordering uses wire positions");

template <std::uint32_t control_value1, std::uint32_t control_value2>
auto check_toffoli() -> void
{
  for (auto target = 0u; target < 3u; ++target)
    for (auto control1 = 0u; control1 < 3u; ++control1)
    {
      if (target == control1) continue;
      auto const control2 = 3u - target - control1;
      for (auto basis = 0u; basis < 8u; ++basis)
      {
        auto state = std::vector<std::complex<double>>(8u);
        state[basis] = 1.0;
        auto const expected = (((basis >> control1) bitand 1u) == control_value1
                               and ((basis >> control2) bitand 1u) == control_value2)
          ? basis xor (1u << target) : basis;
        ket::gate::ranges::toffoli(state, qubit{target}, ket::control<qubit, control_value1>{control1}, ket::control<qubit, control_value2>{control2});
        for (auto i = 0u; i < 8u; ++i) assert(state[i] == std::complex<double>{i == expected ? 1.0 : 0.0});
        ket::gate::adj_toffoli(state.begin(), state.end(), qubit{target}, ket::control<qubit, control_value1>{control1}, ket::control<qubit, control_value2>{control2});
        for (auto i = 0u; i < 8u; ++i) assert(state[i] == std::complex<double>{i == basis ? 1.0 : 0.0});
        ket::gate::not_(state.begin(), state.end(), qubit{target}, ket::control<qubit, control_value1>{control1}, ket::control<qubit, control_value2>{control2});
        for (auto i = 0u; i < 8u; ++i) assert(state[i] == std::complex<double>{i == expected ? 1.0 : 0.0});
      }
    }
}

int main()
{
  check_toffoli<0u, 0u>();
  check_toffoli<0u, 1u>();
  check_toffoli<1u, 0u>();
  check_toffoli<1u, 1u>();
  auto controls = std::deque<runtime>{runtime{qubit{3u}, 0}, runtime{qubit{2u}, 1}, runtime{qubit{1u}, 0}};
  assert(ket::gate::utility::negative_control_mask_from_range<unsigned int>(controls, 2u) == 20u);
  auto empty = std::vector<runtime>{};
  assert(ket::gate::utility::negative_control_mask_from_range<unsigned int>(empty, 0u) == 0u);
  auto threw = false;
  try { ket::gate::utility::negative_control_mask_from_range<unsigned int>(controls, 32u); }
  catch (std::out_of_range const&) { threw = true; }
  assert(threw);
}
