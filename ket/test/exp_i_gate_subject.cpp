#include <array>
#include <complex>
#include <cstdint>

#include <ket/gate/phase_shift.hpp>
#include <ket/gate/fused/phase_shift.hpp>

#if KET_TEST_PRECISION == 0
using real_type = float;
#elif KET_TEST_PRECISION == 1
using real_type = double;
#else
using real_type = long double;
#endif

// The oracle lives in a separate translation unit without fast-math.
auto exp_i_gate_subject(std::complex<real_type>* const state, real_type const phase,
                        int const kind, bool const adjoint, bool const fused) -> void
{
  using qubit = ket::qubit<std::uint64_t, unsigned int>;
  // Three fused wires with an unfused spectator at wire 1. Fused gate operands
  // are local indices; ordinary gate operands are physical wire indices.
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
  std::array<qubit, 3u> const wires{{qubit{0u}, qubit{2u}, qubit{3u}}};
  std::array<qubit, 4u> const indices{{qubit{0u}, qubit{2u}, qubit{3u}, qubit{4u}}};
#else
  std::array<std::uint64_t, 3u> const wires{{1u, 4u, 8u}};
  // Masks for the gaps before, between, and after sorted operated wires.
  std::array<std::uint64_t, 4u> const indices{{0u, 1u, 0u, ~std::uint64_t{1u}}};
#endif
  auto const apply = [&](auto const... operands) {
    if (fused)
      for (auto block = std::uint64_t{0}; block < 2u; ++block)
      {
        if (adjoint)
          ket::gate::fused::adj_phase_shift(state, block, wires, indices, phase, operands...);
        else
          ket::gate::fused::phase_shift(state, block, wires, indices, phase, operands...);
      }
    else if (adjoint)
      ket::gate::adj_phase_shift(state, state + 16, phase, operands...);
    else
      ket::gate::phase_shift(state, state + 16, phase, operands...);
  };
  auto const positive = ket::control<qubit, 1u>{qubit{fused ? 1u : 2u}};
  auto const negative = ket::control<qubit, 0u>{qubit{fused ? 2u : 3u}};
  switch (kind)
  {
    case 0: apply(); break; // global phase
    case 1: apply(qubit{0u}); break;
    case 2: apply(qubit{0u}, positive); break;
    case 3: apply(qubit{0u}, negative); break;
    case 4: apply(qubit{0u}, positive, negative); break;
  }
}
