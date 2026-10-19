#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
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

namespace
{
  using qubit = ket::qubit<>;
  using complex_type = std::complex<double>;
  using state_type = std::vector<complex_type>;
  auto flip(state_type& state) -> void
  {
    for (auto i = std::size_t{0u}; i < state.size(); ++i)
      if ((i bitand 16u) == 0u) std::swap(state[i], state[i xor 16u]);
  }

  template <typename Positive, typename NativeRange, typename RuntimeRange>
  auto check(char const* name, Positive const& positive, NativeRange const& native, RuntimeRange const& runtime) -> void
  {
    auto initial = state_type(128u);
    for (auto i = std::size_t{0u}; i < initial.size(); ++i)
      initial[i] = complex_type{double(i + 1u) * 0.01, double((i * 3u) % 17u) * -0.02};
    auto expected = initial;
    flip(expected); positive(expected); flip(expected);
    auto actual = initial;
    native(actual);
    auto dynamic = initial;
    runtime(dynamic);
    for (auto i = std::size_t{0u}; i < initial.size(); ++i)
      if (std::abs(actual[i] - expected[i]) > 1e-11 or std::abs(dynamic[i] - expected[i]) > 1e-11)
      { std::cerr << name << ": state-range wrapper mismatch\n"; std::exit(1); }
  }
}

int main()
{
  auto const controls = std::vector<ket::runtime_control<qubit>>{
    ket::runtime_control<qubit>{qubit{4u}, 0}, ket::runtime_control<qubit>{qubit{1u}, 1}};
  check("hadamard/1",
    [&](state_type& state) { ket::gate::hadamard(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::hadamard(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::hadamard(state, qubit{3u}, controls); });
  check("adj_hadamard/1",
    [&](state_type& state) { ket::gate::adj_hadamard(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_hadamard(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_hadamard(state, qubit{3u}, controls); });
  check("not_/1",
    [&](state_type& state) { ket::gate::not_(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::not_(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::not_(state, qubit{3u}, controls); });
  check("adj_not_/1",
    [&](state_type& state) { ket::gate::adj_not_(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_not_(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_not_(state, qubit{3u}, controls); });
  check("controlled_not/1",
    [&](state_type& state) { ket::gate::controlled_not(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::controlled_not(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::controlled_not(state, qubit{3u}, controls); });
  check("adj_controlled_not/1",
    [&](state_type& state) { ket::gate::adj_controlled_not(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_controlled_not(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_controlled_not(state, qubit{3u}, controls); });
  check("pauli_x/1",
    [&](state_type& state) { ket::gate::pauli_x(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::pauli_x(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::pauli_x(state, std::vector<qubit>{qubit{3u}}, controls); });
  check("adj_pauli_x/1",
    [&](state_type& state) { ket::gate::adj_pauli_x(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_pauli_x(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_pauli_x(state, std::vector<qubit>{qubit{3u}}, controls); });
  check("pauli_y/1",
    [&](state_type& state) { ket::gate::pauli_y(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::pauli_y(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::pauli_y(state, std::vector<qubit>{qubit{3u}}, controls); });
  check("adj_pauli_y/1",
    [&](state_type& state) { ket::gate::adj_pauli_y(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_pauli_y(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_pauli_y(state, std::vector<qubit>{qubit{3u}}, controls); });
  check("pauli_z/1",
    [&](state_type& state) { ket::gate::pauli_z(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::pauli_z(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::pauli_z(state, std::vector<qubit>{qubit{3u}}, controls); });
  check("adj_pauli_z/1",
    [&](state_type& state) { ket::gate::adj_pauli_z(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_pauli_z(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_pauli_z(state, std::vector<qubit>{qubit{3u}}, controls); });
  check("sqrt_pauli_x/1",
    [&](state_type& state) { ket::gate::sqrt_pauli_x(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::sqrt_pauli_x(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::sqrt_pauli_x(state, qubit{3u}, controls); });
  check("adj_sqrt_pauli_x/1",
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_x(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_sqrt_pauli_x(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_sqrt_pauli_x(state, qubit{3u}, controls); });
  check("sqrt_pauli_y/1",
    [&](state_type& state) { ket::gate::sqrt_pauli_y(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::sqrt_pauli_y(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::sqrt_pauli_y(state, qubit{3u}, controls); });
  check("adj_sqrt_pauli_y/1",
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_y(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_sqrt_pauli_y(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_sqrt_pauli_y(state, qubit{3u}, controls); });
  check("sqrt_pauli_z/1",
    [&](state_type& state) { ket::gate::sqrt_pauli_z(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::sqrt_pauli_z(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::sqrt_pauli_z(state, std::vector<qubit>{qubit{3u}}, controls); });
  check("adj_sqrt_pauli_z/1",
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_z(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_sqrt_pauli_z(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_sqrt_pauli_z(state, std::vector<qubit>{qubit{3u}}, controls); });
  check("x_rotation_half_pi/1",
    [&](state_type& state) { ket::gate::x_rotation_half_pi(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::x_rotation_half_pi(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::x_rotation_half_pi(state, qubit{3u}, controls); });
  check("adj_x_rotation_half_pi/1",
    [&](state_type& state) { ket::gate::adj_x_rotation_half_pi(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_x_rotation_half_pi(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_x_rotation_half_pi(state, qubit{3u}, controls); });
  check("y_rotation_half_pi/1",
    [&](state_type& state) { ket::gate::y_rotation_half_pi(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::y_rotation_half_pi(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::y_rotation_half_pi(state, qubit{3u}, controls); });
  check("adj_y_rotation_half_pi/1",
    [&](state_type& state) { ket::gate::adj_y_rotation_half_pi(state.begin(), state.end(), qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_y_rotation_half_pi(state, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_y_rotation_half_pi(state, qubit{3u}, controls); });
  check("phase_shift/1",
    [&](state_type& state) { ket::gate::phase_shift(state.begin(), state.end(), 0.37, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::phase_shift(state, 0.37, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::phase_shift(state, 0.37, qubit{3u}, controls); });
  check("adj_phase_shift/1",
    [&](state_type& state) { ket::gate::adj_phase_shift(state.begin(), state.end(), 0.37, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_phase_shift(state, 0.37, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_phase_shift(state, 0.37, qubit{3u}, controls); });
  check("phase_shift_coeff/1",
    [&](state_type& state) { ket::gate::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::phase_shift_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::phase_shift_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, controls); });
  check("adj_phase_shift_coeff/1",
    [&](state_type& state) { ket::gate::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_phase_shift_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_phase_shift_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, controls); });
  check("phase_shift2/1",
    [&](state_type& state) { ket::gate::phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::phase_shift2(state, 0.21, -0.34, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::phase_shift2(state, 0.21, -0.34, qubit{3u}, controls); });
  check("adj_phase_shift2/1",
    [&](state_type& state) { ket::gate::adj_phase_shift2(state.begin(), state.end(), 0.21, -0.34, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_phase_shift2(state, 0.21, -0.34, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_phase_shift2(state, 0.21, -0.34, qubit{3u}, controls); });
  check("phase_shift3/1",
    [&](state_type& state) { ket::gate::phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::phase_shift3(state, 0.41, 0.23, -0.57, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::phase_shift3(state, 0.41, 0.23, -0.57, qubit{3u}, controls); });
  check("adj_phase_shift3/1",
    [&](state_type& state) { ket::gate::adj_phase_shift3(state.begin(), state.end(), 0.41, 0.23, -0.57, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_phase_shift3(state, 0.41, 0.23, -0.57, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_phase_shift3(state, 0.41, 0.23, -0.57, qubit{3u}, controls); });
  check("exponential_pauli_x/1",
    [&](state_type& state) { ket::gate::exponential_pauli_x(state.begin(), state.end(), 0.37, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::exponential_pauli_x(state, 0.37, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::exponential_pauli_x(state, 0.37, std::vector<qubit>{qubit{3u}}, controls); });
  check("adj_exponential_pauli_x/1",
    [&](state_type& state) { ket::gate::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_exponential_pauli_x(state, 0.37, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_exponential_pauli_x(state, 0.37, std::vector<qubit>{qubit{3u}}, controls); });
  check("exponential_pauli_x_coeff/1",
    [&](state_type& state) { ket::gate::exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::exponential_pauli_x_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::exponential_pauli_x_coeff(state, complex_type{0.8, 0.6}, std::vector<qubit>{qubit{3u}}, controls); });
  check("adj_exponential_pauli_x_coeff/1",
    [&](state_type& state) { ket::gate::adj_exponential_pauli_x_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_exponential_pauli_x_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_exponential_pauli_x_coeff(state, complex_type{0.8, 0.6}, std::vector<qubit>{qubit{3u}}, controls); });
  check("exponential_pauli_y/1",
    [&](state_type& state) { ket::gate::exponential_pauli_y(state.begin(), state.end(), 0.37, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::exponential_pauli_y(state, 0.37, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::exponential_pauli_y(state, 0.37, std::vector<qubit>{qubit{3u}}, controls); });
  check("adj_exponential_pauli_y/1",
    [&](state_type& state) { ket::gate::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_exponential_pauli_y(state, 0.37, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_exponential_pauli_y(state, 0.37, std::vector<qubit>{qubit{3u}}, controls); });
  check("exponential_pauli_y_coeff/1",
    [&](state_type& state) { ket::gate::exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::exponential_pauli_y_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::exponential_pauli_y_coeff(state, complex_type{0.8, 0.6}, std::vector<qubit>{qubit{3u}}, controls); });
  check("adj_exponential_pauli_y_coeff/1",
    [&](state_type& state) { ket::gate::adj_exponential_pauli_y_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_exponential_pauli_y_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_exponential_pauli_y_coeff(state, complex_type{0.8, 0.6}, std::vector<qubit>{qubit{3u}}, controls); });
  check("exponential_pauli_z/1",
    [&](state_type& state) { ket::gate::exponential_pauli_z(state.begin(), state.end(), 0.37, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::exponential_pauli_z(state, 0.37, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::exponential_pauli_z(state, 0.37, std::vector<qubit>{qubit{3u}}, controls); });
  check("adj_exponential_pauli_z/1",
    [&](state_type& state) { ket::gate::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_exponential_pauli_z(state, 0.37, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_exponential_pauli_z(state, 0.37, std::vector<qubit>{qubit{3u}}, controls); });
  check("exponential_pauli_z_coeff/1",
    [&](state_type& state) { ket::gate::exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::exponential_pauli_z_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::exponential_pauli_z_coeff(state, complex_type{0.8, 0.6}, std::vector<qubit>{qubit{3u}}, controls); });
  check("adj_exponential_pauli_z_coeff/1",
    [&](state_type& state) { ket::gate::adj_exponential_pauli_z_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_exponential_pauli_z_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_exponential_pauli_z_coeff(state, complex_type{0.8, 0.6}, std::vector<qubit>{qubit{3u}}, controls); });
  check("controlled_v/1",
    [&](state_type& state) { ket::gate::controlled_v(state.begin(), state.end(), 0.37, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::controlled_v(state, 0.37, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::controlled_v(state, 0.37, qubit{3u}, controls); });
  check("adj_controlled_v/1",
    [&](state_type& state) { ket::gate::adj_controlled_v(state.begin(), state.end(), 0.37, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_controlled_v(state, 0.37, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_controlled_v(state, 0.37, qubit{3u}, controls); });
  check("controlled_v_coeff/1",
    [&](state_type& state) { ket::gate::controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::controlled_v_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::controlled_v_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, controls); });
  check("adj_controlled_v_coeff/1",
    [&](state_type& state) { ket::gate::adj_controlled_v_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_controlled_v_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_controlled_v_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, controls); });
  check("pauli_x/3",
    [&](state_type& state) { ket::gate::pauli_x(state.begin(), state.end(), qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::pauli_x(state, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::pauli_x(state, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("adj_pauli_x/3",
    [&](state_type& state) { ket::gate::adj_pauli_x(state.begin(), state.end(), qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_pauli_x(state, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_pauli_x(state, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("pauli_y/3",
    [&](state_type& state) { ket::gate::pauli_y(state.begin(), state.end(), qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::pauli_y(state, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::pauli_y(state, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("adj_pauli_y/3",
    [&](state_type& state) { ket::gate::adj_pauli_y(state.begin(), state.end(), qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_pauli_y(state, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_pauli_y(state, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("pauli_z/3",
    [&](state_type& state) { ket::gate::pauli_z(state.begin(), state.end(), qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::pauli_z(state, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::pauli_z(state, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("adj_pauli_z/3",
    [&](state_type& state) { ket::gate::adj_pauli_z(state.begin(), state.end(), qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_pauli_z(state, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_pauli_z(state, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("exponential_pauli_x/3",
    [&](state_type& state) { ket::gate::exponential_pauli_x(state.begin(), state.end(), 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::exponential_pauli_x(state, 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::exponential_pauli_x(state, 0.37, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("adj_exponential_pauli_x/3",
    [&](state_type& state) { ket::gate::adj_exponential_pauli_x(state.begin(), state.end(), 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_exponential_pauli_x(state, 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_exponential_pauli_x(state, 0.37, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("exponential_pauli_y/3",
    [&](state_type& state) { ket::gate::exponential_pauli_y(state.begin(), state.end(), 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::exponential_pauli_y(state, 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::exponential_pauli_y(state, 0.37, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("adj_exponential_pauli_y/3",
    [&](state_type& state) { ket::gate::adj_exponential_pauli_y(state.begin(), state.end(), 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_exponential_pauli_y(state, 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_exponential_pauli_y(state, 0.37, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("exponential_pauli_z/3",
    [&](state_type& state) { ket::gate::exponential_pauli_z(state.begin(), state.end(), 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::exponential_pauli_z(state, 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::exponential_pauli_z(state, 0.37, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("adj_exponential_pauli_z/3",
    [&](state_type& state) { ket::gate::adj_exponential_pauli_z(state.begin(), state.end(), 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_exponential_pauli_z(state, 0.37, qubit{3u}, qubit{0u}, qubit{5u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_exponential_pauli_z(state, 0.37, std::vector<qubit>{qubit{3u}, qubit{0u}, qubit{5u}}, controls); });
  check("swap/2",
    [&](state_type& state) { ket::gate::swap(state.begin(), state.end(), qubit{3u}, qubit{0u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::swap(state, qubit{3u}, qubit{0u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::swap(state, qubit{3u}, qubit{0u}, controls); });
  check("adj_swap/2",
    [&](state_type& state) { ket::gate::adj_swap(state.begin(), state.end(), qubit{3u}, qubit{0u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_swap(state, qubit{3u}, qubit{0u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_swap(state, qubit{3u}, qubit{0u}, controls); });
  check("exponential_swap/2",
    [&](state_type& state) { ket::gate::exponential_swap(state.begin(), state.end(), 0.37, qubit{3u}, qubit{0u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::exponential_swap(state, 0.37, qubit{3u}, qubit{0u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::exponential_swap(state, 0.37, qubit{3u}, qubit{0u}, controls); });
  check("adj_exponential_swap/2",
    [&](state_type& state) { ket::gate::adj_exponential_swap(state.begin(), state.end(), 0.37, qubit{3u}, qubit{0u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_exponential_swap(state, 0.37, qubit{3u}, qubit{0u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_exponential_swap(state, 0.37, qubit{3u}, qubit{0u}, controls); });
  check("exponential_swap_coeff/2",
    [&](state_type& state) { ket::gate::exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, qubit{0u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::exponential_swap_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, qubit{0u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::exponential_swap_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, qubit{0u}, controls); });
  check("adj_exponential_swap_coeff/2",
    [&](state_type& state) { ket::gate::adj_exponential_swap_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, qubit{3u}, qubit{0u}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_exponential_swap_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, qubit{0u}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_exponential_swap_coeff(state, complex_type{0.8, 0.6}, qubit{3u}, qubit{0u}, controls); });
  check("phase_shift/0",
    [&](state_type& state) { ket::gate::phase_shift(state.begin(), state.end(), 0.37, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::phase_shift(state, 0.37, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::phase_shift(state, 0.37, controls); });
  check("adj_phase_shift/0",
    [&](state_type& state) { ket::gate::adj_phase_shift(state.begin(), state.end(), 0.37, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_phase_shift(state, 0.37, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_phase_shift(state, 0.37, controls); });
  check("phase_shift_coeff/0",
    [&](state_type& state) { ket::gate::phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::phase_shift_coeff(state, complex_type{0.8, 0.6}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::phase_shift_coeff(state, complex_type{0.8, 0.6}, controls); });
  check("adj_phase_shift_coeff/0",
    [&](state_type& state) { ket::gate::adj_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_phase_shift_coeff(state, complex_type{0.8, 0.6}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_phase_shift_coeff(state, complex_type{0.8, 0.6}, controls); });
  check("controlled_phase_shift/0",
    [&](state_type& state) { ket::gate::controlled_phase_shift(state.begin(), state.end(), 0.37, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::controlled_phase_shift(state, 0.37, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::controlled_phase_shift(state, 0.37, controls); });
  check("adj_controlled_phase_shift/0",
    [&](state_type& state) { ket::gate::adj_controlled_phase_shift(state.begin(), state.end(), 0.37, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_controlled_phase_shift(state, 0.37, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_controlled_phase_shift(state, 0.37, controls); });
  check("controlled_phase_shift_coeff/0",
    [&](state_type& state) { ket::gate::controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::controlled_phase_shift_coeff(state, complex_type{0.8, 0.6}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::controlled_phase_shift_coeff(state, complex_type{0.8, 0.6}, controls); });
  check("adj_controlled_phase_shift_coeff/0",
    [&](state_type& state) { ket::gate::adj_controlled_phase_shift_coeff(state.begin(), state.end(), complex_type{0.8, 0.6}, ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_controlled_phase_shift_coeff(state, complex_type{0.8, 0.6}, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_controlled_phase_shift_coeff(state, complex_type{0.8, 0.6}, controls); });
  check("pauli_z/0",
    [&](state_type& state) { ket::gate::pauli_z(state.begin(), state.end(), ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::pauli_z(state, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::pauli_z(state, controls); });
  check("adj_pauli_z/0",
    [&](state_type& state) { ket::gate::adj_pauli_z(state.begin(), state.end(), ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_pauli_z(state, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_pauli_z(state, controls); });
  check("sqrt_pauli_z/0",
    [&](state_type& state) { ket::gate::sqrt_pauli_z(state.begin(), state.end(), ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::sqrt_pauli_z(state, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::sqrt_pauli_z(state, controls); });
  check("adj_sqrt_pauli_z/0",
    [&](state_type& state) { ket::gate::adj_sqrt_pauli_z(state.begin(), state.end(), ket::control<qubit>{4u}, ket::control<qubit>{1u}); },
    [&](state_type& state) { ket::gate::ranges::adj_sqrt_pauli_z(state, ket::control<qubit, 0>{4u}, ket::control<qubit, 1>{1u}); },
    [&](state_type& state) { ket::gate::runtime::ranges::adj_sqrt_pauli_z(state, controls); });
  auto state = state_type(8u, complex_type{0.1, 0.2});
  auto expected = state;
  auto const empty = std::vector<ket::runtime_control<qubit>>{};
  ket::gate::hadamard(expected.begin(), expected.end(), qubit{1u});
  ket::gate::runtime::ranges::hadamard(state, qubit{1u}, empty);
  for (auto i = std::size_t{0u}; i < state.size(); ++i) if (std::abs(state[i] - expected[i]) > 1e-11) return 1;
  ket::gate::global_phase(expected.begin(), expected.end(), 0.37);
  ket::gate::runtime::ranges::phase_shift(state, 0.37, empty);
  for (auto i = std::size_t{0u}; i < state.size(); ++i) if (std::abs(state[i] - expected[i]) > 1e-11) return 1;
  std::cout << "Control state-range wrappers passed\n";
}
