// Example:
//   g++ -std=c++14 -Iket/include ket/test/reset.cpp -o /tmp/reset_test
//   /tmp/reset_test

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <ket/gate/reset.hpp>
#include <ket/qubit.hpp>


namespace
{
  using complex_type = std::complex<double>;
  using qubit_type = ket::qubit<std::uint64_t, unsigned int>;

  auto is_close(complex_type const lhs, complex_type const rhs) -> bool
  { return std::abs(lhs - rhs) < 1e-12; }

  auto is_zero_reset(std::vector<complex_type> const& state) -> bool
  {
    auto norm = 0.0;
    for (auto const amplitude : state)
      norm += std::norm(amplitude);
    for (auto index = std::size_t{1u}; index < state.size(); index += 2u)
      if (std::abs(state[index]) >= 1e-12)
        return false;
    return std::abs(norm - 1.0) < 1e-12;
  }
} // namespace


int main()
{
  auto random_number_generator = std::mt19937{42u};
  auto failed = false;

  {
    auto state = std::vector<complex_type>{{1.0, 0.0}, {0.0, 0.0}};
    ket::gate::ranges::reset(
      state, random_number_generator, qubit_type{0u});
    failed = failed or not is_close(state[0u], {1.0, 0.0})
      or not is_close(state[1u], {0.0, 0.0});
  }

  {
    auto state = std::vector<complex_type>{{0.0, 0.0}, {1.0, 0.0}};
    ket::gate::ranges::reset(
      state, random_number_generator, qubit_type{0u});
    failed = failed or not is_close(state[0u], {1.0, 0.0})
      or not is_close(state[1u], {0.0, 0.0});
  }

  {
    auto const inverse_sqrt_two = 1.0 / std::sqrt(2.0);
    auto state = std::vector<complex_type>{
      {inverse_sqrt_two, 0.0}, {inverse_sqrt_two, 0.0}};
    ket::gate::ranges::reset(
      state, random_number_generator, qubit_type{0u});
    failed = failed or not is_close(state[0u], {1.0, 0.0})
      or not is_close(state[1u], {0.0, 0.0});
  }

  {
    auto const inverse_sqrt_two = 1.0 / std::sqrt(2.0);
    auto state = std::vector<complex_type>{
      {inverse_sqrt_two, 0.0}, {0.0, 0.0},
      {0.0, 0.0}, {inverse_sqrt_two, 0.0}};
    ket::gate::ranges::reset(
      state, random_number_generator, qubit_type{0u});
    failed = failed or not is_zero_reset(state);
  }

  if (failed)
  {
    std::cerr << "reset test failed\n";
    return 1;
  }
}
