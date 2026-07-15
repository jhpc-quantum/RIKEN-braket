#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <ket/expectation_value.hpp>
#include <ket/fidelity.hpp>
#include <ket/gate/utility/index_with_qubits.hpp>
#include <ket/inner_product.hpp>
#include <ket/qubit.hpp>
#include <ket/utility/integer_exp2.hpp>
#include <ket/utility/loop_n.hpp>

namespace
{
  using complex_type = std::complex<double>;
  using real_type = double;
  using state_integer_type = std::uint64_t;
  using bit_integer_type = unsigned int;
  using qubit_type = ket::qubit<state_integer_type, bit_integer_type>;

  constexpr auto total_qubits = bit_integer_type{4u};
  constexpr auto state_size = std::size_t{1u} << total_qubits;

  auto state1() -> std::vector<complex_type>
  {
    auto result = std::vector<complex_type>(state_size);
    for (auto index = std::size_t{0u}; index < result.size(); ++index)
      result[index] = complex_type{
        0.0625 * static_cast<double>((index * 5u + 3u) % 17u),
        -0.03125 * static_cast<double>((index * 7u + 1u) % 13u)};
    return result;
  }

  auto state2() -> std::vector<complex_type>
  {
    auto result = std::vector<complex_type>(state_size);
    for (auto index = std::size_t{0u}; index < result.size(); ++index)
      result[index] = complex_type{
        -0.046875 * static_cast<double>((index * 11u + 2u) % 19u),
        0.078125 * static_cast<double>((index * 3u + 5u) % 11u)};
    return result;
  }

  struct identity_observable
  {
    template <typename Iterator, typename StateInteger, typename Qubits, typename SortedQubits>
    auto operator()(
      Iterator const first, StateInteger const index_wo_qubits,
      Qubits const& qubits, SortedQubits const& sorted_qubits_with_sentinel) const
    -> typename std::iterator_traits<Iterator>::value_type
    {
      using complex_type = typename std::iterator_traits<Iterator>::value_type;
      using std::begin;
      using std::end;
      using std::conj;

      auto const num_indices = ::ket::utility::integer_exp2<std::size_t>(std::distance(begin(qubits), end(qubits)));
      auto result = complex_type{};
      for (auto qubits_value = std::size_t{0u}; qubits_value < num_indices; ++qubits_value)
      {
        auto const index
          = ::ket::gate::utility::ranges::index_with_qubits(
              index_wo_qubits, qubits_value, qubits, sorted_qubits_with_sentinel);
        result += conj(*(first + index)) * *(first + index);
      }

      return result;
    }

    template <typename Iterator1, typename Iterator2, typename StateInteger, typename Qubits, typename SortedQubits>
    auto operator()(
      Iterator1 const first1, Iterator2 const first2, StateInteger const index_wo_qubits,
      Qubits const& qubits, SortedQubits const& sorted_qubits_with_sentinel) const
    -> typename std::iterator_traits<Iterator1>::value_type
    {
      using complex_type = typename std::iterator_traits<Iterator1>::value_type;
      using std::begin;
      using std::end;
      using std::conj;

      auto const num_indices = ::ket::utility::integer_exp2<std::size_t>(std::distance(begin(qubits), end(qubits)));
      auto result = complex_type{};
      for (auto qubits_value = std::size_t{0u}; qubits_value < num_indices; ++qubits_value)
      {
        auto const index
          = ::ket::gate::utility::ranges::index_with_qubits(
              index_wo_qubits, qubits_value, qubits, sorted_qubits_with_sentinel);
        result += conj(*(first2 + index)) * *(first1 + index);
      }

      return result;
    }
  };

  auto expected_expectation_value(std::vector<complex_type> const& state) -> complex_type
  {
    using std::conj;
    auto result = complex_type{};
    for (auto const& value: state)
      result += conj(value) * value;
    return result;
  }

  auto expected_inner_product(
    std::vector<complex_type> const& lhs, std::vector<complex_type> const& rhs)
    -> complex_type
  {
    using std::conj;
    auto result = complex_type{};
    for (auto index = std::size_t{0u}; index < lhs.size(); ++index)
      result += conj(rhs[index]) * lhs[index];
    return result;
  }

  auto close(complex_type const lhs, complex_type const rhs) -> bool
  { return std::abs(lhs - rhs) < real_type{1e-12}; }

  auto close(real_type const lhs, real_type const rhs) -> bool
  { return std::abs(lhs - rhs) < real_type{1e-12}; }

  template <typename Value>
  auto run_case(std::string const& name, Value const actual, Value const expected) -> bool
  {
    if (close(actual, expected))
      return true;

    std::cerr << name << " failed: actual = " << actual << ", expected = " << expected << '\n';
    return false;
  }
}

int main()
{
  using namespace ket::literals::qubit_literals;

  auto const psi1 = state1();
  auto const psi2 = state2();
  auto const observable = identity_observable{};
  auto const qubits = std::vector<qubit_type>{0_q, 2_q, 3_q};
  auto const expected_ev = expected_expectation_value(psi1);
  auto const expected_ip = expected_inner_product(psi1, psi2);
  using std::norm;
  auto const expected_fidelity = norm(expected_ip);
  auto const sequential = ket::utility::policy::make_sequential();

  auto failed = false;
  auto const run = [&failed](bool const passed) { failed = failed or not passed; };

  auto const nonruntime_ranges_ev
    = ket::ranges::expectation_value(sequential, psi1, observable, 0_q, 2_q, 3_q);
  auto const nonruntime_iterator_ev
    = ket::expectation_value(sequential, psi1.begin(), psi1.end(), observable, 0_q, 2_q, 3_q);
  auto const nonruntime_ranges_ip
    = ket::ranges::inner_product(sequential, psi1, psi2, observable, 0_q, 2_q, 3_q);
  auto const nonruntime_iterator_ip
    = ket::inner_product(sequential, psi1.begin(), psi1.end(), psi2.begin(), observable, 0_q, 2_q, 3_q);
  auto const nonruntime_ranges_fidelity
    = ket::ranges::fidelity(sequential, psi1, psi2, observable, 0_q, 2_q, 3_q);
  auto const nonruntime_iterator_fidelity
    = ket::fidelity(sequential, psi1.begin(), psi1.end(), psi2.begin(), observable, 0_q, 2_q, 3_q);

  run(run_case(
    "ranges::expectation_value",
    nonruntime_ranges_ev, expected_ev));

  run(run_case(
    "expectation_value",
    nonruntime_iterator_ev, expected_ev));

  run(run_case(
    "runtime::ranges::expectation_value vs non-runtime",
    ket::runtime::ranges::expectation_value(sequential, psi1, observable, qubits),
    nonruntime_ranges_ev));

  run(run_case(
    "runtime::expectation_value vs non-runtime",
    ket::runtime::expectation_value(sequential, psi1.begin(), psi1.end(), observable, qubits.begin(), qubits.end()),
    nonruntime_iterator_ev));

  run(run_case(
    "runtime::qubit_ranges::expectation_value vs non-runtime",
    ket::runtime::qubit_ranges::expectation_value(psi1.begin(), psi1.end(), observable, qubits),
    nonruntime_iterator_ev));

  run(run_case(
    "ranges::inner_product",
    nonruntime_ranges_ip, expected_ip));

  run(run_case(
    "inner_product",
    nonruntime_iterator_ip, expected_ip));

  run(run_case(
    "runtime::ranges::inner_product vs non-runtime",
    ket::runtime::ranges::inner_product(sequential, psi1, psi2, observable, qubits),
    nonruntime_ranges_ip));

  run(run_case(
    "runtime::inner_product vs non-runtime",
    ket::runtime::inner_product(sequential, psi1.begin(), psi1.end(), psi2.begin(), observable, qubits.begin(), qubits.end()),
    nonruntime_iterator_ip));

  run(run_case(
    "runtime::qubit_ranges::inner_product vs non-runtime",
    ket::runtime::qubit_ranges::inner_product(psi1.begin(), psi1.end(), psi2.begin(), observable, qubits),
    nonruntime_iterator_ip));

  run(run_case(
    "ranges::fidelity",
    nonruntime_ranges_fidelity, expected_fidelity));

  run(run_case(
    "fidelity",
    nonruntime_iterator_fidelity, expected_fidelity));

  run(run_case(
    "runtime::ranges::fidelity vs non-runtime",
    ket::runtime::ranges::fidelity(sequential, psi1, psi2, observable, qubits),
    nonruntime_ranges_fidelity));

  run(run_case(
    "runtime::fidelity vs non-runtime",
    ket::runtime::fidelity(sequential, psi1.begin(), psi1.end(), psi2.begin(), observable, qubits.begin(), qubits.end()),
    nonruntime_iterator_fidelity));

  run(run_case(
    "runtime::qubit_ranges::fidelity vs non-runtime",
    ket::runtime::qubit_ranges::fidelity(psi1.begin(), psi1.end(), psi2.begin(), observable, qubits),
    nonruntime_iterator_fidelity));

  if (failed)
    return EXIT_FAILURE;

  std::cout << "runtime non-MPI observable numerical tests passed\n";
  return EXIT_SUCCESS;
}
