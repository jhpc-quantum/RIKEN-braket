#include <mpi.h>

// Example:
//   mpicxx -std=c++14 -DNDEBUG -Iket/include -I../yampi/include \
//     ket/test/mpi/runtime_observable_numerical.cpp -o /tmp/runtime_observable_numerical
//   mpiexec -n 2 /tmp/runtime_observable_numerical

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <ket/mpi/expectation_value.hpp>
#include <ket/mpi/fidelity.hpp>
#include <ket/mpi/inner_product.hpp>
#include <ket/mpi/qubit_permutation.hpp>
#include <ket/mpi/utility/simple_mpi.hpp>
#include <ket/gate/utility/index_with_qubits.hpp>
#include <ket/qubit.hpp>
#include <ket/utility/integer_exp2.hpp>
#include <ket/utility/loop_n.hpp>
#include <yampi/communicator.hpp>
#include <yampi/environment.hpp>
#include <yampi/predefined_datatype.hpp>
#include <yampi/rank.hpp>

namespace
{
  using complex_type = std::complex<double>;
  using real_type = double;
  using state_integer_type = std::uint64_t;
  using bit_integer_type = unsigned int;
  using qubit_type = ket::qubit<state_integer_type, bit_integer_type>;
  using permutation_type = ket::mpi::qubit_permutation<state_integer_type, bit_integer_type>;

  using namespace ket::literals::qubit_literals;
  using namespace yampi::literals::rank_literals;

  constexpr auto total_qubits = bit_integer_type{4u};
  constexpr auto local_qubits = bit_integer_type{3u};
  constexpr auto total_state_size = std::size_t{1u} << total_qubits;
  constexpr auto local_state_size = std::size_t{1u} << local_qubits;

  auto state1() -> std::vector<complex_type>
  {
    auto result = std::vector<complex_type>(total_state_size);
    for (auto index = std::size_t{0u}; index < result.size(); ++index)
      result[index] = complex_type{
        0.0625 * static_cast<double>((index * 5u + 3u) % 17u),
        -0.03125 * static_cast<double>((index * 7u + 1u) % 13u)};
    return result;
  }

  auto state2() -> std::vector<complex_type>
  {
    auto result = std::vector<complex_type>(total_state_size);
    for (auto index = std::size_t{0u}; index < result.size(); ++index)
      result[index] = complex_type{
        -0.046875 * static_cast<double>((index * 11u + 2u) % 19u),
        0.078125 * static_cast<double>((index * 3u + 5u) % 11u)};
    return result;
  }

  auto local_slice(std::vector<complex_type> const& full_state, yampi::rank const rank)
    -> std::vector<complex_type>
  {
    auto const rank_index = static_cast<int>(rank);
    auto result = std::vector<complex_type>(local_state_size);
    std::copy(
      full_state.begin() + rank_index * local_state_size,
      full_state.begin() + (rank_index + 1) * local_state_size,
      result.begin());
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
  auto report_failure(std::string const& name, Value const actual, Value const expected, yampi::rank const rank) -> void
  {
    if (rank == 0_r)
      std::cerr << name << " failed: actual = " << actual << ", expected = " << expected << '\n';
  }

  template <typename Value>
  auto run_value_case(std::string const& name, Value const actual, Value const expected, yampi::rank const rank) -> bool
  {
    if (close(actual, expected))
      return true;

    report_failure(name, actual, expected, rank);
    return false;
  }

  template <typename Value>
  auto run_optional_case(
    std::string const& name, boost::optional<Value> const& actual, Value const expected,
    yampi::rank const rank, yampi::rank const root)
    -> bool
  {
    if (rank != root)
      return not actual;

    if (actual and close(*actual, expected))
      return true;

    if (actual)
      report_failure(name, *actual, expected, rank);
    else if (rank == root)
      std::cerr << name << " failed: root has no result\n";
    return false;
  }

  template <typename Value>
  auto run_optional_pair_case(
    std::string const& name, boost::optional<Value> const& actual, boost::optional<Value> const& expected,
    yampi::rank const rank)
    -> bool
  {
    if (not actual and not expected)
      return true;
    if (actual and expected and close(*actual, *expected))
      return true;

    if (rank == 0_r)
    {
      std::cerr << name << " failed:";
      if (actual)
        std::cerr << " actual = " << *actual;
      else
        std::cerr << " actual = none";
      if (expected)
        std::cerr << ", expected = " << *expected;
      else
        std::cerr << ", expected = none";
      std::cerr << '\n';
    }

    return false;
  }

  template <typename ActualOperation, typename ExpectedOperation>
  auto run_value_operation_pair(
    std::string const& name, yampi::rank const rank,
    ActualOperation const& actual_operation, ExpectedOperation const& expected_operation)
    -> bool
  {
    auto const actual = actual_operation();
    auto const expected = expected_operation();
    return run_value_case(name, actual, expected, rank);
  }

  template <typename ActualOperation, typename ExpectedOperation>
  auto run_optional_operation_pair(
    std::string const& name, yampi::rank const rank,
    ActualOperation const& actual_operation, ExpectedOperation const& expected_operation)
    -> bool
  {
    auto const actual = actual_operation();
    auto const expected = expected_operation();
    return run_optional_pair_case(name, actual, expected, rank);
  }

  template <typename Operation>
  auto with_local_state1(yampi::rank const rank, Operation const& operation)
    -> decltype(operation(
         std::declval<std::vector<complex_type>&>(),
         std::declval<permutation_type&>(),
         std::declval<std::vector<complex_type>&>()))
  {
    auto local_state = local_slice(state1(), rank);
    auto permutation = permutation_type{total_qubits};
    auto buffer = std::vector<complex_type>(local_state.size());
    return operation(local_state, permutation, buffer);
  }

  template <typename Operation>
  auto with_local_states(yampi::rank const rank, Operation const& operation)
    -> decltype(operation(
         std::declval<std::vector<complex_type>&>(),
         std::declval<permutation_type&>(),
         std::declval<std::vector<complex_type>&>(),
         std::declval<permutation_type&>(),
         std::declval<std::vector<complex_type>&>()))
  {
    auto local_state1 = local_slice(state1(), rank);
    auto local_state2 = local_slice(state2(), rank);
    auto permutation1 = permutation_type{total_qubits};
    auto permutation2 = permutation_type{total_qubits};
    auto buffer = std::vector<complex_type>(local_state1.size());
    return operation(local_state1, permutation1, local_state2, permutation2, buffer);
  }
}

int main(int argc, char** argv)
{
  yampi::environment environment{argc, argv};
  auto communicator = yampi::communicator{yampi::tags::world_communicator};

  auto const rank = communicator.rank(environment);
  auto const size = communicator.size(environment);
  if (size != 2)
  {
    if (rank == 0_r)
      std::cerr << "runtime_observable_numerical requires exactly 2 MPI processes\n";
    return EXIT_FAILURE;
  }

  auto const mpi_policy = ket::mpi::utility::policy::make_simple_mpi();
  auto const parallel_policy = ket::utility::policy::make_sequential();
  auto const datatype = yampi::predefined_datatype<complex_type>{};
  auto const root = 0_r;
  auto const observable = identity_observable{};
  auto const qubits = std::vector<qubit_type>{0_q, 2_q, 3_q};
  auto const local_qubits = std::vector<qubit_type>{0_q, 1_q, 2_q};
  auto const expected_ev = expected_expectation_value(state1());
  auto const expected_ip = expected_inner_product(state1(), state2());
  using std::norm;
  auto const expected_fidelity = norm(expected_ip);

  auto failed = false;
  auto const run = [&failed](bool const passed) { failed = failed or not passed; };

  auto const mpi_expectation_value
    = [&](auto& local_state, auto& permutation, auto& buffer)
      {
        return ket::mpi::expectation_value(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, observable, 0_q, 2_q, 3_q);
      };

  run(run_value_operation_pair(
    "mpi::runtime::ranges::expectation_value all_reduce",
    rank,
    [&]
    {
      return with_local_state1(
        rank,
        [&](auto& local_state, auto& permutation, auto& buffer)
        {
          return ket::mpi::runtime::ranges::expectation_value(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, observable, qubits);
        });
    },
    [&] { return with_local_state1(rank, mpi_expectation_value); }));

  run(run_value_operation_pair(
    "mpi::runtime::expectation_value all_reduce datatype",
    rank,
    [&]
    {
      return with_local_state1(
        rank,
        [&](auto& local_state, auto& permutation, auto& buffer)
        {
          return ket::mpi::runtime::expectation_value(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment,
            observable, local_qubits.begin(), local_qubits.end());
        });
    },
    [&]
    {
      return with_local_state1(
        rank,
        [&](auto& local_state, auto& permutation, auto& buffer)
        {
          return ket::mpi::expectation_value(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment,
            observable, 0_q, 1_q, 2_q);
        });
    }));

  run(run_optional_operation_pair(
    "mpi::runtime::ranges::expectation_value reduce",
    rank,
    [&]
    {
      return with_local_state1(
        rank,
        [&](auto& local_state, auto& permutation, auto& buffer)
        {
          return ket::mpi::runtime::ranges::expectation_value(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, root, communicator, environment, observable, qubits);
        });
    },
    [&]
    {
      return with_local_state1(
        rank,
        [&](auto& local_state, auto& permutation, auto& buffer)
        {
          return ket::mpi::expectation_value(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, root, communicator, environment, observable, 0_q, 2_q, 3_q);
        });
    }));

  run(run_optional_operation_pair(
    "mpi::runtime::expectation_value reduce datatype",
    rank,
    [&]
    {
      return with_local_state1(
        rank,
        [&](auto& local_state, auto& permutation, auto& buffer)
        {
          return ket::mpi::runtime::expectation_value(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, root, communicator, environment,
            observable, local_qubits.begin(), local_qubits.end());
        });
    },
    [&]
    {
      return with_local_state1(
        rank,
        [&](auto& local_state, auto& permutation, auto& buffer)
        {
          return ket::mpi::expectation_value(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, root, communicator, environment,
            observable, 0_q, 1_q, 2_q);
        });
    }));

  run(run_value_case(
    "mpi::expectation_value all_reduce",
    with_local_state1(rank, mpi_expectation_value),
    expected_ev, rank));

  auto const mpi_inner_product
    = [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
      {
        return ket::mpi::inner_product(
          mpi_policy, parallel_policy,
          local_state1, permutation1, local_state2, permutation2,
          buffer, communicator, environment, observable, 0_q, 2_q, 3_q);
      };

  run(run_value_operation_pair(
    "mpi::runtime::ranges::inner_product all_reduce",
    rank,
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::runtime::ranges::inner_product(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, communicator, environment, observable, qubits);
        });
    },
    [&] { return with_local_states(rank, mpi_inner_product); }));

  run(run_value_operation_pair(
    "mpi::runtime::inner_product all_reduce datatype",
    rank,
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::runtime::inner_product(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, datatype, communicator, environment, observable, local_qubits.begin(), local_qubits.end());
        });
    },
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::inner_product(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, datatype, communicator, environment, observable, 0_q, 1_q, 2_q);
        });
    }));

  run(run_optional_operation_pair(
    "mpi::runtime::ranges::inner_product reduce",
    rank,
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::runtime::ranges::inner_product(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, root, communicator, environment, observable, qubits);
        });
    },
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::inner_product(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, root, communicator, environment, observable, 0_q, 2_q, 3_q);
        });
    }));

  run(run_optional_operation_pair(
    "mpi::runtime::inner_product reduce datatype",
    rank,
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::runtime::inner_product(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, datatype, root, communicator, environment, observable, local_qubits.begin(), local_qubits.end());
        });
    },
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::inner_product(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, datatype, root, communicator, environment, observable, 0_q, 1_q, 2_q);
        });
    }));

  run(run_value_case(
    "mpi::inner_product all_reduce",
    with_local_states(rank, mpi_inner_product),
    expected_ip, rank));

  auto const mpi_fidelity
    = [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
      {
        return ket::mpi::fidelity(
          mpi_policy, parallel_policy,
          local_state1, permutation1, local_state2, permutation2,
          buffer, communicator, environment, observable, 0_q, 2_q, 3_q);
      };

  run(run_value_operation_pair(
    "mpi::runtime::ranges::fidelity all_reduce",
    rank,
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::runtime::ranges::fidelity(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, communicator, environment, observable, qubits);
        });
    },
    [&] { return with_local_states(rank, mpi_fidelity); }));

  run(run_value_operation_pair(
    "mpi::runtime::fidelity all_reduce datatype",
    rank,
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::runtime::fidelity(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, datatype, communicator, environment, observable, local_qubits.begin(), local_qubits.end());
        });
    },
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::fidelity(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, datatype, communicator, environment, observable, 0_q, 1_q, 2_q);
        });
    }));

  run(run_optional_operation_pair(
    "mpi::runtime::ranges::fidelity reduce",
    rank,
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::runtime::ranges::fidelity(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, root, communicator, environment, observable, qubits);
        });
    },
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::fidelity(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, root, communicator, environment, observable, 0_q, 2_q, 3_q);
        });
    }));

  run(run_optional_operation_pair(
    "mpi::runtime::fidelity reduce datatype",
    rank,
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::runtime::fidelity(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, datatype, root, communicator, environment, observable, local_qubits.begin(), local_qubits.end());
        });
    },
    [&]
    {
      return with_local_states(
        rank,
        [&](auto& local_state1, auto& permutation1, auto& local_state2, auto& permutation2, auto& buffer)
        {
          return ket::mpi::fidelity(
            mpi_policy, parallel_policy,
            local_state1, permutation1, local_state2, permutation2,
            buffer, datatype, root, communicator, environment, observable, 0_q, 1_q, 2_q);
        });
    }));

  run(run_value_case(
    "mpi::fidelity all_reduce",
    with_local_states(rank, mpi_fidelity),
    expected_fidelity, rank));

  if (rank == 0_r and not failed)
    std::cout << "runtime MPI observable numerical tests passed\n";

  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
