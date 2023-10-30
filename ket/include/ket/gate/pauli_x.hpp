#ifndef KET_GATE_PAULI_X_HPP
# define KET_GATE_PAULI_X_HPP

# include <cassert>
# include <array>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif


namespace ket
{
  namespace gate
  {
    // X_i (NOT_i)
    // X_1 (a_0 |0> + a_1 |1>) = a_1 |0> + a_0 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void pauli_x(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
      auto const lower_bits_mask = qubit_mask - StateInteger{1u};
      auto const upper_bits_mask = compl lower_bits_mask;

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 1u,
        [first, qubit_mask, lower_bits_mask, upper_bits_mask](StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index
            = ((value_wo_qubit bitand upper_bits_mask) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;

          std::iter_swap(first + zero_index, first + one_index);
        });
    }

    // XX_{ij} = X_i X_j
    // XX_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{11} |00> + a_{10} |01> + a_{01} |10> + a_{00} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void pauli_x(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      assert(::ket::utility::integer_exp2<StateInteger>(qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(qubit2) < static_cast<StateInteger>(last - first));
      assert(qubit1 != qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const minmax_qubits = std::minmax(qubit1, qubit2);
      auto const qubit1_mask = ::ket::utility::integer_exp2<StateInteger>(qubit1);
      auto const qubit2_mask = ::ket::utility::integer_exp2<StateInteger>(qubit2);
      auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
      auto const middle_bits_mask
        = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
          xor lower_bits_mask;
      auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 2u,
        [first, qubit1_mask, qubit2_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx0_1xxx0_2xxx
          auto const base_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          auto const off_iter = first + base_index;
          // xxx1_1xxx0_2xxx
          auto const qubit1_on_index = base_index bitor qubit1_mask;
          auto const qubit1_on_iter = first + qubit1_on_index;
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_iter = first + (base_index bitor qubit2_mask);
          // xxx1_1xxx1_2xxx
          auto const qubit12_on_iter = first + (qubit1_on_index bitor qubit2_mask);

          std::iter_swap(off_iter, qubit12_on_iter);
          std::iter_swap(qubit1_on_iter, qubit2_on_iter);
        });
    }

    // X...X_{i...j} = X_i ... X_j
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void pauli_x(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
      ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits)
    {
      constexpr auto num_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 3u);
      constexpr auto num_indices = ::ket::utility::integer_exp2<std::size_t>(num_qubits);

      ::ket::gate::gate(
        parallel_policy, first, last,
        [](RandomAccessIterator const first, std::array<StateInteger, num_indices> const& indices, int const)
        {
          auto const num_indices = static_cast<StateInteger>(boost::size(indices));
          auto const half_num_indices = num_indices / StateInteger{2u};
          for (auto i = StateInteger{0u}; i < half_num_indices; ++i)
          {
            auto iter1 = first + indices[i];
            auto iter2 = first + indices[num_indices - StateInteger{1u} - i];
            std::iter_swap(iter1, iter2);
          }
        },
        qubit1, qubit2, qubit3, qubits...);
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void pauli_x(
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    { ::ket::gate::pauli_x(::ket::utility::policy::make_sequential(), first, last, qubit, qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& pauli_x(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        ::ket::gate::pauli_x(parallel_policy, std::begin(state), std::end(state), qubit, qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& pauli_x(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      { return ::ket::gate::ranges::pauli_x(::ket::utility::policy::make_sequential(), state, qubit, qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void adj_pauli_x(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    { ::ket::gate::pauli_x(parallel_policy, first, last, qubit, qubits...); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void adj_pauli_x(
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    { ::ket::gate::pauli_x(first, last, qubit, qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& adj_pauli_x(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      { return ::ket::gate::ranges::pauli_x(parallel_policy, state, qubit, qubits...); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& adj_pauli_x(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      { return ::ket::gate::ranges::pauli_x(state, qubit, qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_PAULI_X_HPP
