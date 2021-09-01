#ifndef KET_GATE_PAULI_X_HPP
# define KET_GATE_PAULI_X_HPP

# include <cassert>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif


namespace ket
{
  namespace gate
  {
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

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void pauli_x(
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_x(::ket::utility::policy::make_sequential(), first, last, qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& pauli_x(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::pauli_x(parallel_policy, std::begin(state), std::end(state), qubit);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& pauli_x(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_x(::ket::utility::policy::make_sequential(), state, qubit); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_pauli_x(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_x(parallel_policy, first, last, qubit); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_pauli_x(
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_x(first, last, qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_pauli_x(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_x(parallel_policy, state, qubit); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_pauli_x(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_x(state, qubit); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_PAULI_X_HPP
