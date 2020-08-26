#ifndef KET_GATE_CONTROLLED_NOT_HPP
# define KET_GATE_CONTROLLED_NOT_HPP

# include <cassert>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>


namespace ket
{
  namespace gate
  {
    namespace controlled_not_detail
    {
      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger>
      inline void controlled_not_impl(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        static_assert(
          std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(
          ::ket::utility::integer_exp2<StateInteger>(target_qubit)
            < static_cast<StateInteger>(last - first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(control_qubit.qubit())
            < static_cast<StateInteger>(last - first));
        assert(target_qubit != control_qubit.qubit());
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last - first))
          == static_cast<StateInteger>(last - first));

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;

        auto const minmax_qubits = std::minmax(target_qubit, control_qubit.qubit());
        auto const target_qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
        auto const control_qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(control_qubit.qubit());
        auto const lower_bits_mask
          = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last - first)/4u,
          [first, target_qubit_mask, control_qubit_mask,
           lower_bits_mask, middle_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubits, int const)
          {
            // xxx0_txxx0_cxxx
            auto const base_index
              = ((value_wo_qubits bitand upper_bits_mask) << 2u)
                bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
                bitor (value_wo_qubits bitand lower_bits_mask);
            // xxx0_txxx1_cxxx
            auto const control_on_index = base_index bitor control_qubit_mask;
            // xxx1_txxx1_cxxx
            auto const target_control_on_index
              = control_on_index bitor target_qubit_mask;

            std::iter_swap(first + control_on_index, first + target_control_on_index);
          });
      }
    } // namespace controlled_not_detail

    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void controlled_not(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    {
      ::ket::gate::controlled_not_detail::controlled_not_impl(
        ::ket::utility::policy::make_sequential(),
        first, last, target_qubit, control_qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void controlled_not(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    {
      ::ket::gate::controlled_not_detail::controlled_not_impl(
        parallel_policy, first, last, target_qubit, control_qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_not(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        ::ket::gate::controlled_not_detail::controlled_not_impl(
          ::ket::utility::policy::make_sequential(),
          ::ket::utility::begin(state), ::ket::utility::end(state), target_qubit, control_qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_not(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        ::ket::gate::controlled_not_detail::controlled_not_impl(
          parallel_policy,
          ::ket::utility::begin(state), ::ket::utility::end(state), target_qubit, control_qubit);
        return state;
      }
    } // namespace ranges


    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void adj_controlled_not(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    { ::ket::gate::controlled_not(first, last, target_qubit, control_qubit); }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void adj_controlled_not(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    {
      ::ket::gate::controlled_not(
        parallel_policy, first, last, target_qubit, control_qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_not(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      { return ::ket::gate::ranges::controlled_not(state, target_qubit, control_qubit); }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_not(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        return ::ket::gate::ranges::controlled_not(
          parallel_policy, state, target_qubit, control_qubit);
      }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_CONTROLLED_NOT_HPP
