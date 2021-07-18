#ifndef KET_GATE_CONTROLLED_NOT_HPP
# define KET_GATE_CONTROLLED_NOT_HPP

# include <boost/config.hpp>

# include <array>
# include <iterator>
# include <algorithm>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/loop_n.hpp>


namespace ket
{
  namespace gate
  {
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void controlled_not(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      ::ket::gate::gate(
        parallel_policy, first, last,
        [](RandomAccessIterator const first, std::array<StateInteger, 4u> const& indices, int const)
        {
# ifndef BOOST_NO_CXX14_BINARY_LITERALS
          std::iter_swap(first + indices[0b01u], first + indices[0b11u]);
# else // BOOST_NO_CXX14_BINARY_LITERALS
          std::iter_swap(first + indices[1u], first + indices[3u]);
# endif // BOOST_NO_CXX14_BINARY_LITERALS
        },
        target_qubit, control_qubit);
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void controlled_not(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    { ::ket::gate::controlled_not(::ket::utility::policy::make_sequential(), first, last, target_qubit, control_qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_not(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      {
        ::ket::gate::controlled_not(parallel_policy, std::begin(state), std::end(state), target_qubit, control_qubit);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_not(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      { return ::ket::gate::ranges::controlled_not(::ket::utility::policy::make_sequential(), state, target_qubit, control_qubit); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_controlled_not(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    { ::ket::gate::controlled_not(parallel_policy, first, last, target_qubit, control_qubit); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_controlled_not(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    { ::ket::gate::controlled_not(first, last, target_qubit, control_qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_not(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      { return ::ket::gate::ranges::controlled_not(parallel_policy, state, target_qubit, control_qubit); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_not(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      { return ::ket::gate::ranges::controlled_not(state, target_qubit, control_qubit); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_CONTROLLED_NOT_HPP
