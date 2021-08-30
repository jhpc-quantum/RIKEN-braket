#ifndef KET_GATE_TOFFOLI_HPP
# define KET_GATE_TOFFOLI_HPP

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
    inline void toffoli(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    {
      ::ket::gate::gate(
        parallel_policy, first, last,
        [](RandomAccessIterator const first, std::array<StateInteger, 8u> const& indices, int const)
        {
# ifndef BOOST_NO_CXX14_BINARY_LITERALS
          std::iter_swap(first + indices[0b110u], first + indices[0b111u]);
# else // BOOST_NO_CXX14_BINARY_LITERALS
          std::iter_swap(first + indices[6u], first + indices[7u]);
# endif // BOOST_NO_CXX14_BINARY_LITERALS
        },
        target_qubit, control_qubit1, control_qubit2);
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void toffoli(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    { ::ket::gate::toffoli(::ket::utility::policy::make_sequential(), first, last, target_qubit, control_qubit1, control_qubit2); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& toffoli(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      {
        ::ket::gate::toffoli(parallel_policy, std::begin(state), std::end(state), target_qubit, control_qubit1, control_qubit2);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& toffoli(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      { return ::ket::gate::ranges::toffoli(::ket::utility::policy::make_sequential(), state, target_qubit, control_qubit1, control_qubit2); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_toffoli(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    { ::ket::gate::toffoli(parallel_policy, first, last, target_qubit, control_qubit1, control_qubit2); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_toffoli(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    { ::ket::gate::toffoli(first, last, target_qubit, control_qubit1, control_qubit2); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_toffoli(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      { return ::ket::gate::ranges::toffoli(parallel_policy, state, target_qubit, control_qubit1, control_qubit2); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_toffoli(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      { return ::ket::gate::ranges::toffoli(state, target_qubit, control_qubit1, control_qubit2); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_TOFFOLI_HPP
