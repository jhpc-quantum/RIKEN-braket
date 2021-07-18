#ifndef KET_GATE_CONTROLLED_PHASE_SHIFT_HPP
# define KET_GATE_CONTROLLED_PHASE_SHIFT_HPP

# include <boost/config.hpp>

# include <complex>
# include <array>
# include <iterator>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/exp_i.hpp>


namespace ket
{
  namespace gate
  {
    // controlled_phase_shift_coeff
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void controlled_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      ::ket::gate::gate(
        parallel_policy, first, last,
        [&phase_coefficient](RandomAccessIterator const first, std::array<StateInteger, 4u> const& indices, int const)
        {
# ifndef BOOST_NO_CXX14_BINARY_LITERALS
          *(first + indices[0b11u]) *= phase_coefficient;
# else // BOOST_NO_CXX14_BINARY_LITERALS
          *(first + indices[3u]) *= phase_coefficient;
# endif // BOOST_NO_CXX14_BINARY_LITERALS
        },
        target_qubit, control_qubit);
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void controlled_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    { ::ket::gate::controlled_phase_shift_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, target_qubit, control_qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      {
        ::ket::gate::controlled_phase_shift_coeff(parallel_policy, std::begin(state), std::end(state), phase_coefficient, target_qubit, control_qubit);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      { return ::ket::gate::ranges::controlled_phase_shift_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, target_qubit, control_qubit); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void adj_controlled_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      using std::conj;
      ::ket::gate::controlled_phase_shift_coeff(parallel_policy, first, last, conj(phase_coefficient), target_qubit, control_qubit);
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void adj_controlled_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      using std::conj;
      ::ket::gate::controlled_phase_shift_coeff(first, last, conj(phase_coefficient), target_qubit, control_qubit);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      {
        using std::conj;
        return ::ket::gate::ranges::controlled_phase_shift_coeff(parallel_policy, state, conj(phase_coefficient), target_qubit, control_qubit);
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      {
        using std::conj;
        return ::ket::gate::ranges::controlled_phase_shift_coeff(state, conj(phase_coefficient), target_qubit, control_qubit);
      }
    } // namespace ranges

    // controlled_phase_shift
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void controlled_phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_phase_shift_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void controlled_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_phase_shift_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::controlled_phase_shift_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_phase_shift(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::controlled_phase_shift_coeff(state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void adj_controlled_phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    { ::ket::gate::controlled_phase_shift(parallel_policy, first, last, -phase, target_qubit, control_qubit); }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void adj_controlled_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    { ::ket::gate::controlled_phase_shift(first, last, -phase, target_qubit, control_qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      { return ::ket::gate::ranges::controlled_phase_shift(parallel_policy, state, -phase, target_qubit, control_qubit); }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_phase_shift(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      { return ::ket::gate::ranges::controlled_phase_shift(state, -phase, target_qubit, control_qubit); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_CONTROLLED_PHASE_SHIFT_HPP
