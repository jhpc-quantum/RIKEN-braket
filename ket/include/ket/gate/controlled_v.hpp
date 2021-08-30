#ifndef KET_GATE_CONTROLLED_V_HPP
# define KET_GATE_CONTROLLED_V_HPP

# include <boost/config.hpp>

# include <complex>
# include <array>
# include <iterator>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/exp_i.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    // controlled_v_coeff
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void controlled_v_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      auto const one_plus_phase_coefficient = complex_type{real_type{1}} + phase_coefficient;
      auto const one_minus_phase_coefficient = complex_type{real_type{1}} - phase_coefficient;

      ::ket::gate::gate(
        parallel_policy, first, last,
        [&one_plus_phase_coefficient, &one_minus_phase_coefficient](RandomAccessIterator const first, std::array<StateInteger, 4u> const& indices, int const)
        {
# ifndef BOOST_NO_CXX14_BINARY_LITERALS
          auto const control_on_iter = first + indices[0b10u];
          auto const target_control_on_iter = first + indices[0b11u];
# else // BOOST_NO_CXX14_BINARY_LITERALS
          auto const control_on_iter = first + indices[2u];
          auto const target_control_on_iter = first + indices[3u];
# endif // BOOST_NO_CXX14_BINARY_LITERALS
          auto const control_on_iter_value = *control_on_iter;

          using boost::math::constants::half;
          *control_on_iter
            = half<real_type>()
              * (one_plus_phase_coefficient * control_on_iter_value
                 + one_minus_phase_coefficient * (*target_control_on_iter));
          *target_control_on_iter
            = half<real_type>()
              * (one_minus_phase_coefficient * control_on_iter_value
                 + one_plus_phase_coefficient * (*target_control_on_iter));
        },
        target_qubit, control_qubit);
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void controlled_v_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    { ::ket::gate::controlled_v_coeff(::ket::utility::policy::make_sequential(), first, last, phase_coefficient, target_qubit, control_qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_v_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      {
        ::ket::gate::controlled_v_coeff(parallel_policy, std::begin(state), std::end(state), phase_coefficient, target_qubit, control_qubit);
        return state;
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_v_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      { return ::ket::gate::ranges::controlled_v_coeff(::ket::utility::policy::make_sequential(), state, phase_coefficient, target_qubit, control_qubit); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void adj_controlled_v_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      using std::conj;
      ::ket::gate::controlled_v_coeff(parallel_policy, first, last, conj(phase_coefficient), target_qubit, control_qubit);
    }

    template <typename RandomAccessIterator, typename Complex, typename StateInteger, typename BitInteger>
    inline void adj_controlled_v_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      using std::conj;
      ::ket::gate::controlled_v_coeff(first, last, conj(phase_coefficient), target_qubit, control_qubit);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_v_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      {
        using std::conj;
        return ::ket::gate::ranges::controlled_v_coeff(parallel_policy, state, conj(phase_coefficient), target_qubit, control_qubit);
      }

      template <typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_ontrolled_v_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      {
        using std::conj;
        return ::ket::gate::ranges::controlled_v_coeff(state, conj(phase_coefficient), target_qubit, control_qubit);
      }
    } // namespace ranges


    // controlled_v
    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void controlled_v(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_v_coeff(parallel_policy, first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
    }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void controlled_v(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_v_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_v(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const& phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::controlled_v_coeff(parallel_policy, state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
      }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_v(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::controlled_v_coeff(state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
      }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void adj_controlled_v(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    { ::ket::gate::controlled_v(parallel_policy, first, last, -phase, target_qubit, control_qubit); }

    template <typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
    inline void adj_controlled_v(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    { ::ket::gate::controlled_v(first, last, -phase, target_qubit, control_qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_v(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const& phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      { return ::ket::gate::ranges::controlled_v(parallel_policy, state, -phase, target_qubit, control_qubit); }

      template <typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_v(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      { return ::ket::gate::ranges::controlled_v(state, -phase, target_qubit, control_qubit); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_CONTROLLED_V_HPP
