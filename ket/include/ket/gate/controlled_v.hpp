#ifndef KET_GATE_CONTROLLED_V_HPP
# define KET_GATE_CONTROLLED_V_HPP

# include <cassert>
# include <complex>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/exp_i.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    // controlled_v_coeff
    namespace controlled_v_detail
    {
      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename Complex, typename StateInteger, typename BitInteger>
      void controlled_v_coeff_impl(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        static_assert(
          std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        static_assert(
          (std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value),
          "Complex must be the same to value_type of RandomAccessIterator");

        assert(
          ::ket::utility::integer_exp2<StateInteger>(target_qubit)
            < static_cast<StateInteger>(last - first)
          and ::ket::utility::integer_exp2<StateInteger>(control_qubit.qubit())
                < static_cast<StateInteger>(last - first)
          and target_qubit != control_qubit.qubit());
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last - first))
          == static_cast<StateInteger>(last - first));

        using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
        auto const one_plus_phase_coefficient = Complex{real_type{1}} + phase_coefficient;
        auto const one_minus_phase_coefficient = Complex{real_type{1}} - phase_coefficient;

        auto const minmax_qubits = std::minmax(target_qubit, control_qubit.qubit());
        auto const target_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
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
          static_cast<StateInteger>(last - first) / 4u,
          [first, &one_plus_phase_coefficient, &one_minus_phase_coefficient,
           target_qubit_mask, control_qubit_mask,
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
            auto const target_control_on_index = control_on_index bitor target_qubit_mask;
            auto const control_on_iter = first + control_on_index;
            auto const target_control_on_iter = first + target_control_on_index;
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
          });
      }
    } // namespace controlled_v_detail

    template <
      typename RandomAccessIterator, typename Complex,
      typename StateInteger, typename BitInteger>
    inline void controlled_v_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    {
      ::ket::gate::controlled_v_detail::controlled_v_coeff_impl(
        ::ket::utility::policy::make_sequential(),
        first, last, phase_coefficient, target_qubit, control_qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename Complex, typename StateInteger, typename BitInteger>
    inline void controlled_v_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    {
      ::ket::gate::controlled_v_detail::controlled_v_coeff_impl(
        parallel_policy,
        first, last, phase_coefficient, target_qubit, control_qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_v_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        ::ket::gate::controlled_v_detail::controlled_v_coeff_impl(
          ::ket::utility::policy::make_sequential(),
          ::ket::utility::begin(state), ::ket::utility::end(state), phase_coefficient, target_qubit, control_qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_v_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        ::ket::gate::controlled_v_detail::controlled_v_coeff_impl(
          parallel_policy,
          ::ket::utility::begin(state), ::ket::utility::end(state), phase_coefficient, target_qubit, control_qubit);
        return state;
      }
    } // namespace ranges


    template <
      typename RandomAccessIterator, typename Complex,
      typename StateInteger, typename BitInteger>
    inline void adj_controlled_v_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    {
      using std::conj;
      ::ket::gate::controlled_v_coeff(
        first, last, conj(phase_coefficient), target_qubit, control_qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename Complex, typename StateInteger, typename BitInteger>
    inline void adj_controlled_v_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    {
      using std::conj;
      ::ket::gate::controlled_v_coeff(
        parallel_policy, first, last,
        conj(phase_coefficient), target_qubit, control_qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_ontrolled_v_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        using std::conj;
        return ::ket::gate::ranges::controlled_v_coeff(
          state, conj(phase_coefficient), target_qubit, control_qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename Complex, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_v_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        using std::conj;
        return ::ket::gate::ranges::controlled_v_coeff(
          parallel_policy, state,
          conj(phase_coefficient), target_qubit, control_qubit);
      }
    } // namespace ranges


    // controlled_v
    template <
      typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void controlled_v(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_v_coeff(
        first, last, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename Real, typename StateInteger, typename BitInteger>
    inline void controlled_v(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::controlled_v_coeff(
        parallel_policy, first, last,
        ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_v(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::controlled_v_coeff(
          state, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& controlled_v(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const& phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::controlled_v_coeff(
          parallel_policy, state,
          ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubit);
      }
    } // namespace ranges


    template <
      typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void adj_controlled_v(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    {
      ::ket::gate::controlled_v(
        first, last, -phase, target_qubit, control_qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename Real, typename StateInteger, typename BitInteger>
    inline void adj_controlled_v(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
        control_qubit)
    {
      ::ket::gate::controlled_v(
        parallel_policy, first, last, -phase, target_qubit, control_qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_v(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        return ::ket::gate::ranges::controlled_v(
          state, -phase, target_qubit, control_qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename Real, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_controlled_v(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const& phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const
          control_qubit)
      {
        return ::ket::gate::ranges::controlled_v(
          parallel_policy, state, -phase, target_qubit, control_qubit);
      }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_CONTROLLED_V_HPP
