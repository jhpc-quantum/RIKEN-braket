#ifndef KET_GATE_PHASE_SHIFT_HPP
# define KET_GATE_PHASE_SHIFT_HPP

# include <cassert>
# include <cmath>
# include <iterator>
# include <utility>
# include <type_traits>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
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
    // phase_shift_coeff
    namespace phase_shift_detail
    {
      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename Complex, typename StateInteger, typename BitInteger>
      void phase_shift_coeff_impl(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        static_assert(
          std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        static_assert(
          (std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value),
          "Complex must be the same to value_type of RandomAccessIterator");

        assert(
          ::ket::utility::integer_exp2<StateInteger>(qubit)
          < static_cast<StateInteger>(last - first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last - first))
          == static_cast<StateInteger>(last - first));

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last - first) / 2u,
          [first, phase_coefficient,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx1xxxxxx
            auto const one_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask) bitor qubit_mask;
            *(first + one_index) *= phase_coefficient;
          });
      }
    } // namespace phase_shift_detail

    template <
      typename RandomAccessIterator, typename Complex,
      typename StateInteger, typename BitInteger>
    inline void phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::phase_shift_detail::phase_shift_coeff_impl(
        ::ket::utility::policy::make_sequential(), first, last, phase_coefficient, qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Complex,
      typename StateInteger, typename BitInteger>
    inline void phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::phase_shift_detail::phase_shift_coeff_impl(
        parallel_policy, first, last, phase_coefficient, qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& phase_shift_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift_coeff_impl(
          ::ket::utility::policy::make_sequential(),
          ::ket::utility::begin(state), ::ket::utility::end(state), phase_coefficient, qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift_coeff_impl(
          parallel_policy,
          ::ket::utility::begin(state), ::ket::utility::end(state), phase_coefficient, qubit);
        return state;
      }
    } // namespace ranges


    template <
      typename RandomAccessIterator, typename Complex,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { using std::conj; ::ket::gate::phase_shift_coeff(first, last, conj(phase_coefficient), qubit); }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Complex,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      using std::conj;
      ::ket::gate::phase_shift_coeff(
        parallel_policy, first, last, conj(phase_coefficient), qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { using std::conj; return ::ket::gate::ranges::phase_shift_coeff(state, conj(phase_coefficient), qubit); }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        using std::conj;
        return ::ket::gate::ranges::phase_shift_coeff(
          parallel_policy, state, conj(phase_coefficient), qubit);
      }
    } // namespace ranges


    // phase_shift
    template <
      typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::phase_shift_coeff(first, last, ::ket::utility::exp_i<complex_type>(phase), qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      ::ket::gate::phase_shift_coeff(
        parallel_policy,
        first, last, ::ket::utility::exp_i<complex_type>(phase), qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& phase_shift(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::phase_shift_coeff(state, ::ket::utility::exp_i<complex_type>(phase), qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        return ::ket::gate::ranges::phase_shift_coeff(
          parallel_policy,
          state, ::ket::utility::exp_i<complex_type>(phase), qubit);
      }
    } // namespace ranges


    template <
      typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::phase_shift(first, last, -phase, qubit); }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::phase_shift(parallel_policy, first, last, -phase, qubit); }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::phase_shift(state, -phase, qubit); }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::phase_shift(parallel_policy, state, -phase, qubit); }
    } // namespace ranges


    // generalized phase_shift
    namespace phase_shift_detail
    {
      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename Real, typename StateInteger, typename BitInteger>
      void phase_shift2_impl(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        static_assert(
          std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          (std::is_same<
             Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        assert(
          ::ket::utility::integer_exp2<StateInteger>(qubit)
          < static_cast<StateInteger>(last - first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last - first))
          == static_cast<StateInteger>(last - first));

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last - first) / 2u,
          [first, modified_phase_coefficient1, phase_coefficient2,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            auto const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            auto const one_index = zero_index bitor qubit_mask;
            auto const zero_iter = first + zero_index;
            auto const one_iter = first + one_index;
            auto const zero_iter_value = *zero_iter;

            *zero_iter -= phase_coefficient2 * *one_iter;
            *zero_iter *= one_div_root_two<Real>();
            *one_iter *= phase_coefficient2;
            *one_iter += zero_iter_value;
            *one_iter *= modified_phase_coefficient1;
          });
      }
    } // namespace phase_shift_detail

    template <
      typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void phase_shift2(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::phase_shift_detail::phase_shift2_impl(
        ::ket::utility::policy::make_sequential(), first, last, phase1, phase2, qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::phase_shift_detail::phase_shift2_impl(
        parallel_policy, first, last, phase1, phase2, qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& phase_shift2(
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift2_impl(
          ::ket::utility::policy::make_sequential(),
          ::ket::utility::begin(state), ::ket::utility::end(state), phase1, phase2, qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& phase_shift2(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift2_impl(
          parallel_policy,
          ::ket::utility::begin(state), ::ket::utility::end(state), phase1, phase2, qubit);
        return state;
      }
    } // namespace ranges


    namespace phase_shift_detail
    {
      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename Real, typename StateInteger, typename BitInteger>
      void adj_phase_shift2_impl(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        static_assert(
          std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          (std::is_same<
             Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        assert(
          ::ket::utility::integer_exp2<StateInteger>(qubit)
          < static_cast<StateInteger>(last - first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last - first))
          == static_cast<StateInteger>(last - first));

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last - first) / 2u,
          [first, phase_coefficient1, modified_phase_coefficient2,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            auto const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            auto const one_index = zero_index bitor qubit_mask;
            auto const zero_iter = first + zero_index;
            auto const one_iter = first + one_index;
            auto const zero_iter_value = *zero_iter;

            *zero_iter += phase_coefficient1 * *one_iter;
            *zero_iter *= one_div_root_two<Real>();
            *one_iter *= phase_coefficient1;
            *one_iter -= zero_iter_value;
            *one_iter *= modified_phase_coefficient2;
          });
      }
    } // namespace phase_shift_detail

    template <
      typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift2(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::phase_shift_detail::adj_phase_shift2_impl(
        ::ket::utility::policy::make_sequential(), first, last, phase1, phase2, qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift2(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::phase_shift_detail::adj_phase_shift2_impl(
        parallel_policy, first, last, phase1, phase2, qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift2(
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::adj_phase_shift2_impl(
          ::ket::utility::policy::make_sequential(),
          ::ket::utility::begin(state), ::ket::utility::end(state), phase1, phase2, qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift2(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::adj_phase_shift2_impl(
          parallel_policy,
          ::ket::utility::begin(state), ::ket::utility::end(state), phase1, phase2, qubit);
        return state;
      }
    } // namespace ranges


    namespace phase_shift_detail
    {
      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename Real, typename StateInteger, typename BitInteger>
      void phase_shift3_impl(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        static_assert(
          std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          (std::is_same<
             Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        assert(
          ::ket::utility::integer_exp2<StateInteger>(qubit)
          < static_cast<StateInteger>(last - first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last - first))
          == static_cast<StateInteger>(last - first));

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

        auto const sine_phase_coefficient3 = sine * phase_coefficient3;
        auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last - first) / 2u,
          [first, sine, cosine, phase_coefficient2,
           sine_phase_coefficient3, cosine_phase_coefficient3,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            auto const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            auto const one_index = zero_index bitor qubit_mask;
            auto const zero_iter = first + zero_index;
            auto const one_iter = first + one_index;
            auto const zero_iter_value = *zero_iter;

            *zero_iter *= cosine;
            *zero_iter -= sine_phase_coefficient3 * *one_iter;
            *one_iter *= cosine_phase_coefficient3;
            *one_iter += sine * zero_iter_value;
            *one_iter *= phase_coefficient2;
          });
      }
    } // namespace phase_shift_detail

    template <
      typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void phase_shift3(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::phase_shift_detail::phase_shift3_impl(
        ::ket::utility::policy::make_sequential(), first, last, phase1, phase2, phase3, qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::phase_shift_detail::phase_shift3_impl(
        parallel_policy, first, last, phase1, phase2, phase3, qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& phase_shift3(
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift3_impl(
          ::ket::utility::policy::make_sequential(),
          ::ket::utility::begin(state), ::ket::utility::end(state), phase1, phase2, phase3, qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& phase_shift3(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift3_impl(
          parallel_policy,
          ::ket::utility::begin(state), ::ket::utility::end(state), phase1, phase2, phase3, qubit);
        return state;
      }
    } // namespace ranges


    namespace phase_shift_detail
    {
      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename Real, typename StateInteger, typename BitInteger>
      void adj_phase_shift3_impl(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        static_assert(
          std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          (std::is_same<
             Real, typename ::ket::utility::meta::real_of<complex_type>::type>::value),
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        assert(
          ::ket::utility::integer_exp2<StateInteger>(qubit)
          < static_cast<StateInteger>(last - first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last - first))
          == static_cast<StateInteger>(last - first));

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

        auto const sine_phase_coefficient2 = sine * phase_coefficient2;
        auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last - first) / 2u,
          [first, sine, cosine,
           sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            auto const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            auto const one_index = zero_index bitor qubit_mask;
            auto const zero_iter = first + zero_index;
            auto const one_iter = first + one_index;
            auto const zero_iter_value = *zero_iter;

            *zero_iter *= cosine;
            *zero_iter += sine_phase_coefficient2 * *one_iter;
            *one_iter *= cosine_phase_coefficient2;
            *one_iter -= sine * zero_iter_value;
            *one_iter *= phase_coefficient3;
          });
      }
    } // namespace phase_shift_detail

    template <
      typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift3(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::phase_shift_detail::adj_phase_shift3_impl(
        ::ket::utility::policy::make_sequential(), first, last, phase1, phase2, phase3, qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift3(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase1, Real const phase2, Real const phase3,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::phase_shift_detail::adj_phase_shift3_impl(
        parallel_policy, first, last, phase1, phase2, phase3, qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift3(
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::adj_phase_shift3_impl(
          ::ket::utility::policy::make_sequential(),
          ::ket::utility::begin(state), ::ket::utility::end(state), phase1, phase2, phase3, qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift3(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::adj_phase_shift3_impl(
          parallel_policy,
          ::ket::utility::begin(state), ::ket::utility::end(state), phase1, phase2, phase3, qubit);
        return state;
      }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_PHASE_SHIFT_HPP
