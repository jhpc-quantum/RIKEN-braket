#ifndef KET_GATE_PHASE_SHIFT_HPP
# define KET_GATE_PHASE_SHIFT_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cmath>
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
# endif
# include <iterator>
# include <utility>
# ifndef NDEBUG
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     include <type_traits>
#   else
#     include <boost/type_traits/is_unsigned.hpp>
#     include <boost/type_traits/is_same.hpp>
#   endif
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif

# include <boost/math/constants/constants.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/iterator.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/exp_i.hpp>
# include <ket/utility/meta/real_of.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_is_unsigned std::is_unsigned
#   define KET_is_same std::is_same
# else
#   define KET_is_unsigned boost::is_unsigned
#   define KET_is_same boost::is_same
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     define KET_addressof std::addressof
#   else
#     define KET_addressof boost::addressof
#   endif
# endif


namespace ket
{
  namespace gate
  {
    // phase_shift_coeff
    namespace phase_shift_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <
        typename RandomAccessIterator,
        typename Complex, typename StateInteger>
      struct phase_shift_coeff_loop_inside
      {
        RandomAccessIterator first_;
        Complex phase_coefficient_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;

        phase_shift_coeff_loop_inside(
          RandomAccessIterator const first,
          Complex const& phase_coefficient,
          StateInteger const qubit_mask,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask)
          : first_(first),
            phase_coefficient_(phase_coefficient),
            qubit_mask_(qubit_mask),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask)
        { }

        void operator()(StateInteger const value_wo_qubit, int const) const
        {
          // xxxxx1xxxxxx
          StateInteger const one_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_) bitor qubit_mask_;
          *(first_+one_index) *= phase_coefficient_;
        }
      };

      template <
        typename RandomAccessIterator,
        typename Complex, typename StateInteger>
      inline
      phase_shift_coeff_loop_inside<
        RandomAccessIterator, Complex, StateInteger>
      make_phase_shift_coeff_loop_inside(
        RandomAccessIterator const first,
        Complex const& phase_coefficient,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask)
      {
        return phase_shift_coeff_loop_inside<
          RandomAccessIterator, Complex, StateInteger>(
            first, phase_coefficient,
            qubit_mask, lower_bits_mask, upper_bits_mask);
      }
# endif // BOOST_NO_CXX11_LAMBDAS

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
          KET_is_unsigned<StateInteger>::value,
          "StateInteger should be unsigned");
        static_assert(
          KET_is_unsigned<BitInteger>::value,
          "BitInteger should be unsigned");
        static_assert(
          (KET_is_same<
             Complex,
             typename std::iterator_traits<RandomAccessIterator>::value_type>::value),
          "Complex must be the same to value_type of RandomAccessIterator");

        assert(
          ::ket::utility::integer_exp2<StateInteger>(qubit)
          < static_cast<StateInteger>(last-first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last-first))
          == static_cast<StateInteger>(last-first));

        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(qubit);
        StateInteger const lower_bits_mask
          = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          [first, phase_coefficient,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx1xxxxxx
            StateInteger const one_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask) bitor qubit_mask;
            *(first+one_index) *= phase_coefficient;
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::phase_shift_detail::make_phase_shift_coeff_loop_inside(
            first, phase_coefficient,
            qubit_mask, lower_bits_mask, upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
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
          boost::begin(state), boost::end(state), phase_coefficient, qubit);
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
          boost::begin(state), boost::end(state), phase_coefficient, qubit);
        return state;
      }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& phase_shift_coeff(
        std::vector<Complex, Allocator>& state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift_coeff_impl(
          ::ket::utility::policy::make_sequential(),
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          phase_coefficient, qubit);
        return state;
      }

      template <
        typename ParallelPolicy,
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        std::vector<Complex, Allocator>& state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift_coeff_impl(
          parallel_policy,
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          phase_coefficient, qubit);
        return state;
      }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
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
      typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
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
      typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
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
        typedef typename boost::range_value<RandomAccessRange>::type complex_type;
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
        typedef typename boost::range_value<RandomAccessRange>::type complex_type;
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
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <
        typename RandomAccessIterator,
        typename Complex, typename StateInteger>
      struct phase_shift2_loop_inside
      {
        RandomAccessIterator first_;
        Complex modified_phase_coefficient1_;
        Complex phase_coefficient2_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;

        phase_shift2_loop_inside(
          RandomAccessIterator const first,
          Complex const& modified_phase_coefficient1,
          Complex const& phase_coefficient2,
          StateInteger const qubit_mask,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask)
          : first_(first),
            modified_phase_coefficient1_(modified_phase_coefficient1),
            phase_coefficient2_(phase_coefficient2),
            qubit_mask_(qubit_mask),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask)
        { }

        void operator()(StateInteger const value_wo_qubit, int const) const
        {
          // xxxxx0xxxxxx
          StateInteger const zero_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_);
          // xxxxx1xxxxxx
          StateInteger const one_index = zero_index bitor qubit_mask_;
          RandomAccessIterator const zero_iter = first_+zero_index;
          RandomAccessIterator const one_iter = first_+one_index;
          Complex const zero_iter_value = *zero_iter;

          typedef
            typename ::ket::utility::meta::real_of<Complex>::type real_type;
          using boost::math::constants::one_div_root_two;
          *zero_iter -= phase_coefficient2_ * *one_iter;
          *zero_iter *= one_div_root_two<real_type>();
          *one_iter *= phase_coefficient2_;
          *one_iter += zero_iter_value;
          *one_iter *= modified_phase_coefficient1_;
        }
      };

      template <
        typename RandomAccessIterator,
        typename Complex, typename StateInteger>
      inline
      phase_shift2_loop_inside<
        RandomAccessIterator, Complex, StateInteger>
      make_phase_shift2_loop_inside(
        RandomAccessIterator const first,
        Complex const& modified_phase_coefficient1,
        Complex const& phase_coefficient2,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask)
      {
        return phase_shift2_loop_inside<
          RandomAccessIterator, Complex, StateInteger>(
            first, modified_phase_coefficient1, phase_coefficient2,
            qubit_mask, lower_bits_mask, upper_bits_mask);
      }
# endif // BOOST_NO_CXX11_LAMBDAS

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
          KET_is_unsigned<StateInteger>::value,
          "StateInteger should be unsigned");
        static_assert(
          KET_is_unsigned<BitInteger>::value,
          "BitInteger should be unsigned");
        static_assert(
          (KET_is_same<
             Real,
             typename ::ket::utility::meta::real_of<
               typename std::iterator_traits<RandomAccessIterator>::value_type>::type>::value),
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        assert(
          ::ket::utility::integer_exp2<StateInteger>(qubit)
          < static_cast<StateInteger>(last-first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last-first))
          == static_cast<StateInteger>(last-first));

        typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
        complex_type const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
        complex_type const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

        using boost::math::constants::one_div_root_two;
        complex_type const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(qubit);
        StateInteger const lower_bits_mask
          = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          [first, modified_phase_coefficient1, phase_coefficient2,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            StateInteger const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            StateInteger const one_index = zero_index bitor qubit_mask;
            RandomAccessIterator const zero_iter = first+zero_index;
            RandomAccessIterator const one_iter = first+one_index;
            complex_type const zero_iter_value = *zero_iter;

            *zero_iter -= phase_coefficient2 * *one_iter;
            *zero_iter *= one_div_root_two<Real>();
            *one_iter *= phase_coefficient2;
            *one_iter += zero_iter_value;
            *one_iter *= modified_phase_coefficient1;
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::phase_shift_detail::make_phase_shift2_loop_inside(
            first, modified_phase_coefficient1, phase_coefficient2,
            qubit_mask, lower_bits_mask, upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
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
          boost::begin(state), boost::end(state), phase1, phase2, qubit);
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
          boost::begin(state), boost::end(state), phase1, phase2, qubit);
        return state;
      }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& phase_shift2(
        std::vector<Complex, Allocator>& state,
        typename ::ket::utility::meta::real_of<Complex>::type const phase1,
        typename ::ket::utility::meta::real_of<Complex>::type const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift2_impl(
          ::ket::utility::policy::make_sequential(),
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          phase1, phase2, qubit);
        return state;
      }

      template <
        typename ParallelPolicy,
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& phase_shift2(
        ParallelPolicy const parallel_policy,
        std::vector<Complex, Allocator>& state,
        typename ::ket::utility::meta::real_of<Complex>::type const phase1,
        typename ::ket::utility::meta::real_of<Complex>::type const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift2_impl(
          parallel_policy,
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          phase1, phase2, qubit);
        return state;
      }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    } // namespace ranges


    namespace phase_shift_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <
        typename RandomAccessIterator,
        typename Complex, typename StateInteger>
      struct adj_phase_shift2_loop_inside
      {
        RandomAccessIterator first_;
        Complex phase_coefficient1_;
        Complex modified_phase_coefficient2_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;

        adj_phase_shift2_loop_inside(
          RandomAccessIterator const first,
          Complex const& phase_coefficient1,
          Complex const& modified_phase_coefficient2,
          StateInteger const qubit_mask,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask)
          : first_(first),
            phase_coefficient1_(phase_coefficient1),
            modified_phase_coefficient2_(modified_phase_coefficient2),
            qubit_mask_(qubit_mask),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask)
        { }

        void operator()(StateInteger const value_wo_qubit, int const) const
        {
          // xxxxx0xxxxxx
          StateInteger const zero_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_);
          // xxxxx1xxxxxx
          StateInteger const one_index = zero_index bitor qubit_mask_;
          RandomAccessIterator const zero_iter = first_+zero_index;
          RandomAccessIterator const one_iter = first_+one_index;
          Complex const zero_iter_value = *zero_iter;

          typedef
            typename ::ket::utility::meta::real_of<Complex>::type real_type;
          using boost::math::constants::one_div_root_two;
          *zero_iter += phase_coefficient1_ * *one_iter;
          *zero_iter *= one_div_root_two<real_type>();
          *one_iter *= phase_coefficient1_;
          *one_iter -= zero_iter_value;
          *one_iter *= modified_phase_coefficient2_;
        }
      };

      template <
        typename RandomAccessIterator,
        typename Complex, typename StateInteger>
      inline
      adj_phase_shift2_loop_inside<
        RandomAccessIterator, Complex, StateInteger>
      make_adj_phase_shift2_loop_inside(
        RandomAccessIterator const first,
        Complex const& phase_coefficient1,
        Complex const& modified_phase_coefficient2,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask)
      {
        return adj_phase_shift2_loop_inside<
          RandomAccessIterator, Complex, StateInteger>(
            first, phase_coefficient1, modified_phase_coefficient2,
            qubit_mask, lower_bits_mask, upper_bits_mask);
      }
# endif // BOOST_NO_CXX11_LAMBDAS

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
          KET_is_unsigned<StateInteger>::value,
          "StateInteger should be unsigned");
        static_assert(
          KET_is_unsigned<BitInteger>::value,
          "BitInteger should be unsigned");
        static_assert(
          (KET_is_same<
             Real,
             typename ::ket::utility::meta::real_of<
               typename std::iterator_traits<RandomAccessIterator>::value_type>::type>::value),
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        assert(
          ::ket::utility::integer_exp2<StateInteger>(qubit)
          < static_cast<StateInteger>(last-first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last-first))
          == static_cast<StateInteger>(last-first));

        typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
        complex_type const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
        complex_type const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

        using boost::math::constants::one_div_root_two;
        complex_type const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(qubit);
        StateInteger const lower_bits_mask
          = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          [first, phase_coefficient1, modified_phase_coefficient2,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            StateInteger const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            StateInteger const one_index = zero_index bitor qubit_mask;
            RandomAccessIterator const zero_iter = first+zero_index;
            RandomAccessIterator const one_iter = first+one_index;
            complex_type const zero_iter_value = *zero_iter;

            *zero_iter += phase_coefficient1 * *one_iter;
            *zero_iter *= one_div_root_two<Real>();
            *one_iter *= phase_coefficient1;
            *one_iter -= zero_iter_value;
            *one_iter *= modified_phase_coefficient2;
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::phase_shift_detail::make_adj_phase_shift2_loop_inside(
            first, phase_coefficient1, modified_phase_coefficient2,
            qubit_mask, lower_bits_mask, upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
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
          boost::begin(state), boost::end(state), phase1, phase2, qubit);
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
          boost::begin(state), boost::end(state), phase1, phase2, qubit);
        return state;
      }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& adj_phase_shift2(
        std::vector<Complex, Allocator>& state,
        typename ::ket::utility::meta::real_of<Complex>::type const phase1,
        typename ::ket::utility::meta::real_of<Complex>::type const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::adj_phase_shift2_impl(
          ::ket::utility::policy::make_sequential(),
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          phase1, phase2, qubit);
        return state;
      }

      template <
        typename ParallelPolicy,
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& adj_phase_shift2(
        ParallelPolicy const parallel_policy,
        std::vector<Complex, Allocator>& state,
        typename ::ket::utility::meta::real_of<Complex>::type const phase1,
        typename ::ket::utility::meta::real_of<Complex>::type const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::adj_phase_shift2_impl(
          parallel_policy,
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          phase1, phase2, qubit);
        return state;
      }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    } // namespace ranges


    namespace phase_shift_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <
        typename RandomAccessIterator,
        typename Real, typename Complex, typename StateInteger>
      struct phase_shift3_loop_inside
      {
        RandomAccessIterator first_;
        Real sine_;
        Real cosine_;
        Complex phase_coefficient2_;
        Complex sine_phase_coefficient3_;
        Complex cosine_phase_coefficient3_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;

        phase_shift3_loop_inside(
          RandomAccessIterator const first,
          Real const sine, Real const cosine,
          Complex const& phase_coefficient2,
          Complex const& sine_phase_coefficient3,
          Complex const& cosine_phase_coefficient3,
          StateInteger const qubit_mask,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask)
          : first_(first),
            sine_(sine),
            cosine_(cosine),
            phase_coefficient2_(phase_coefficient2),
            sine_phase_coefficient3_(sine_phase_coefficient3),
            cosine_phase_coefficient3_(cosine_phase_coefficient3),
            qubit_mask_(qubit_mask),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask)
        { }

        void operator()(StateInteger const value_wo_qubit, int const) const
        {
          // xxxxx0xxxxxx
          StateInteger const zero_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_);
          // xxxxx1xxxxxx
          StateInteger const one_index = zero_index bitor qubit_mask_;
          RandomAccessIterator const zero_iter = first_+zero_index;
          RandomAccessIterator const one_iter = first_+one_index;
          Complex const zero_iter_value = *zero_iter;

          *zero_iter *= cosine_;
          *zero_iter -= sine_phase_coefficient3_ * *one_iter;
          *one_iter *= cosine_phase_coefficient3_;
          *one_iter += sine_ * zero_iter_value;
          *one_iter *= phase_coefficient2_;
        }
      };

      template <
        typename RandomAccessIterator,
        typename Real, typename Complex, typename StateInteger>
      inline
      phase_shift3_loop_inside<
        RandomAccessIterator, Real, Complex, StateInteger>
      make_phase_shift3_loop_inside(
        RandomAccessIterator const first,
        Real const sine, Real const cosine,
        Complex const& phase_coefficient2,
        Complex const& sine_phase_coefficient3,
        Complex const& cosine_phase_coefficient3,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask)
      {
        return phase_shift3_loop_inside<
          RandomAccessIterator, Real, Complex, StateInteger>(
            first, sine, cosine, phase_coefficient2,
            sine_phase_coefficient3, cosine_phase_coefficient3,
            qubit_mask, lower_bits_mask, upper_bits_mask);
      }
# endif // BOOST_NO_CXX11_LAMBDAS

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
          KET_is_unsigned<StateInteger>::value,
          "StateInteger should be unsigned");
        static_assert(
          KET_is_unsigned<BitInteger>::value,
          "BitInteger should be unsigned");
        static_assert(
          (KET_is_same<
             Real,
             typename ::ket::utility::meta::real_of<
               typename std::iterator_traits<RandomAccessIterator>::value_type>::type>::value),
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        assert(
          ::ket::utility::integer_exp2<StateInteger>(qubit)
          < static_cast<StateInteger>(last-first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last-first))
          == static_cast<StateInteger>(last-first));

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        Real const sine = sin(half<Real>() * phase1);
        Real const cosine = cos(half<Real>() * phase1);

        typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
        complex_type const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
        complex_type const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

        complex_type const sine_phase_coefficient3 = sine * phase_coefficient3;
        complex_type const cosine_phase_coefficient3 = cosine * phase_coefficient3;

        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(qubit);
        StateInteger const lower_bits_mask
          = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          [first, sine, cosine, phase_coefficient2,
           sine_phase_coefficient3, cosine_phase_coefficient3,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            StateInteger const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            StateInteger const one_index = zero_index bitor qubit_mask;
            RandomAccessIterator const zero_iter = first+zero_index;
            RandomAccessIterator const one_iter = first+one_index;
            complex_type const zero_iter_value = *zero_iter;

            *zero_iter *= cosine;
            *zero_iter -= sine_phase_coefficient3 * *one_iter;
            *one_iter *= cosine_phase_coefficient3;
            *one_iter += sine * zero_iter_value;
            *one_iter *= phase_coefficient2;
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::phase_shift_detail::make_phase_shift3_loop_inside(
            first, sine, cosine, phase_coefficient2,
            sine_phase_coefficient3, cosine_phase_coefficient3,
            qubit_mask, lower_bits_mask, upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
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
          boost::begin(state), boost::end(state), phase1, phase2, phase3, qubit);
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
          boost::begin(state), boost::end(state), phase1, phase2, phase3, qubit);
        return state;
      }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& phase_shift3(
        std::vector<Complex, Allocator>& state,
        typename ::ket::utility::meta::real_of<Complex>::type const phase1,
        typename ::ket::utility::meta::real_of<Complex>::type const phase2,
        typename ::ket::utility::meta::real_of<Complex>::type const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift3_impl(
          ::ket::utility::policy::make_sequential(),
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          phase1, phase2, phase3, qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& phase_shift3(
        ParallelPolicy const parallel_policy,
        std::vector<Complex, Allocator>& state,
        typename ::ket::utility::meta::real_of<Complex>::type const phase1,
        typename ::ket::utility::meta::real_of<Complex>::type const phase2,
        typename ::ket::utility::meta::real_of<Complex>::type const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::phase_shift3_impl(
          parallel_policy,
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          phase1, phase2, phase3, qubit);
        return state;
      }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    } // namespace ranges


    namespace phase_shift_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <
        typename RandomAccessIterator,
        typename Real, typename Complex, typename StateInteger>
      struct adj_phase_shift3_loop_inside
      {
        RandomAccessIterator first_;
        Real sine_;
        Real cosine_;
        Complex sine_phase_coefficient2_;
        Complex cosine_phase_coefficient2_;
        Complex phase_coefficient3_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;

        adj_phase_shift3_loop_inside(
          RandomAccessIterator const first,
          Real const sine, Real const cosine,
          Complex const& sine_phase_coefficient2,
          Complex const& cosine_phase_coefficient2,
          Complex const& phase_coefficient3,
          StateInteger const qubit_mask,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask)
          : first_(first),
            sine_(sine),
            cosine_(cosine),
            sine_phase_coefficient2_(sine_phase_coefficient2),
            cosine_phase_coefficient2_(cosine_phase_coefficient2),
            phase_coefficient3_(phase_coefficient3),
            qubit_mask_(qubit_mask),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask)
        { }

        void operator()(StateInteger const value_wo_qubit, int const) const
        {
          // xxxxx0xxxxxx
          StateInteger const zero_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_);
          // xxxxx1xxxxxx
          StateInteger const one_index = zero_index bitor qubit_mask_;
          RandomAccessIterator const zero_iter = first_+zero_index;
          RandomAccessIterator const one_iter = first_+one_index;
          Complex const zero_iter_value = *zero_iter;

          *zero_iter *= cosine_;
          *zero_iter += sine_phase_coefficient2_ * *one_iter;
          *one_iter *= cosine_phase_coefficient2_;
          *one_iter -= sine_ * zero_iter_value;
          *one_iter *= phase_coefficient3_;
        }
      };

      template <
        typename RandomAccessIterator,
        typename Real, typename Complex, typename StateInteger>
      inline
      adj_phase_shift3_loop_inside<
        RandomAccessIterator, Real, Complex, StateInteger>
      make_adj_phase_shift3_loop_inside(
        RandomAccessIterator const first,
        Real const sine, Real const cosine,
        Complex const& sine_phase_coefficient2,
        Complex const& cosine_phase_coefficient2,
        Complex const& phase_coefficient3,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask)
      {
        return adj_phase_shift3_loop_inside<
          RandomAccessIterator, Real, Complex, StateInteger>(
            first, sine, cosine,
            sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3,
            qubit_mask, lower_bits_mask, upper_bits_mask);
      }
# endif // BOOST_NO_CXX11_LAMBDAS

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
          KET_is_unsigned<StateInteger>::value,
          "StateInteger should be unsigned");
        static_assert(
          KET_is_unsigned<BitInteger>::value,
          "BitInteger should be unsigned");
        static_assert(
          (KET_is_same<
             Real,
             typename ::ket::utility::meta::real_of<
               typename std::iterator_traits<RandomAccessIterator>::value_type>::type>::value),
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        assert(
          ::ket::utility::integer_exp2<StateInteger>(qubit)
          < static_cast<StateInteger>(last-first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last-first))
          == static_cast<StateInteger>(last-first));

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        Real const sine = sin(half<Real>() * phase1);
        Real const cosine = cos(half<Real>() * phase1);

        typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
        complex_type const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
        complex_type const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

        complex_type const sine_phase_coefficient2 = sine * phase_coefficient2;
        complex_type const cosine_phase_coefficient2 = cosine * phase_coefficient2;

        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(qubit);
        StateInteger const lower_bits_mask
          = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          [first, sine, cosine,
           sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            StateInteger const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            StateInteger const one_index = zero_index bitor qubit_mask;
            RandomAccessIterator const zero_iter = first+zero_index;
            RandomAccessIterator const one_iter = first+one_index;
            complex_type const zero_iter_value = *zero_iter;

            *zero_iter *= cosine;
            *zero_iter += sine_phase_coefficient2 * *one_iter;
            *one_iter *= cosine_phase_coefficient2;
            *one_iter -= sine * zero_iter_value;
            *one_iter *= phase_coefficient3;
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::phase_shift_detail::make_adj_phase_shift3_loop_inside(
            first, sine, cosine,
            sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3,
            qubit_mask, lower_bits_mask, upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
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
          boost::begin(state), boost::end(state), phase1, phase2, phase3, qubit);
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
          boost::begin(state), boost::end(state), phase1, phase2, phase3, qubit);
        return state;
      }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& adj_phase_shift3(
        std::vector<Complex, Allocator>& state,
        typename ::ket::utility::meta::real_of<Complex>::type const phase1,
        typename ::ket::utility::meta::real_of<Complex>::type const phase2,
        typename ::ket::utility::meta::real_of<Complex>::type const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::adj_phase_shift3_impl(
          ::ket::utility::policy::make_sequential(),
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          phase1, phase2, phase3, qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& adj_phase_shift3(
        ParallelPolicy const parallel_policy,
        std::vector<Complex, Allocator>& state,
        typename ::ket::utility::meta::real_of<Complex>::type const phase1,
        typename ::ket::utility::meta::real_of<Complex>::type const phase2,
        typename ::ket::utility::meta::real_of<Complex>::type const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::phase_shift_detail::adj_phase_shift3_impl(
          parallel_policy,
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          phase1, phase2, phase3, qubit);
        return state;
      }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    } // namespace ranges
  } // namespace gate
} // namespace ket


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif
# undef KET_is_same
# undef KET_is_unsigned
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

