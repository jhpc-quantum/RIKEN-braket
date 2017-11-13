#ifndef KET_GATE_PHASE_SHIFT_HPP
# define KET_GATE_PHASE_SHIFT_HPP

# include <boost/config.hpp>

# include <cassert>
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
    inline void conj_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::phase_shift_coeff(first, last, static_cast<Complex>(1)/phase_coefficient, qubit); }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Complex,
      typename StateInteger, typename BitInteger>
    inline void conj_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::phase_shift_coeff(
        parallel_policy, first, last, static_cast<Complex>(1)/phase_coefficient, qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& conj_phase_shift_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::phase_shift_coeff(state, static_cast<Complex>(1)/phase_coefficient, qubit); }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& conj_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        return ::ket::gate::ranges::phase_shift_coeff(
          parallel_policy, state, static_cast<Complex>(1)/phase_coefficient, qubit);
      }
    } // namespace ranges


    template <
      typename RandomAccessIterator, typename Complex,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift_coeff(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::conj_phase_shift_coeff(first, last, phase_coefficient, qubit); }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Complex,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift_coeff(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Complex const& phase_coefficient,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::conj_phase_shift_coeff(
        parallel_policy, first, last, phase_coefficient, qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift_coeff(
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::conj_phase_shift_coeff(state, phase_coefficient, qubit); }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        return ::ket::gate::ranges::conj_phase_shift_coeff(
          parallel_policy, state, phase_coefficient, qubit);
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
        typedef typename boost::range_iterator<RandomAccessRange>::type complex_type;
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
        typedef typename boost::range_iterator<RandomAccessRange>::type complex_type;
        return ::ket::gate::ranges::phase_shift_coeff(
          parallel_policy,
          state, ::ket::utility::exp_i<complex_type>(phase), qubit);
      }
    } // namespace ranges


    template <
      typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void conj_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::phase_shift(first, last, -phase, qubit); }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void conj_phase_shift(
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
      inline RandomAccessRange& conj_phase_shift(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::phase_shift(state, -phase, qubit); }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& conj_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::phase_shift(parallel_policy, state, -phase, qubit); }
    } // namespace ranges


    template <
      typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift(
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::conj_phase_shift(first, last, phase, qubit); }

    template <
      typename ParallelPolicy, typename RandomAccessIterator, typename Real,
      typename StateInteger, typename BitInteger>
    inline void adj_phase_shift(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Real const phase,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::conj_phase_shift(parallel_policy, first, last, phase, qubit); }

    namespace ranges
    {
      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift(
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::conj_phase_shift(state, phase, qubit); }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& state, Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::conj_phase_shift(parallel_policy, state, phase, qubit); }
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

