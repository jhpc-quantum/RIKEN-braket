#ifndef KET_GATE_PROJECTIVE_MEASUREMENT_HPP
# define KET_GATE_PROJECTIVE_MEASUREMENT_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cmath>
# include <complex>
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
# endif
# include <iterator>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_RANDOM
#   include <random>
# else
#   include <boost/random/uniform_real_distribution.hpp>
# endif
# ifndef NDEBUG
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     include <type_traits>
#   else
#     include <boost/type_traits/is_unsigned.hpp>
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

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/meta/real_of.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_is_unsigned std::is_unsigned
# else
#   define KET_is_unsigned boost::is_unsigned
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif

# ifndef BOOST_NO_CXX11_HDR_RANDOM
#   define KET_uniform_real_distribution std::uniform_real_distribution
# else
#   define KET_uniform_real_distribution boost::random::uniform_real_distribution
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
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
    enum class outcome : int { unspecified = -1, zero = 0, one = 1 };
# else // BOOST_NO_CXX11_SCOPED_ENUMS
    namespace outcome_ { enum outcome { unspecified = -1, zero = 0, one = 1 }; }
# endif // BOOST_NO_CXX11_SCOPED_ENUMS

    namespace projective_measurement_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <typename Real, typename RandomAccessIterator, typename StateInteger>
      struct zero_probability_loop_inside
      {
        Real& zero_probability_;
        RandomAccessIterator first_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;

        zero_probability_loop_inside(
          Real& zero_probability,
          RandomAccessIterator const first,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask)
          : zero_probability_(zero_probability),
            first_(first),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask)
        { }

        void operator()(StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          StateInteger const zero_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_);
          using std::norm;
          zero_probability_ += norm(*(first_+zero_index));
        }
      };

      template <typename Real, typename RandomAccessIterator, typename StateInteger>
      inline zero_probability_loop_inside<Real, RandomAccessIterator, StateInteger>
      make_zero_probability_loop_inside(
        Real& zero_probability,
        RandomAccessIterator const first,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask)
      {
        return zero_probability_loop_inside<Real, RandomAccessIterator, StateInteger>(
          zero_probability, first, lower_bits_mask, upper_bits_mask);
      }


      template <typename RandomAccessIterator, typename StateInteger, typename Real>
      struct change_state_after_measuring_zero_loop_inside
      {
        RandomAccessIterator first_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;
        Real multiplier_;

        change_state_after_measuring_zero_loop_inside(
          RandomAccessIterator const first,
          StateInteger const qubit_mask,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask,
          Real const multiplier)
          : first_(first),
            qubit_mask_(qubit_mask),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask),
            multiplier_(multiplier)
        { }

        void operator()(StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          StateInteger const zero_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_);
          // xxxxx1xxxxxx
          StateInteger const one_index = zero_index bitor qubit_mask_;

          typedef
            typename std::iterator_traits<RandomAccessIterator>::value_type
            complex_type;
          *(first_+zero_index) *= multiplier_;
          *(first_+one_index) = static_cast<complex_type>(static_cast<Real>(0));
        }
      };

      template <typename RandomAccessIterator, typename StateInteger, typename Real>
      inline change_state_after_measuring_zero_loop_inside<RandomAccessIterator, StateInteger, Real>
      make_change_state_after_measuring_zero_loop_inside(
        RandomAccessIterator const first,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask,
        Real const multiplier)
      {
        return change_state_after_measuring_zero_loop_inside<RandomAccessIterator, StateInteger, Real>(
          first, qubit_mask, lower_bits_mask, upper_bits_mask, multiplier);
      }


      template <typename RandomAccessIterator, typename StateInteger, typename Real>
      struct change_state_after_measuring_one_loop_inside
      {
        RandomAccessIterator first_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;
        Real multiplier_;

        change_state_after_measuring_one_loop_inside(
          RandomAccessIterator const first,
          StateInteger const qubit_mask,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask,
          Real const multiplier)
          : first_(first),
            qubit_mask_(qubit_mask),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask),
            multiplier_(multiplier)
        { }

        void operator()(StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          StateInteger const zero_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_);
          // xxxxx1xxxxxx
          StateInteger const one_index = zero_index bitor qubit_mask_;

          typedef
            typename std::iterator_traits<RandomAccessIterator>::value_type
            complex_type;
          *(first_+zero_index) = static_cast<complex_type>(static_cast<Real>(0));
          *(first_+one_index) *= multiplier_;
        }
      };

      template <typename RandomAccessIterator, typename StateInteger, typename Real>
      inline change_state_after_measuring_one_loop_inside<RandomAccessIterator, StateInteger, Real>
      make_change_state_after_measuring_one_loop_inside(
        RandomAccessIterator const first,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask,
        Real const multiplier)
      {
        return change_state_after_measuring_one_loop_inside<RandomAccessIterator, StateInteger, Real>(
          first, qubit_mask, lower_bits_mask, upper_bits_mask, multiplier);
      }
# endif // BOOST_NO_CXX11_LAMBDAS


      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger>
      inline
      typename ::ket::utility::meta::real_of<
        typename std::iterator_traits<RandomAccessIterator>::value_type>::type
      zero_probability(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(qubit);
        StateInteger const lower_bits_mask
          = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        typedef
          typename std::iterator_traits<RandomAccessIterator>::value_type
          complex_type;
        typedef
          typename ::ket::utility::meta::real_of<complex_type>::type real_type;
        real_type result = static_cast<real_type>(0);

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          [&result, first, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            StateInteger const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            using std::norm;
            result += norm(*(first+zero_index));
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::projective_measurement_detail::make_zero_probability_loop_inside(
            result, first, lower_bits_mask, upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS

        return result;
      }


      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger, typename Real>
      inline void change_state_after_measuring_zero(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Real const zero_probability)
      {
        static_assert(
          KET_is_same<
            typename ::ket::utility::meta::real_of<
              typename std::iterator_traits<RandomAccesIterator>::value_type>::type,
            Real>::value,
          "Real must be the same as real number type corresponding to value type of iterator");

        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(qubit);
        StateInteger const lower_bits_mask
          = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        using std::pow;
        using boost::math::constants::half;
        Real const multiplier = pow(zero_probability, -half<real_type>());

# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          [first, qubit_mask, lower_bits_mask, upper_bits_mask, multiplier](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            StateInteger const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            StateInteger const one_index = zero_index bitor qubit_mask;

            typedef
              typename std::iterator_traits<RandomAccessIterator>::value_type
              complex_type;
            *(first+zero_index) *= multiplier;
            *(first+one_index) = static_cast<complex_type>(static_cast<Real>(0));
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::projective_measurement_detail::make_change_state_after_measuring_zero_loop_inside(
            first, qubit_mask, lower_bits_mask, upper_bits_mask, multiplier));
# endif // BOOST_NO_CXX11_LAMBDAS
      }


      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger, typename Real>
      inline void change_state_after_measuring_one(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Real const one_probability)
      {
        static_assert(
          KET_is_same<
            typename ::ket::utility::meta::real_of<
              typename std::iterator_traits<RandomAccesIterator>::value_type>::type,
            Real>::value,
          "Real must be the same as real number type corresponding to value type of iterator");

        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(qubit);
        StateInteger const lower_bits_mask
          = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        using std::pow;
        using boost::math::constants::half;
        real_type const multiplier = pow(one_probability, -half<Real>());

# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          [first, qubit_mask, lower_bits_mask, upper_bits_mask, multiplier](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            StateInteger const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            StateInteger const one_index = zero_index bitor qubit_mask;

            typedef
              typename std::iterator_traits<RandomAccessIterator>::value_type
              complex_type;
            *(first+zero_index) = static_cast<complex_type>(static_cast<Real>(0));
            *(first+one_index) *= multiplier;
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::projective_measurement_detail::make_change_state_after_measuring_one_loop_inside(
            first, qubit_mask, lower_bits_mask, upper_bits_mask, multiplier));
# endif // BOOST_NO_CXX11_LAMBDAS
      }


      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
      inline
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
      ::ket::gate::outcome
# else // BOOST_NO_CXX11_SCOPED_ENUMS
      ::ket::gate::outcome_::outcome
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
      projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator)
      {
        static_assert(
          KET_is_unsigned<StateInteger>::value,
          "StateInteger should be unsigned");
        static_assert(
          KET_is_unsigned<BitInteger>::value,
          "BitInteger should be unsigned");
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

        typedef
          typename std::iterator_traits<RandomAccessIterator>::value_type
          complex_type;
        typedef
          typename ::ket::utility::meta::real_of<complex_type>::type real_type;
        real_type const zero_probability
          = ::ket::gate::projective_measurement_detail::zero_probability(
              parallel_policy, first, last, qubit);

        KET_uniform_real_distribution<double> distribution(0.0, 1.0);
        if (distribution(random_number_generator) < static_cast<double>(zero_probability))
        {
          ::ket::gate::projective_measurement_detail::change_state_after_measuring_zero(
            parallel_policy, first, last, qubit, zero_probability);
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
          return ::ket::gate::outcome::zero;
# else // BOOST_NO_CXX11_SCOPED_ENUMS
          return ::ket::gate::outcome_::zero;
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
        }

        ::ket::gate::projective_measurement_detail::change_state_after_measuring_one(
          parallel_policy, first, last, qubit, static_cast<real_type>(1)-zero_probability);
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
        return ::ket::gate::outcome::one;
# else // BOOST_NO_CXX11_SCOPED_ENUMS
        return ::ket::gate::outcome_::one;
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
      }
    } // namespace projective_measurement_detail

    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
    inline
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
    ::ket::gate::outcome
# else // BOOST_NO_CXX11_SCOPED_ENUMS
    ::ket::gate::outcome_::outcome
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
    projective_measurement(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      RandomNumberGenerator& random_number_generator)
    {
      return ::ket::gate::projective_measurement_detail::projective_measurement(
        ::ket::utility::policy::make_sequential(),
        first, last, qubit, random_number_generator);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
    inline
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
    ::ket::gate::outcome
# else // BOOST_NO_CXX11_SCOPED_ENUMS
    ::ket::gate::outcome_::outcome
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
    projective_measurement(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      RandomNumberGenerator& random_number_generator)
    {
      return ::ket::gate::projective_measurement_detail::projective_measurement(
        parallel_policy, first, last, qubit, random_number_generator);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
      inline
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
      ::ket::gate::outcome
# else // BOOST_NO_CXX11_SCOPED_ENUMS
      ::ket::gate::outcome_::outcome
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
      projective_measurement(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator)
      {
        return ::ket::gate::projective_measurement_detail::projective_measurement(
          ::ket::utility::policy::make_sequential(),
          boost::begin(state), boost::end(state), qubit, random_number_generator);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
      inline
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
      ::ket::gate::outcome
# else // BOOST_NO_CXX11_SCOPED_ENUMS
      ::ket::gate::outcome_::outcome
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
      projective_measurement(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator)
      {
        return ::ket::gate::projective_measurement_detail::projective_measurement(
          parallel_policy,
          boost::begin(state), boost::end(state), qubit, random_number_generator);
      }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
      inline
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
      ::ket::gate::outcome
# else // BOOST_NO_CXX11_SCOPED_ENUMS
      ::ket::gate::outcome_::outcome
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
      projective_measurement(
        std::vector<Complex, Allocator>& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator)
      {
        return ::ket::gate::projective_measurement_detail::projective_measurement(
          ::ket::utility::policy::make_sequential(),
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          qubit, random_number_generator);
      }

      template <
        typename ParallelPolicy,
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
      inline
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
      ::ket::gate::outcome
# else // BOOST_NO_CXX11_SCOPED_ENUMS
      ::ket::gate::outcome_::outcome
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
      projective_measurement(
        ParallelPolicy const parallel_policy,
        std::vector<Complex, Allocator>& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator)
      {
        return ::ket::gate::projective_measurement_detail::projective_measurement(
          parallel_policy,
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          qubit, random_number_generator);
      }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    } // namespace ranges
  } // namespace gate
} // namespace ket


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif
# undef KET_uniform_real_distribution
# undef KET_is_unsigned
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

