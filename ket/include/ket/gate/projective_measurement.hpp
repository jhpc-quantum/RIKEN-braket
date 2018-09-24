#ifndef KET_GATE_PROJECTIVE_MEASUREMENT_HPP
# define KET_GATE_PROJECTIVE_MEASUREMENT_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cmath>
# include <complex>
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

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/positive_random_value_upto.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
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


namespace ket
{
  namespace gate
  {
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
    enum class outcome : int { unspecified = -1, zero = 0, one = 1 };

#   define KET_GATE_OUTCOME_TYPE ket::gate::outcome
#   define KET_GATE_OUTCOME_VALUE(value) ket::gate::outcome::value
# else // BOOST_NO_CXX11_SCOPED_ENUMS
    namespace outcome_ { enum outcome { unspecified = -1, zero = 0, one = 1 }; }

#   define KET_GATE_OUTCOME_TYPE ket::gate::outcome_::outcome
#   define KET_GATE_OUTCOME_VALUE(value) ket::gate::outcome_::value
# endif // BOOST_NO_CXX11_SCOPED_ENUMS

    namespace projective_measurement_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <typename RandomAccessIterator, typename StateInteger>
      struct zero_one_probabilities_loop_inside
      {
        long double& zero_probability_;
        long double& one_probability_;
        RandomAccessIterator first_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;

        zero_one_probabilities_loop_inside(
          long double& zero_probability,
          long double& one_probability,
          RandomAccessIterator const first,
          StateInteger const qubit_mask,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask)
          : zero_probability_(zero_probability),
            one_probability_(one_probability),
            first_(first),
            qubit_mask_(qubit_mask),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask)
        { }

        void operator()(StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          StateInteger const zero_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_);
          // xxxxx1xxxxxx
          StateInteger const one_index = zero_index bitor qubit_mask_;

          using std::norm;
          zero_probability_ += static_cast<long double>(norm(*(first_+zero_index)));
          one_probability_ += static_cast<long double>(norm(*(first_+one_index)));
        }
      };

      template <typename RandomAccessIterator, typename StateInteger>
      inline zero_one_probabilities_loop_inside<RandomAccessIterator, StateInteger>
      make_zero_one_probabilities_loop_inside(
        long double& zero_probability,
        long double& one_probability,
        RandomAccessIterator const first,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask)
      {
        return zero_one_probabilities_loop_inside<RandomAccessIterator, StateInteger>(
          zero_probability, one_probability, first,
          qubit_mask, lower_bits_mask, upper_bits_mask);
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
      std::pair<
        typename ::ket::utility::meta::real_of<
          typename std::iterator_traits<RandomAccessIterator>::value_type>::type,
        typename ::ket::utility::meta::real_of<
          typename std::iterator_traits<RandomAccessIterator>::value_type>::type>
      zero_one_probabilities(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(qubit);
        StateInteger const lower_bits_mask
          = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        long double zero_probability = 0.0l;
        long double one_probability = 0.0l;

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          [&zero_probability, &one_probability, first,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            StateInteger const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            StateInteger const one_index = zero_index bitor qubit_mask;

            using std::norm;
            zero_probability += static_cast<long double>(norm(*(first+zero_index)));
            one_probability += static_cast<long double>(norm(*(first+one_index)));
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::projective_measurement_detail::make_zero_one_probabilities_loop_inside(
            zero_probability, one_probability, first,
            qubit_mask, lower_bits_mask, upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS

        typedef
          typename std::iterator_traits<RandomAccessIterator>::value_type
          complex_type;
        typedef
          typename ::ket::utility::meta::real_of<complex_type>::type real_type;
        return std::make_pair(
          static_cast<real_type>(zero_probability), static_cast<real_type>(one_probability));
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
          (KET_is_same<
             typename ::ket::utility::meta::real_of<
               typename std::iterator_traits<RandomAccessIterator>::value_type>::type,
             Real>::value),
          "Real must be the same as real number type corresponding to value type of iterator");

        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(qubit);
        StateInteger const lower_bits_mask
          = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        // a' = a/sqrt(p_0)
        // If p = p_0 + p_1 != 1 because of numerical error, a' = (a / sqrt(p)) / sqrt(p_0/p) = a/sqrt(p_0)
        using std::pow;
        using boost::math::constants::half;
        Real const multiplier = pow(zero_probability, -half<Real>());

        using ::ket::utility::loop_n;
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
          (KET_is_same<
             typename ::ket::utility::meta::real_of<
               typename std::iterator_traits<RandomAccessIterator>::value_type>::type,
             Real>::value),
          "Real must be the same as real number type corresponding to value type of iterator");

        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(qubit);
        StateInteger const lower_bits_mask
          = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        // a' = a/sqrt(p_1)
        // If p = p_0 + p_1 != 1 because of numerical error, a' = (a / sqrt(p)) / sqrt(p_1/p) = a/sqrt(p_1)
        using std::pow;
        using boost::math::constants::half;
        Real const multiplier = pow(one_probability, -half<Real>());

        using ::ket::utility::loop_n;
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
      inline KET_GATE_OUTCOME_TYPE projective_measurement(
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
        std::pair<real_type, real_type> const zero_one_probabilities
          = ::ket::gate::projective_measurement_detail::zero_one_probabilities(
              parallel_policy, first, last, qubit);
        real_type const total_probability = zero_one_probabilities.first + zero_one_probabilities.second;

        if (::ket::utility::positive_random_value_upto(total_probability, random_number_generator)
            < zero_one_probabilities.first)
        {
          ::ket::gate::projective_measurement_detail::change_state_after_measuring_zero(
            parallel_policy, first, last, qubit, zero_one_probabilities.first);
          return KET_GATE_OUTCOME_VALUE(zero);
        }

        ::ket::gate::projective_measurement_detail::change_state_after_measuring_one(
          parallel_policy, first, last, qubit, zero_one_probabilities.second);
        return KET_GATE_OUTCOME_VALUE(one);
      }
    } // namespace projective_measurement_detail

    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
    inline KET_GATE_OUTCOME_TYPE projective_measurement(
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
    inline KET_GATE_OUTCOME_TYPE projective_measurement(
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
      inline KET_GATE_OUTCOME_TYPE projective_measurement(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator)
      {
        return ::ket::gate::projective_measurement_detail::projective_measurement(
          ::ket::utility::policy::make_sequential(),
          ::ket::utility::begin(state), ::ket::utility::end(state), qubit, random_number_generator);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
      inline KET_GATE_OUTCOME_TYPE projective_measurement(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator)
      {
        return ::ket::gate::projective_measurement_detail::projective_measurement(
          parallel_policy,
          ::ket::utility::begin(state), ::ket::utility::end(state), qubit, random_number_generator);
      }
    } // namespace ranges
  } // namespace gate
} // namespace ket


# undef KET_is_unsigned
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

