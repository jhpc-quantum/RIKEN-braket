#ifndef KET_GATE_PROJECTIVE_MEASUREMENT_HPP
# define KET_GATE_PROJECTIVE_MEASUREMENT_HPP

# include <cassert>
# include <cmath>
# include <complex>
# include <vector>
# include <iterator>
# include <numeric>
# include <utility>
# include <type_traits>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/positive_random_value_upto.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    enum class outcome : int { unspecified = -1, zero = 0, one = 1 };

    namespace projective_measurement_detail
    {
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
        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        auto zero_probabilities = std::vector<long double>(::ket::utility::num_threads(parallel_policy), 0.0l);
        auto one_probabilities = std::vector<long double>(::ket::utility::num_threads(parallel_policy), 0.0l);

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last - first) / 2u,
          [&zero_probabilities, &one_probabilities, first,
           qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const thread_index)
          {
            // xxxxx0xxxxxx
            auto const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            auto const one_index = zero_index bitor qubit_mask;

            using std::norm;
            zero_probabilities[thread_index] += static_cast<long double>(norm(*(first + zero_index)));
            one_probabilities[thread_index] += static_cast<long double>(norm(*(first + one_index)));
          });

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
        return std::make_pair(
          static_cast<real_type>(std::accumulate(std::begin(zero_probabilities), std::end(zero_probabilities), 0.0l)),
          static_cast<real_type>(std::accumulate(std::begin(one_probabilities), std::end(one_probabilities), 0.0l)));
      }

      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger, typename Real>
      inline void change_state_after_measuring_zero(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Real const zero_probability)
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          (std::is_same<typename ::ket::utility::meta::real_of<complex_type>::type, Real>::value),
          "Real must be the same as real number type corresponding to value type of iterator");

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        // a' = a/sqrt(p_0)
        // If p = p_0 + p_1 != 1 because of numerical error, a' = (a / sqrt(p)) / sqrt(p_0/p) = a/sqrt(p_0)
        using std::pow;
        using boost::math::constants::half;
        auto const multiplier = pow(zero_probability, -half<Real>());

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last - first) / 2u,
          [first, qubit_mask, lower_bits_mask, upper_bits_mask, multiplier](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            auto const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            auto const one_index = zero_index bitor qubit_mask;

            *(first + zero_index) *= multiplier;
            *(first + one_index) = complex_type{Real{0}};
          });
      }

      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger, typename Real>
      inline void change_state_after_measuring_one(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Real const one_probability)
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          (std::is_same<typename ::ket::utility::meta::real_of<complex_type>::type, Real>::value),
          "Real must be the same as real number type corresponding to value type of iterator");

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        // a' = a/sqrt(p_1)
        // If p = p_0 + p_1 != 1 because of numerical error, a' = (a / sqrt(p)) / sqrt(p_1/p) = a/sqrt(p_1)
        using std::pow;
        using boost::math::constants::half;
        auto const multiplier = pow(one_probability, -half<Real>());

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last - first) / 2u,
          [first, qubit_mask, lower_bits_mask, upper_bits_mask, multiplier](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            auto const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            auto const one_index = zero_index bitor qubit_mask;

            *(first + zero_index) = complex_type{Real{0}};
            *(first + one_index) *= multiplier;
          });
      }

      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
      inline ::ket::gate::outcome projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator)
      {
        static_assert(
          std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
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

        auto const zero_one_probabilities
          = ::ket::gate::projective_measurement_detail::zero_one_probabilities(
              parallel_policy, first, last, qubit);
        auto const total_probability = zero_one_probabilities.first + zero_one_probabilities.second;

        if (::ket::utility::positive_random_value_upto(total_probability, random_number_generator)
            < zero_one_probabilities.first)
        {
          ::ket::gate::projective_measurement_detail::change_state_after_measuring_zero(
            parallel_policy, first, last, qubit, zero_one_probabilities.first);
          return ::ket::gate::outcome::zero;
        }

        ::ket::gate::projective_measurement_detail::change_state_after_measuring_one(
          parallel_policy, first, last, qubit, zero_one_probabilities.second);
        return ::ket::gate::outcome::one;
      }
    } // namespace projective_measurement_detail

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
    inline ::ket::gate::outcome projective_measurement(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, RandomNumberGenerator& random_number_generator)
    { return ::ket::gate::projective_measurement_detail::projective_measurement(parallel_policy, first, last, qubit, random_number_generator); }

    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
    inline ::ket::gate::outcome projective_measurement(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, RandomNumberGenerator& random_number_generator)
    { return ::ket::gate::projective_measurement(::ket::utility::policy::make_sequential(), first, last, qubit, random_number_generator); }

    namespace ranges
    {
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
      inline ::ket::gate::outcome projective_measurement(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit, RandomNumberGenerator& random_number_generator)
      { return ::ket::gate::projective_measurement(parallel_policy, std::begin(state), std::end(state), qubit, random_number_generator); }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator>
      inline ::ket::gate::outcome projective_measurement(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit, RandomNumberGenerator& random_number_generator)
      { return ::ket::gate::ranges::projective_measurement(::ket::utility::policy::make_sequential(), state, qubit, random_number_generator); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_PROJECTIVE_MEASUREMENT_HPP
