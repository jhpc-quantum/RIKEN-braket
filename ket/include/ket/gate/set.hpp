#ifndef KET_GATE_SET_HPP
# define KET_GATE_SET_HPP

# include <cassert>
# include <cmath>
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
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace set_detail
    {
      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline void set(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
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

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
        auto one_probabilities = std::vector<real_type>(::ket::utility::num_threads(parallel_policy), real_type{0});

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last - first) / 2u,
          [&one_probabilities, first, qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const thread_index)
          {
            // xxxxx0xxxxxx
            auto const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            auto const one_index = zero_index bitor qubit_mask;
            *(first + zero_index) = complex_type{0};

            using std::norm;
            one_probabilities[thread_index] += norm(*(first + one_index));
          });

        using std::pow;
        using boost::math::constants::half;
        auto const multiplier = pow(std::accumulate(std::begin(one_probabilities), std::end(one_probabilities), real_type{0}), -half<real_type>());

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
            *(first + one_index) *= multiplier;
          });
      }
    } // namespace set_detail

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void set(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::set_detail::set(parallel_policy, first, last, qubit); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void set(
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::set(::ket::utility::policy::make_sequential(), first, last, qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& set(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::set(parallel_policy, std::begin(state), std::end(state), qubit);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& set(
        RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::set(::ket::utility::policy::make_sequential(), state, qubit); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_SET_HPP
