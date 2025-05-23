#ifndef KET_GATE_CLEAR_HPP
# define KET_GATE_CLEAR_HPP

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
    // CLEAR_i
    // CLEAR_1 (a_{0} |0> + a_{1} |1>) = |0>
    namespace clear_detail
    {
      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger>
      inline auto zero_probability(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>
      {
        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        using real_type = ::ket::utility::meta::real_t<complex_type>;
        auto zero_probabilities = std::vector<long double>(::ket::utility::num_threads(parallel_policy), real_type{0});

        ::ket::utility::loop_n(
          parallel_policy, static_cast<StateInteger>(last - first) / 2u,
          [&zero_probabilities, first, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const thread_index)
          {
            // xxxxx0xxxxxx
            auto const zero_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask);

            using std::norm;
            zero_probabilities[thread_index] += static_cast<long double>(norm(*(first + zero_index)));
          });

        using std::begin;
        using std::end;
        return static_cast<real_type>(std::accumulate(begin(zero_probabilities), end(zero_probabilities), 0.0l));
      }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename Real, typename StateInteger, typename BitInteger>
      inline auto do_clear(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        Real const multiplier, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
          == static_cast<StateInteger>(last - first));

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        ::ket::utility::loop_n(
          parallel_policy, static_cast<StateInteger>(last - first)/2u,
          [first, multiplier, qubit_mask, lower_bits_mask, upper_bits_mask](StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            auto const zero_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            auto const one_index = zero_index bitor qubit_mask;
            *(first + zero_index) *= multiplier;
            *(first + one_index) = complex_type{0};
          });
      }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto clear(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
          == static_cast<StateInteger>(last - first));

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        using real_type = ::ket::utility::meta::real_t<complex_type>;
        using std::pow;
        using boost::math::constants::half;
        ::ket::gate::clear_detail::do_clear(
          parallel_policy, first, last,
          pow(::ket::gate::clear_detail::zero_probability(parallel_policy, first, last, qubit), -half<real_type>()), qubit);
      }
    } // namespace clear_detail

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto clear(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> void
    { ::ket::gate::clear_detail::clear(parallel_policy, first, last, qubit); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto clear(RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit) -> void
    { ::ket::gate::clear(::ket::utility::policy::make_sequential(), first, last, qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline auto clear(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::clear(parallel_policy, begin(state), end(state), qubit);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline auto clear(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit) -> RandomAccessRange&
      { return ::ket::gate::ranges::clear(::ket::utility::policy::make_sequential(), state, qubit); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_CLEAR_HPP
