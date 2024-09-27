#ifndef KET_MEASURE_HPP
# define KET_MEASURE_HPP

# include <cmath>
# include <iterator>
# include <algorithm>

# include <ket/utility/positive_random_value_upto.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  template <typename ParallelPolicy, typename RandomAccessIterator, typename RandomNumberGenerator>
  inline auto measure(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    RandomNumberGenerator& random_number_generator)
  -> typename std::iterator_traits<RandomAccessIterator>::difference_type
  {
    using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    ::ket::utility::transform_inclusive_scan(
      parallel_policy, first, last, first,
      [](complex_type const& lhs, complex_type const& rhs)
      { using std::real; return static_cast<complex_type>(real(lhs) + real(rhs)); },
      [](complex_type const& value)
      { using std::norm; return static_cast<complex_type>(norm(value)); });

    auto const found
      = std::upper_bound(
          first, last,
          static_cast<complex_type>(
            ::ket::utility::positive_random_value_upto(
              real(*std::prev(last)), random_number_generator)),
          [](complex_type const& lhs, complex_type const& rhs)
          { using std::real; return real(lhs) < real(rhs); });
    auto const result = found - first;

    using real_type = ::ket::utility::meta::real_t<complex_type>;
    std::fill(first, last, complex_type{real_type{0}});
    *(first + result) = complex_type{real_type{1}};

    return result;
  }

  template <typename RandomAccessIterator, typename RandomNumberGenerator>
  inline auto measure(
    RandomAccessIterator const first, RandomAccessIterator const last,
    RandomNumberGenerator& random_number_generator)
  -> typename std::iterator_traits<RandomAccessIterator>::difference_type
  { return ::ket::measure(::ket::utility::policy::make_sequential(), first, last, random_number_generator); }


  namespace ranges
  {
    template <typename ParallelPolicy, typename RandomAccessRange, typename RandomNumberGenerator>
    inline auto measure(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& state, RandomNumberGenerator& random_number_generator)
    -> ::ket::utility::meta::range_difference_t<RandomAccessRange>
    {
      using std::begin;
      using std::end;
      return ::ket::measure(parallel_policy, begin(state), end(state), random_number_generator);
    }

    template <typename RandomAccessRange, typename RandomNumberGenerator>
    inline auto measure(RandomAccessRange& state, RandomNumberGenerator& random_number_generator)
    -> ::ket::utility::meta::range_difference_t<RandomAccessRange>
    {
      using std::begin;
      using std::end;
      return ::ket::measure(begin(state), end(state), random_number_generator);
    }
  } // namespace ranges
} // namespace ket


#endif // KET_MEASURE_HPP
