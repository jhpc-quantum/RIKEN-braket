#ifndef KET_GENERATE_EVENTS_HPP
# define KET_GENERATE_EVENTS_HPP

# include <cmath>
# include <vector>
# include <iterator>
# include <algorithm>

# include <ket/utility/positive_random_value_upto.hpp>


namespace ket
{
  template <
    typename ParallelPolicy,
    typename StateInteger, typename Allocator,
    typename RandomAccessIterator, typename RandomNumberGenerator>
  inline auto generate_events(
    ParallelPolicy const parallel_policy,
    std::vector<StateInteger, Allocator>& result,
    RandomAccessIterator const first, RandomAccessIterator const last,
    int const num_events, RandomNumberGenerator& random_number_generator)
  -> void
  {
    result.clear();
    result.reserve(num_events);

    using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    ::ket::utility::transform_inclusive_scan(
      parallel_policy, first, last, first,
      [](complex_type const& lhs, complex_type const& rhs)
      { using std::real; return static_cast<complex_type>(real(lhs) + real(rhs)); },
      [](complex_type const& value)
      { using std::norm; return static_cast<complex_type>(norm(value)); });

    using std::real;
    auto const total_probability = real(*std::prev(last));

    for (auto event_index = int{0}; event_index < num_events; ++event_index)
    {
      RandomAccessIterator const found
        = std::upper_bound(
            first, last,
            static_cast<complex_type>(
              ::ket::utility::positive_random_value_upto(total_probability, random_number_generator)),
            [](complex_type const& lhs, complex_type const& rhs)
            { return real(lhs) < real(rhs); });

      result.push_back(found - first);
    }
  }

  template <
    typename ParallelPolicy,
    typename StateInteger, typename Allocator,
    typename RandomAccessIterator, typename RandomNumberGenerator>
  inline auto generate_events(
    ParallelPolicy const parallel_policy,
    std::vector<StateInteger, Allocator>& result,
    RandomAccessIterator const first, RandomAccessIterator const last,
    int const num_events, RandomNumberGenerator const&,
    typename RandomNumberGenerator::result_type const seed)
  -> void
  {
    RandomNumberGenerator random_number_generator(seed);
    ::ket::generate_events(parallel_policy, result, first, last, num_events, random_number_generator);
  }

  template <
    typename StateInteger, typename Allocator,
    typename RandomAccessIterator, typename RandomNumberGenerator>
  inline auto generate_events(
    std::vector<StateInteger, Allocator>& result,
    RandomAccessIterator const first, RandomAccessIterator const last,
    int const num_events, RandomNumberGenerator const& random_number_generator)
  -> void
  {
    ::ket::generate_events(
      ::ket::utility::policy::make_sequential(),
      result, first, last, num_events, random_number_generator);
  }

  template <
    typename StateInteger, typename Allocator,
    typename RandomAccessIterator, typename RandomNumberGenerator>
  inline auto generate_events(
    std::vector<StateInteger, Allocator>& result,
    RandomAccessIterator const first, RandomAccessIterator const last,
    int const num_events, RandomNumberGenerator const&,
    typename RandomNumberGenerator::result_type const seed)
  -> void
  {
    RandomNumberGenerator random_number_generator(seed);
    ::ket::generate_events(
      ::ket::utility::policy::make_sequential(),
      result, first, last, num_events, random_number_generator);
  }


  namespace ranges
  {
    template <typename ParallelPolicy, typename StateInteger, typename Allocator, typename RandomAccessRange, typename RandomNumberGenerator>
    inline auto generate_events(
      ParallelPolicy const parallel_policy,
      std::vector<StateInteger, Allocator>& result,
      RandomAccessRange& state,
      int const num_events, RandomNumberGenerator& random_number_generator)
    -> void
    {
      using std::begin;
      using std::end;
      ::ket::generate_events(parallel_policy, result, begin(state), end(state), num_events, random_number_generator);
    }

    template <typename ParallelPolicy, typename StateInteger, typename Allocator, typename RandomAccessRange, typename RandomNumberGenerator>
    inline auto generate_events(
      ParallelPolicy const parallel_policy,
      std::vector<StateInteger, Allocator>& result,
      RandomAccessRange& state,
      int const num_events, RandomNumberGenerator const& random_number_generator,
      typename RandomNumberGenerator::result_type const seed)
    -> void
    {
      using std::begin;
      using std::end;
      ::ket::generate_events(parallel_policy, result, begin(state), end(state), num_events, random_number_generator, seed);
    }

    template <typename StateInteger, typename Allocator, typename RandomAccessRange, typename RandomNumberGenerator>
    inline auto generate_events(
      std::vector<StateInteger, Allocator>& result,
      RandomAccessRange& state,
      int const num_events, RandomNumberGenerator& random_number_generator)
    -> void
    {
      using std::begin;
      using std::end;
      ::ket::generate_events(result, begin(state), end(state), num_events, random_number_generator);
    }

    template <typename StateInteger, typename Allocator, typename RandomAccessRange, typename RandomNumberGenerator>
    inline auto generate_events(
      std::vector<StateInteger, Allocator>& result,
      RandomAccessRange& state,
      int const num_events, RandomNumberGenerator const& random_number_generator,
      typename RandomNumberGenerator::result_type const seed)
    -> void
    {
      using std::begin;
      using std::end;
      ::ket::generate_events(result, begin(state), end(state), num_events, random_number_generator, seed);
    }
  } // namespace ranges
} // namespace ket


#endif // KET_GENERATE_EVENTS_HPP
