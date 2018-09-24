#ifndef KET_GENERATE_EVENTS_HPP
#define KET_GENERATE_EVENTS_HPP

# include <boost/config.hpp>

# include <cmath>
# include <vector>
# include <iterator>
# include <algorithm>

# include <boost/utility.hpp>

# include <ket/utility/loop_n.hpp>
# include <ket/utility/positive_random_value_upto.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace generate_events_detail
  {
# ifdef BOOST_NO_CXX11_LAMBDAS
    template <typename Complex>
    struct real_part_plus
    {
      typedef Complex result_type;

      Complex operator()(Complex const& lhs, Complex const& rhs) const
      { using std::real; return static_cast<Complex>(real(lhs) + real(rhs)); }
    };

    template <typename Complex>
    struct complex_norm
    {
      typedef Complex result_type;

      Complex operator()(Complex const& value) const
      { using std::norm; return static_cast<Complex>(norm(value)); }
    };

    template <typename Complex>
    struct real_part_less_than
    {
      typedef bool result_type;

      bool operator()(Complex const& lhs, Complex const& rhs) const
      { using std::real; return real(lhs) < real(rhs); }
    };
# endif
  }


  template <
    typename ParallelPolicy,
    typename StateInteger, typename Allocator,
    typename RandomAccessIterator, typename RandomNumberGenerator>
  inline void generate_events(
    ParallelPolicy const parallel_policy,
    std::vector<StateInteger, Allocator>& result,
    RandomAccessIterator const first, RandomAccessIterator const last,
    int const num_events,
    RandomNumberGenerator& random_number_generator)
  {
    result.clear();
    result.reserve(num_events);

    typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
# ifndef BOOST_NO_CXX11_LAMBDAS
    ::ket::utility::transform_inclusive_scan(
      parallel_policy, first, last, first,
      [](complex_type const& lhs, complex_type const& rhs)
      { using std::real; return static_cast<complex_type>(real(lhs) + real(rhs)); },
      [](complex_type const& value)
      { using std::norm; return static_cast<complex_type>(norm(value)); });
# else // BOOST_NO_CXX11_LAMBDAS
    ::ket::utility::transform_inclusive_scan(
      parallel_policy, first, last, first,
      ::ket::generate_events_detail::real_part_plus<complex_type>(),
      ::ket::generate_events_detail::complex_norm<complex_type>());
# endif // BOOST_NO_CXX11_LAMBDAS

    using std::real;
    typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
    real_type const total_probability = real(*boost::prior(last));

    for (int event_index = 0; event_index < num_events; ++event_index)
    {
# ifndef BOOST_NO_CXX11_LAMBDAS
      RandomAccessIterator const found
        = std::upper_bound(
            first, last,
            static_cast<complex_type>(
              ::ket::utility::positive_random_value_upto(
                total_probability, random_number_generator)),
            [](complex_type const& lhs, complex_type const& rhs)
            { return real(lhs) < real(rhs); });
# else // BOOST_NO_CXX11_LAMBDAS
      RandomAccessIterator const found
        = std::upper_bound(
            first, last,
            static_cast<complex_type>(
              ::ket::utility::positive_random_value_upto(
                total_probability, random_number_generator)),
            ::ket::generate_events_detail::real_part_less_than<complex_type>());
# endif // BOOST_NO_CXX11_LAMBDAS

      result.push_back(found - first);
    }
  }

  template <
    typename ParallelPolicy,
    typename StateInteger, typename Allocator,
    typename RandomAccessIterator, typename RandomNumberGenerator>
  inline void generate_events(
    ParallelPolicy const parallel_policy,
    std::vector<StateInteger, Allocator>& result,
    RandomAccessIterator const first, RandomAccessIterator const last,
    int const num_events,
    RandomNumberGenerator const&,
    typename RandomNumberGenerator::result_type const seed)
  {
    RandomNumberGenerator random_number_generator(seed);
    ::ket::generate_events(
      parallel_policy,
      result, first, last, num_events, random_number_generator);
  }

  template <
    typename StateInteger, typename Allocator,
    typename RandomAccessIterator, typename RandomNumberGenerator>
  inline void generate_events(
    std::vector<StateInteger, Allocator>& result,
    RandomAccessIterator const first, RandomAccessIterator const last,
    int const num_events,
    RandomNumberGenerator const& random_number_generator)
  {
    ::ket::generate_events(
      ::ket::utility::policy::make_sequential(),
      result, first, last, num_events, random_number_generator);
  }

  template <
    typename StateInteger, typename Allocator,
    typename RandomAccessIterator, typename RandomNumberGenerator>
  inline void generate_events(
    std::vector<StateInteger, Allocator>& result,
    RandomAccessIterator const first, RandomAccessIterator const last,
    int const num_events,
    RandomNumberGenerator const&,
    typename RandomNumberGenerator::result_type const seed)
  {
    RandomNumberGenerator random_number_generator(seed);
    ::ket::generate_events(
      ::ket::utility::policy::make_sequential(),
      result, first, last, num_events, random_number_generator);
  }


  namespace ranges
  {
    template <typename ParallelPolicy, typename StateInteger, typename Allocator, typename RandomAccessRange, typename RandomNumberGenerator>
    inline void generate_events(
      ParallelPolicy const parallel_policy,
      std::vector<StateInteger, Allocator>& result,
      RandomAccessRange& state,
      int const num_events,
      RandomNumberGenerator& random_number_generator)
    {
      ::ket::generate_events(
        parallel_policy,
        result, ::ket::utility::begin(state), ::ket::utility::end(state), num_events, random_number_generator);
    }

    template <typename ParallelPolicy, typename StateInteger, typename Allocator, typename RandomAccessRange, typename RandomNumberGenerator>
    inline void generate_events(
      ParallelPolicy const parallel_policy,
      std::vector<StateInteger, Allocator>& result,
      RandomAccessRange& state,
      int const num_events,
      RandomNumberGenerator const& random_number_generator,
      typename RandomNumberGenerator::result_type const seed)
    {
      ::ket::generate_events(
        parallel_policy,
        result, ::ket::utility::begin(state), ::ket::utility::end(state),
        num_events, random_number_generator, seed);
    }

    template <typename StateInteger, typename Allocator, typename RandomAccessRange, typename RandomNumberGenerator>
    inline void generate_events(
      std::vector<StateInteger, Allocator>& result,
      RandomAccessRange& state,
      int const num_events,
      RandomNumberGenerator& random_number_generator)
    {
      ::ket::generate_events(
        result, ::ket::utility::begin(state), ::ket::utility::end(state), num_events, random_number_generator);
    }

    template <typename StateInteger, typename Allocator, typename RandomAccessRange, typename RandomNumberGenerator>
    inline void generate_events(
      std::vector<StateInteger, Allocator>& result,
      RandomAccessRange& state,
      int const num_events,
      RandomNumberGenerator const& random_number_generator,
      typename RandomNumberGenerator::result_type const seed)
    {
      ::ket::generate_events(
        result, ::ket::utility::begin(state), ::ket::utility::end(state),
        num_events, random_number_generator, seed);
    }
  }
}


#endif

