#ifndef KET_GENERATE_EVENTS_HPP
#define KET_GENERATE_EVENTS_HPP

# include <boost/config.hpp>

# include <cmath>
# include <vector>
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/utility.hpp>

# include <boost/range/size.hpp>
# include <boost/range/value_type.hpp>
# include <boost/range/adaptor/transformed.hpp>
# include <boost/range/algorithm/fill.hpp>
# include <boost/range/algorithm/upper_bound.hpp>
# include <boost/range/numeric.hpp>

# include <ket/utility/loop_n.hpp>
# include <ket/utility/positive_random_value_upto.hpp>
# include <ket/utility/meta/real_of.hpp>

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     define KET_addressof std::addressof
#   else
#     define KET_addressof boost::addressof
#   endif
# endif


namespace ket
{
  namespace generate_events_detail
  {
    template <typename Complex>
    struct complex_norm
    {
      typedef Complex result_type;

      Complex operator()(Complex const& value) const
      { using std::norm; return static_cast<Complex>(norm(value)); }
    };

# ifdef BOOST_NO_CXX11_LAMBDAS
    struct real_part_less_than
    {
      typedef bool result_type;

      template <typename Complex>
      bool operator()(Complex const& lhs, Complex const& rhs) const
      { using std::real; return real(lhs) < real(rhs); }
    };
# endif
  }


  template <typename ParallelPolicy, typename StateInteger, typename Allocator, typename State, typename RandomNumberGenerator>
  inline void generate_events(
    ParallelPolicy const parallel_policy,
    std::vector<StateInteger, Allocator>& result,
    State& state,
    int const num_events,
    RandomNumberGenerator& random_number_generator)
  {
    result.clear();
    result.reserve(num_events);

    typedef typename boost::range_value<State>::type complex_type;
    ::ket::utility::ranges::inclusive_scan(
      parallel_policy,
      state | boost::adaptors::transformed(
                ::ket::generate_events_detail::complex_norm<complex_type>()),
      boost::begin(state));

    using std::real;
    typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
    real_type const total_probability = real(*boost::prior(boost::end(state)));

# ifndef BOOST_NO_CXX11_LAMBDAS
    for (int event_index = 0; event_index < num_events; ++event_index)
      result.push_back(
        boost::size(boost::upper_bound<boost::return_begin_found>(
          state,
          static_cast<complex_type>(
            ::ket::utility::positive_random_value_upto(
              total_probability, random_number_generator)),
          [](complex_type const& lhs, complex_type const& rhs)
          { return real(lhs) < real(rhs); })));
# else // BOOST_NO_CXX11_LAMBDAS
    for (int event_index = 0; event_index < num_events; ++event_index)
      result.push_back(
        boost::size(boost::upper_bound<boost::return_begin_found>(
          state,
          static_cast<complex_type>(
            ::ket::utility::positive_random_value_upto(
              total_probability, random_number_generator)),
          ::ket::generate_events_detail::real_part_less_than())));
# endif // BOOST_NO_CXX11_LAMBDAS
  }

  template <typename ParallelPolicy, typename StateInteger, typename Allocator, typename State, typename RandomNumberGenerator>
  inline void generate_events(
    ParallelPolicy const parallel_policy,
    std::vector<StateInteger, Allocator>& result,
    State& state,
    int const num_events,
    RandomNumberGenerator const&,
    typename RandomNumberGenerator::result_type const seed)
  {
    RandomNumberGenerator random_number_generator(seed);
    ::ket::generate_events(parallel_policy, result, num_events, random_number_generator);
  }

  template <typename StateInteger, typename Allocator, typename State, typename RandomNumberGenerator>
  inline void generate_events(
    std::vector<StateInteger, Allocator>& result,
    State& state,
    int const num_events,
    RandomNumberGenerator& random_number_generator)
  {
    ::ket::generate_events(
      ::ket::utility::policy::make_sequential(),
      result, state, num_events, random_number_generator);
  }

  template <typename StateInteger, typename Allocator, typename State, typename RandomNumberGenerator>
  inline void generate_events(
    std::vector<StateInteger, Allocator>& result,
    State& state,
    int const num_events,
    RandomNumberGenerator const&,
    typename RandomNumberGenerator::result_type const seed)
  {
    RandomNumberGenerator random_number_generator(seed);
    ::ket::generate_events(result, num_events, random_number_generator);
  }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
  template <
    typename ParallelPolicy, typename StateInteger, typename ResultAllocator,
    typename Complex, typename Allocator, typename RandomNumberGenerator>
  inline void generate_events(
    ParallelPolicy const parallel_policy,
    std::vector<StateInteger, ResultAllocator>& result,
    std::vector<Complex, Allocator>& state,
    int const num_events,
    RandomNumberGenerator& random_number_generator)
  {
    result.clear();
    result.reserve(num_events);

    typedef typename boost::range_value<State>::type complex_type;
    ::ket::utility::ranges::inclusive_scan(
      parallel_policy,
      boost::make_iterator_range(
        KET_addressof(state.front()), KET_addressof(state.front()) + state.size())
        | boost::adaptors::transformed(
            ::ket::generate_events_detail::complex_norm<Complex>()),
      KET_addressof(state.front()));

    using std::real;
    typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
    real_type const total_probability = real(state.back());

#   ifndef BOOST_NO_CXX11_LAMBDAS
    for (int event_index = 0; event_index < num_events; ++event_index)
      result.push_back(
        boost::size(boost::upper_bound<boost::return_begin_found>(
          boost::make_iterator_range(
            KET_addressof(state.front()), KET_addressof(state.front()) + state.size()),
          static_cast<complex_type>(
            ::ket::utility::positive_random_value_upto(
              total_probability, random_number_generator)),
          [](complex_type const& lhs, complex_type const& rhs)
          { return real(lhs) < real(rhs); })));
#   else // BOOST_NO_CXX11_LAMBDAS
    for (int event_index = 0; event_index < num_events; ++event_index)
      result.push_back(
        boost::size(boost::upper_bound<boost::return_begin_found>(
          boost::make_iterator_range(
            KET_addressof(state.front()), KET_addressof(state.front()) + state.size()),
          static_cast<complex_type>(
            ::ket::utility::positive_random_value_upto(
              total_probability, random_number_generator)),
          ::ket::generate_events_detail::real_part_less_than())));
#   endif // BOOST_NO_CXX11_LAMBDAS
  }

  template <
    typename ParallelPolicy, typename StateInteger, typename ResultAllocator,
    typename Complex, typename Allocator, typename RandomNumberGenerator>
  inline void generate_events(
    ParallelPolicy const parallel_policy,
    std::vector<StateInteger, ResultAllocator>& result,
    std::vector<Complex, Allocator>& state,
    int const num_events,
    RandomNumberGenerator const&,
    typename RandomNumberGenerator::result_type const seed)
  {
    RandomNumberGenerator random_number_generator(seed);
    ::ket::generate_events(parallel_policy, result, state, num_events, random_number_generator);
  }

  template <
    typename StateInteger, typename ResultAllocator, typename Complex, typename Allocator, typename RandomNumberGenerator>
  inline void generate_events(
    std::vector<StateInteger, ResultAllocator>& result,
    std::vector<Complex, Allocator>& state,
    int const num_events,
    RandomNumberGenerator& random_number_generator)
  {
    return ::ket::generate_events(
      ::ket::utility::policy::make_sequential(),
      result, state, num_events, random_number_generator);
  }

  template <
    typename StateInteger, typename ResultAllocator,
    typename Complex, typename Allocator, typename RandomNumberGenerator>
  inline void generate_events(
    std::vector<StateInteger, ResultAllocator>& result,
    std::vector<Complex, Allocator>& state,
    int const num_events,
    RandomNumberGenerator const&,
    typename RandomNumberGenerator::result_type const seed)
  {
    RandomNumberGenerator random_number_generator(seed);
    ::ket::generate_events(result, state, num_events, random_number_generator);
  }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
}


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif

#endif

