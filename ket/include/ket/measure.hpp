#ifndef KET_MEASURE_HPP
#define KET_MEASURE_HPP

# include <boost/config.hpp>

# include <cmath>
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
# endif
# ifndef BOOST_NO_CXX11_HDR_RANDOM
#   include <random>
# else
#   include <boost/random/uniform_real_distribution.hpp>
# endif
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
# include <ket/utility/meta/real_of.hpp>

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
  namespace measure_detail
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


  template <typename ParallelPolicy, typename State, typename RandomNumberGenerator>
  inline typename boost::range_size<State>::type measure(
    ParallelPolicy const parallel_policy,
    State& state, RandomNumberGenerator& random_number_generator)
  {
    typedef typename boost::range_value<State>::type complex_type;

    ::ket::utility::ranges::inclusive_scan(
      parallel_policy,
      state | boost::adaptors::transformed(
                ::ket::measure_detail::complex_norm<complex_type>()),
      boost::begin(state));

    KET_uniform_real_distribution<double> distribution(0.0, 1.0);
# ifndef BOOST_NO_CXX11_LAMBDAS
    typename boost::range_size<State>::type const result
      = boost::size(boost::upper_bound<boost::return_begin_found>(
          state,
          static_cast<complex_type>(
            *boost::prior(boost::end(state)) * distribution(random_number_generator)),
          [](complex_type const& lhs, complex_type const& rhs)
          { return real(lhs) < real(rhs); }));
# else // BOOST_NO_CXX11_LAMBDAS
    typename boost::range_size<State>::type const result
      = boost::size(boost::upper_bound<boost::return_begin_found>(
          state,
          static_cast<complex_type>(
            *boost::prior(boost::end(state)) * distribution(random_number_generator)),
          ::ket::measure_detail::real_part_less_than()));
# endif // BOOST_NO_CXX11_LAMBDAS

    typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;

    boost::fill(state, complex_type(static_cast<real_type>(0)));
    *(boost::begin(state)+result) = complex_type(static_cast<real_type>(1));

    return result;
  }

  template <typename State, typename RandomNumberGenerator>
  inline typename boost::range_size<State>::type measure(
    State& state, RandomNumberGenerator& random_number_generator)
  {
    return ::ket::measure(
      ::ket::utility::policy::make_sequential(),
      state, random_number_generator);
  }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
  template <typename ParallelPolicy, typename Complex, typename Allocator, typename RandomNumberGenerator>
  inline typename std::vector<Complex, Allocator>::size_type measure(
    ParallelPolicy const parallel_policy,
    std::vector<Complex, Allocator>& state, RandomNumberGenerator& random_number_generator)
  {
    typedef typename boost::range_value<State>::type complex_type;

    ::ket::utility::ranges::inclusive_scan(
      parallel_policy,
      boost::make_iterator_range(
        KET_addressof(state.front()), KET_addressof(state.front()) + state.size())
        | boost::adaptors::transformed(
            ::ket::measure_detail::complex_norm<Complex>()),
      KET_addressof(state.front()));

    KET_uniform_real_distribution<double> distribution(0.0, 1.0);
#   ifndef BOOST_NO_CXX11_LAMBDAS
    typename boost::range_size<State>::type const result
      = boost::size(boost::upper_bound<boost::return_begin_found>(
          boost::make_iterator_range(
            KET_addressof(state.front()), KET_addressof(state.front()) + state.size()),
          static_cast<complex_type>(
            state.back() * distribution(random_number_generator)),
          [](complex_type const& lhs, complex_type const& rhs)
          { return real(lhs) < real(rhs); }));
#   else // BOOST_NO_CXX11_LAMBDAS
    typename boost::range_size<State>::type const result
      = boost::size(boost::upper_bound<boost::return_begin_found>(
          boost::make_iterator_range(
            KET_addressof(state.front()), KET_addressof(state.front()) + state.size()),
          static_cast<complex_type>(
            state.back() * distribution(random_number_generator)),
          ::ket::measure_detail::real_part_less_than()));
#   endif // BOOST_NO_CXX11_LAMBDAS

    typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;

    std::fill(
      KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
      complex_type(static_cast<real_type>(0)));
    state[result] = complex_type(static_cast<real_type>(1));

    return result;
  }

  template <typename Complex, typename Allocator, typename RandomNumberGenerator>
  inline typename std::vector<Complex, Allocator>::size_type measure(
    std::vector<Complex, Allocator>& state, RandomNumberGenerator& random_number_generator)
  {
    return ::ket::measure(
      ::ket::utility::policy::make_sequential(),
      state, random_number_generator);
  }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
}


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif
# undef KET_uniform_real_distribution

#endif

