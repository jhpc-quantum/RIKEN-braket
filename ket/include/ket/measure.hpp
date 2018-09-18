#ifndef KET_MEASURE_HPP
#define KET_MEASURE_HPP

# include <boost/config.hpp>

# include <cmath>
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
# endif
# include <iterator>
# include <algorithm>
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/difference_type.hpp>
# include <boost/utility.hpp>

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
  namespace measure_detail
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


  template <typename ParallelPolicy, typename RandomAccessIterator, typename RandomNumberGenerator>
  inline typename std::iterator_traits<RandomAccessIterator>::difference_type measure(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    RandomNumberGenerator& random_number_generator)
  {
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
      ::ket::measure_detail::real_part_plus<complex_type>(),
      ::ket::measure_detail::complex_norm<complex_type>());
# endif // BOOST_NO_CXX11_LAMBDAS

# ifndef BOOST_NO_CXX11_LAMBDAS
    RandomAccessIterator const found
      = std::upper_bound(
          first, last,
          static_cast<complex_type>(
            ::ket::utility::positive_random_value_upto(
              real(*boost::prior(last)), random_number_generator)),
          [](complex_type const& lhs, complex_type const& rhs)
          { using std::real; return real(lhs) < real(rhs); });
# else // BOOST_NO_CXX11_LAMBDAS
    RandomAccessIterator const found
      = std::upper_bound(
          first, last,
          static_cast<complex_type>(
            ::ket::utility::positive_random_value_upto(
              real(*boost::prior(last)), random_number_generator)),
          ::ket::measure_detail::real_part_less_than<complex_type>());
# endif // BOOST_NO_CXX11_LAMBDAS
    typename std::iterator_traits<RandomAccessIterator>::difference_type const result
      = found - first;

    typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;

    std::fill(first, last, complex_type(static_cast<real_type>(0)));
    *(first + result) = complex_type(static_cast<real_type>(1));

    return result;
  }

  template <typename RandomAccessIterator, typename RandomNumberGenerator>
  inline typename std::iterator_traits<RandomAccessIterator>::difference_type measure(
    RandomAccessIterator const first, RandomAccessIterator const last,
    RandomNumberGenerator& random_number_generator)
  {
    return ::ket::measure(
      ::ket::utility::policy::make_sequential(),
      first, last, random_number_generator);
  }


  namespace ranges
  {
    template <typename ParallelPolicy, typename RandomAccessRange, typename RandomNumberGenerator>
    inline typename boost::range_difference<RandomAccessRange>::type measure(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& state, RandomNumberGenerator& random_number_generator)
    {
      return ::ket::measure(
        parallel_policy,
        boost::begin(state), boost::end(state),
        random_number_generator);
    }

    template <typename RandomAccessRange, typename RandomNumberGenerator>
    inline typename boost::range_difference<RandomAccessRange>::type measure(
      RandomAccessRange& state, RandomNumberGenerator& random_number_generator)
    {
      return ::ket::measure(
        boost::begin(state), boost::end(state),
        random_number_generator);
    }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    template <typename ParallelPolicy, typename Complex, typename Allocator, typename RandomNumberGenerator>
    inline typename std::vector<Complex, Allocator>::difference_type measure(
      ParallelPolicy const parallel_policy,
      std::vector<Complex, Allocator>& state, RandomNumberGenerator& random_number_generator)
    {
      return ::ket::measure(
        parallel_policy,
        KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
        random_number_generator);
    }

    template <typename Complex, typename Allocator, typename RandomNumberGenerator>
    inline typename std::vector<Complex, Allocator>::difference_type measure(
      std::vector<Complex, Allocator>& state, RandomNumberGenerator& random_number_generator)
    {
      return ::ket::measure(
        ::ket::utility::policy::make_sequential(),
        KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
        random_number_generator);
    }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
  }
}


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif

#endif

