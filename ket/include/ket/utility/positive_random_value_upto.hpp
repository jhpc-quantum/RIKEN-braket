#ifndef KET_UTILITY_POSITIVE_RANDOM_VALUE_UPTO_HPP
# define KET_UTILITY_POSITIVE_RANDOM_VALUE_UPTO_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_RANDOM
#  include <random>
# else
#  include <boost/random/uniform_real_distribution.hpp>
# endif
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#  include <type_traits>
# else
#  include <boost/type_traits/is_floating_point.hpp>
#  include <boost/utility/enable_if.hpp>
# endif

# ifndef BOOST_NO_CXX11_HDR_RANDOM
#  define KET_uniform_real_distribution std::uniform_real_distribution
# else
#  define KET_uniform_real_distribution boost::random::uniform_real_distribution
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_enable_if std::enable_if
#   define KET_is_floating_point std::is_floating_point
# else
#   define KET_enable_if boost::enable_if_c
#   define KET_is_floating_point boost::is_floating_point
# endif


namespace ket
{
  namespace utility
  {
    template <typename Real, typename RandomNumberGenerator>
    inline typename KET_enable_if<KET_is_floating_point<Real>::value, Real>::type
    positive_random_value_upto(
      Real const maximum_value, RandomNumberGenerator& random_number_generator)
    {
      KET_uniform_real_distribution<Real> distribution(static_cast<Real>(0), maximum_value);
      return distribution(random_number_generator);
    }

    template <typename Real, typename RandomNumberGenerator>
    inline typename KET_enable_if<not KET_is_floating_point<Real>::value, Real>::type
    positive_random_value_upto(
      Real const maximum_value, RandomNumberGenerator& random_number_generator)
    {
      KET_uniform_real_distribution<double> distribution(0.0, static_cast<double>(maximum_value));
      return static_cast<Real>(distribution(random_number_generator));
    }
  }
}

# undef KET_enable_if
# undef KET_is_floating_point
# undef KET_uniform_real_distribution


#endif
