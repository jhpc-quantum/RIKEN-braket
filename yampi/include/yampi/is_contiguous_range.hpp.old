#ifndef YAMPI_IS_CONTIGUOUS_RANGE_HPP
# define YAMPI_IS_CONTIGUOUS_RANGE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
#   include <boost/type_traits/integral_constant.hpp>
# endif

# include <boost/range/iterator.hpp>
# include <boost/range/has_range_iterator.hpp>

# include <yampi/is_contiguous_iterator.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
#   define YAMPI_false_type std::false_type
# else
#   define YAMPI_enable_if boost::enable_if_c
#   define YAMPI_false_type boost::false_type
# endif


namespace yampi
{
  template <typename Range, typename Enable = void>
  struct is_contiguous_range
    : YAMPI_false_type
  { };

  template <typename Range>
  struct is_contiguous_range<
    Range,
    typename YAMPI_enable_if<
      boost::has_range_iterator<Range>::value
    >::type>
    : ::yampi::is_contiguous_iterator<typename boost::range_iterator<Range>::type>
  { };
}


# undef YAMPI_enable_if
# undef YAMPI_false_type

#endif

