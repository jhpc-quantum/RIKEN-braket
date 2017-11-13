#ifndef YAMPI_IS_CONTIGUOUS_ITERATOR_HPP
# define YAMPI_IS_CONTIGUOUS_ITERATOR_HPP

# include <boost/config.hpp>

# include <cstddef>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
#   include <boost/type_traits/is_same.hpp>
#   include <boost/type_traits/integral_constant.hpp>
# endif
# include <vector>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
#   define YAMPI_is_same std::is_same
#   define YAMPI_true_type std::true_type
#   define YAMPI_false_type std::false_type
# else
#   define YAMPI_enable_if boost::enable_if_c
#   define YAMPI_is_same boost::is_same
#   define YAMPI_true_type boost::true_type
#   define YAMPI_false_type boost::false_type
# endif


namespace yampi
{
  template <typename Iterator, typename Enable = void>
  struct is_contiguous_iterator
    : YAMPI_false_type
  { };

# if defined(__GNUC__) || defined(__IBMCPP__)
  template <typename Value, typename Allocator>
  struct is_contiguous_iterator<
    __gnu_cxx::__normal_iterator<Value*, std::vector<Value, Allocator> > >
    : YAMPI_true_type
  { };

  template <typename Value, typename Allocator>
  struct is_contiguous_iterator<
    __gnu_cxx::__normal_iterator<Value const*, std::vector<Value, Allocator> > >
    : YAMPI_true_type
  { };
# elif !defined(__FUJITSU)
  template <typename Iterator>
  struct is_contiguous_iterator<
    Iterator,
    typename YAMPI_enable_if<
      YAMPI_is_same<
        Iterator,
        typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator
      >::value
    >::type>
    : YAMPI_true_type
  { };

  template <typename Iterator>
  struct is_contiguous_iterator<
    Iterator,
    typename YAMPI_enable_if<
      YAMPI_is_same<
        Iterator,
        typename std::vector<typename std::iterator_traits<Iterator>::value_type>::const_iterator
      >::value
    >::type>
    : YAMPI_true_type
  { };
# endif

  template <typename Value>
  struct is_contiguous_iterator<Value*>
    : YAMPI_true_type
  { };

  template <typename Value>
  struct is_contiguous_iterator<Value const*>
    : YAMPI_true_type
  { };
}


# undef YAMPI_enable_if
# undef YAMPI_is_same
# undef YAMPI_true_type
# undef YAMPI_false_type

#endif

