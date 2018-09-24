#ifndef KET_UTILITY_META_CONST_ITERATOR_OF_HPP
# define KET_UTILITY_META_CONST_ITERATOR_OF_HPP

# include <boost/config.hpp>

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
#   ifndef BOOST_NO_CXX11_HDR_ARRAY
#     include <array>
#   endif

#   include <boost/array.hpp>
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR

# include <boost/range/const_iterator.hpp>


namespace ket
{
  namespace utility
  {
    namespace meta
    {
      template <typename T>
      struct const_iterator_of
        : boost::range_const_iterator<T>
      { };

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <typename Value, typename Allocator>
      struct const_iterator_of< std::vector<Value, Allocator> >
      { typedef typename std::vector<Value, Allocator>::const_pointer type; };

      template <typename Value, typename Allocator>
      struct const_iterator_of<std::vector<Value, Allocator> const>
      { typedef typename std::vector<Value, Allocator>::const_pointer type; };

      template <typename Value, std::size_t num_elements>
      struct const_iterator_of< boost::array<Value, num_elements> >
      { typedef typename boost::array<Value, num_elements>::const_pointer type; };

      template <typename Value, std::size_t num_elements>
      struct const_iterator_of<boost::array<Value, num_elements> const>
      { typedef typename boost::array<Value, num_elements>::const_pointer type; };

#   ifndef BOOST_NO_CXX11_HDR_ARRAY
      template <typename Value, std::size_t num_elements>
      struct const_iterator_of< std::array<Value, num_elements> >
      { typedef typename std::array<Value, num_elements>::const_pointer type; };

      template <typename Value, std::size_t num_elements>
      struct const_iterator_of<std::array<Value, num_elements> const>
      { typedef typename std::array<Value, num_elements>::const_pointer type; };
#   endif // BOOST_NO_CXX11_HDR_ARRAY
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    }
  }
}


#endif

