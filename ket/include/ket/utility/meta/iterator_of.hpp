#ifndef KET_UTILITY_META_ITERATOR_OF_HPP
# define KET_UTILITY_META_ITERATOR_OF_HPP

# include <boost/config.hpp>

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
#   ifndef BOOST_NO_CXX11_HDR_ARRAY
#     include <array>
#   endif
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR

# include <boost/range/iterator.hpp>


namespace ket
{
  namespace utility
  {
    namespace meta
    {
      template <typename T>
      struct iterator_of
        : boost::range_iterator<T>
      { };

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <typename Value, typename Allocator>
      struct iterator_of< std::vector<Value, Allocator> >
      { typedef typename std::vector<Value, Allocator>::pointer type; };

      template <typename Value, typename Allocator>
      struct iterator_of<std::vector<Value, Allocator> const>
      { typedef typename std::vector<Value, Allocator>::const_pointer type; };

#   ifndef BOOST_NO_CXX11_HDR_ARRAY
      template <typename Value, std::size_t num_elements>
      struct iterator_of< std::array<Value, num_elements> >
      { typedef typename std::array<Value, num_elements>::pointer type; };

      template <typename Value, std::size_t num_elements>
      struct iterator_of<std::array<Value, num_elements> const>
      { typedef typename std::array<Value, num_elements>::const_pointer type; };
#   endif // BOOST_NO_CXX11_HDR_ARRAY
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    }
  }
}


#endif

