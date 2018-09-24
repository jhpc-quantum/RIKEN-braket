#ifndef KET_UTILITY_META_CONST_ITERATOR_OF_HPP
# define KET_UTILITY_META_CONST_ITERATOR_OF_HPP

# include <boost/config.hpp>

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
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
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    }
  }
}


#endif

