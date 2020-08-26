#ifndef KET_UTILITY_META_ITERATOR_OF_HPP
# define KET_UTILITY_META_ITERATOR_OF_HPP

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
#   include <array>
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR

# include <boost/range/iterator.hpp>


namespace ket
{
  namespace utility
  {
    namespace meta
    {
      template <typename Range>
      struct iterator_of
        : boost::range_iterator<Range>
      { };

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <typename Value, typename Allocator>
      struct iterator_of<std::vector<Value, Allocator>>
      { using type = typename std::vector<Value, Allocator>::pointer; };

      template <typename Value, typename Allocator>
      struct iterator_of<std::vector<Value, Allocator> const>
      { using type = typename std::vector<Value, Allocator>::const_pointer; };

      template <typename Value, std::size_t num_elements>
      struct iterator_of<std::array<Value, num_elements>>
      { using type = typename std::array<Value, num_elements>::pointer; };

      template <typename Value, std::size_t num_elements>
      struct iterator_of<std::array<Value, num_elements> const>
      { using type = typename std::array<Value, num_elements>::const_pointer; };
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    } // namespace meta
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_META_ITERATOR_OF_HPP
