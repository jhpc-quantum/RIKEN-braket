#ifndef KET_UTILITY_END_HPP
# define KET_UTILITY_END_HPP

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
#   include <array>
#   include <memory>
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR

# include <boost/range/end.hpp>
# include <boost/range/iterator.hpp>


namespace ket
{
  namespace utility
  {
    template <typename Range>
    inline typename boost::range_iterator<Range>::type end(Range& range)
    { return boost::end(range); }

    template <typename Range>
    inline typename boost::range_iterator<Range const>::type end(Range const& range)
    { return boost::end(range); }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    template <typename Value, typename Allocator>
    inline typename std::vector<Value, Allocator>::pointer end(
      std::vector<Value, Allocator>& vector)
    { return std::addressof(vector.front()) + vector.size(); }

    template <typename Value, typename Allocator>
    inline typename std::vector<Value, Allocator>::const_pointer end(
      std::vector<Value, Allocator> const& vector)
    { return std::addressof(vector.front()) + vector.size(); }

    template <typename Value, std::size_t num_elements>
    inline typename std::array<Value, num_elements>::pointer end(
      std::array<Value, num_elements>& array)
    { return std::addressof(array.front()) + num_elements; }

    template <typename Value, std::size_t num_elements>
    inline typename std::array<Value, num_elements>::const_pointer end(
      std::array<Value, num_elements> const& array)
    { return std::addressof(array.front()) + num_elements; }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_END_HPP
