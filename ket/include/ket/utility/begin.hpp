#ifndef KET_UTILITY_BEGIN_HPP
# define KET_UTILITY_BEGIN_HPP

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
#   include <array>
#   include <memory>
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR

# include <boost/range/begin.hpp>
# include <boost/range/iterator.hpp>


namespace ket
{
  namespace utility
  {
    template <typename Range>
    inline typename boost::range_iterator<Range>::type begin(Range& range)
    { return boost::begin(range); }

    template <typename Range>
    inline typename boost::range_iterator<Range const>::type begin(Range const& range)
    { return boost::begin(range); }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    template <typename Value, typename Allocator>
    inline typename std::vector<Value, Allocator>::pointer begin(
      std::vector<Value, Allocator>& vector)
    { return std::addressof(vector.front()); }

    template <typename Value, typename Allocator>
    inline typename std::vector<Value, Allocator>::const_pointer begin(
      std::vector<Value, Allocator> const& vector)
    { return std::addressof(vector.front()); }

    template <typename Value, std::size_t num_elements>
    inline typename std::array<Value, num_elements>::pointer begin(
      std::array<Value, num_elements>& array)
    { return std::addressof(array.front()); }

    template <typename Value, std::size_t num_elements>
    inline typename std::array<Value, num_elements>::const_pointer begin(
      std::array<Value, num_elements> const& array)
    { return std::addressof(array.front()); }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_BEGIN_HPP
