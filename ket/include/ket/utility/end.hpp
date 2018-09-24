#ifndef KET_UTILITY_END_HPP
# define KET_UTILITY_END_HPP

# include <boost/config.hpp>

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR

# include <boost/range/end.hpp>
# include <boost/range/iterator.hpp>

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     define KET_addressof std::addressof
#   else
#     define KET_addressof boost::addressof
#   endif
# endif


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
    { return KET_addressof(vector.front()) + vector.size(); }

    template <typename Value, typename Allocator>
    inline typename std::vector<Value, Allocator>::const_pointer end(
      std::vector<Value, Allocator> const& vector)
    { return KET_addressof(vector.front()) + vector.size(); }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
  }
}


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif

#endif

