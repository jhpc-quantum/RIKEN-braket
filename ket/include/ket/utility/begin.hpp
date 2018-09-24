#ifndef KET_UTILITY_BEGIN_HPP
# define KET_UTILITY_BEGIN_HPP

# include <boost/config.hpp>

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR

# include <boost/range/begin.hpp>
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
    inline typename boost::range_iterator<Range>::type begin(Range& range)
    { return boost::begin(range); }

    template <typename Range>
    inline typename boost::range_iterator<Range const>::type begin(Range const& range)
    { return boost::begin(range); }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    template <typename Value, typename Allocator>
    inline typename std::vector<Value, Allocator>::pointer begin(
      std::vector<Value, Allocator>& vector)
    { return KET_addressof(vector.front()); }

    template <typename Value, typename Allocator>
    inline typename std::vector<Value, Allocator>::const_pointer begin(
      std::vector<Value, Allocator> const& vector)
    { return KET_addressof(vector.front()); }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
  }
}


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif

#endif

