#ifndef KET_UTILITY_IS_UNIQUE_HPP
# define KET_UTILITY_IS_UNIQUE_HPP

# include <boost/config.hpp>

# include <iterator>
# include <algorithm>
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif

# include <boost/utility.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
//# include <boost/algorithm/cxx11/any_of.hpp>

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
    /*
    namespace is_unique_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <typename Iterator>
      struct is_unique_loop_inside
      {
        Iterator first_;

        explicit is_unique_loop_inside(Iterator const first) : first_(first) { }

        template <typename Value>
        bool operator()(Value const& value) const
        { return value == *first_; }
      };

      template <typename Iterator>
      inline is_unique_loop_inside<Iterator>
      make_is_unique_loop_inside(Iterator const first)
      { return is_unique_loop_inside<Iterator>(first); }
# endif
    }


    template <typename ForwardIterator>
    inline bool is_unique(
      ForwardIterator const first, ForwardIterator const last)
    {
# ifndef BOOST_NO_CXX11_LAMBDAS
      typedef
        typename std::iterator_traits<ForwardIterator>::value_type
        value_type;
      return not boost::algorithm::any_of(
        boost::next(first), last,
        [first](value_type const& value) { return value == *first; });
# else
      return not boost::algorithm::any_of(
        boost::next(first), last,
        ::ket::utility::is_unique_detail::make_is_unique_loop_inside(first));
# endif
    }
    */
    template <typename ForwardIterator>
    inline bool is_unique(
      ForwardIterator const first, ForwardIterator const last)
    {
      typedef
        typename std::iterator_traits<ForwardIterator>::value_type
        value_type;
      std::vector<value_type> sorted_values(first, last);
      std::sort(boost::begin(sorted_values), boost::end(sorted_values));
      return std::unique(boost::begin(sorted_values), boost::end(sorted_values)) == boost::end(sorted_values);
    }

    namespace ranges
    {
      template <typename ForwardRange>
      inline bool is_unique(ForwardRange const& range)
      { return ::ket::utility::is_unique(boost::begin(range), boost::end(range)); }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <typename Value, typename Allocator>
      inline bool is_unique(std::vector<Value, Allocator> const& range)
      {
        return ::ket::utility::is_unique(
          KET_addressof(range.front()), KET_addressof(range.front()) + range.size());
      }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    }
  }
}


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif

#endif

