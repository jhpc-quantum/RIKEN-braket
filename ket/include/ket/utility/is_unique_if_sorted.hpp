#ifndef KET_UTILITY_IS_UNIQUE_IS_SORTED_HPP
# define KET_UTILITY_IS_UNIQUE_IS_SORTED_HPP

# include <iterator>
# include <algorithm>

# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>


namespace ket
{
  namespace utility
  {
    template <typename ForwardIterator>
    inline bool is_unique_if_sorted(
      ForwardIterator const first, ForwardIterator const last)
    {
      using value_type = typename std::iterator_traits<ForwardIterator>::value_type;
      auto sorted_values = std::vector<value_type>(first, last);
      std::sort(::ket::utility::begin(sorted_values), ::ket::utility::end(sorted_values));
      return std::unique(::ket::utility::begin(sorted_values), ::ket::utility::end(sorted_values)) == ::ket::utility::end(sorted_values);
    }

    namespace ranges
    {
      template <typename ForwardRange>
      inline bool is_unique_if_sorted(ForwardRange const& range)
      { return ::ket::utility::is_unique_if_sorted(::ket::utility::begin(range), ::ket::utility::end(range)); }
    }
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_IS_UNIQUE_IS_SORTED_HPP
