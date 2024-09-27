#ifndef KET_UTILITY_IS_UNIQUE_IS_SORTED_HPP
# define KET_UTILITY_IS_UNIQUE_IS_SORTED_HPP

# include <iterator>
# include <algorithm>


namespace ket
{
  namespace utility
  {
    template <typename ForwardIterator>
    inline auto is_unique_if_sorted(ForwardIterator const first, ForwardIterator const last) -> bool
    {
      using value_type = typename std::iterator_traits<ForwardIterator>::value_type;
      auto sorted_values = std::vector<value_type>(first, last);

      using std::begin;
      using std::end;
      std::sort(begin(sorted_values), end(sorted_values));

      return std::unique(begin(sorted_values), end(sorted_values)) == end(sorted_values);
    }

    namespace ranges
    {
      template <typename ForwardRange>
      inline auto is_unique_if_sorted(ForwardRange const& range) -> bool
      { using std::begin; using std::end; return ::ket::utility::is_unique_if_sorted(begin(range), end(range)); }
    }
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_IS_UNIQUE_IS_SORTED_HPP
