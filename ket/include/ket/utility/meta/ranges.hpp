#ifndef KET_UTILITY_META_RANGES_HPP
# define KET_UTILITY_META_RANGES_HPP

# include <iterator>
# include <type_traits>
# include <utility>

namespace ket
{
  namespace utility
  {
    namespace meta
    {
      namespace ranges_detail
      {
        template <typename Range>
        inline auto begin(Range& range) { using std::begin; return begin(range); }

        template <typename Range>
        inline auto begin(Range const& range) { using std::begin; return begin(range); }

        template <typename Range>
        inline auto cbegin(Range const& range) { using std::cbegin; return cbegin(range); }

        template <typename Range>
        inline auto end(Range& range) { using std::end; return end(range); }

        template <typename Range>
        inline auto end(Range const& range) { using std::end; return end(range); }

        template <typename Range>
        inline auto cend(Range const& range) { using std::cend; return cend(range); }
      } // namespace ranges_detail

      template <typename Range>
      using iterator_t = decltype(::ket::utility::meta::ranges_detail::begin(std::declval<Range&>()));

      template <typename Range>
      using const_iterator_t = decltype(::ket::utility::meta::ranges_detail::cbegin(std::declval<std::remove_reference_t<Range> const&>()));

      template <typename Range>
      using range_difference_t = typename std::iterator_traits< ::ket::utility::meta::iterator_t<Range> >::difference_type;

      template <typename Range>
      using range_value_t = typename std::iterator_traits< ::ket::utility::meta::iterator_t<Range> >::value_type;

      template <typename Range>
      using range_pointer_t = typename std::iterator_traits< ::ket::utility::meta::iterator_t<Range> >::pointer;

      template <typename Range>
      using range_reference_t = typename std::iterator_traits< ::ket::utility::meta::iterator_t<Range> >::reference;
    } // namespace meta
  } // namespace utility
} // namespace ket

#endif // KET_UTILITY_META_RANGES_HPP
