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
        using std::begin;
        using std::cbegin;
        using std::cend;
        using std::end;

        template <typename Range>
        auto adl_begin(Range& range) -> decltype(begin(range));

        template <typename Range>
        auto adl_begin(Range const& range) -> decltype(begin(range));

        template <typename Range>
        auto adl_cbegin(Range const& range) -> decltype(cbegin(range));

        template <typename Range>
        auto adl_end(Range& range) -> decltype(end(range));

        template <typename Range>
        auto adl_end(Range const& range) -> decltype(end(range));

        template <typename Range>
        auto adl_cend(Range const& range) -> decltype(cend(range));
      } // namespace ranges_detail

      template <typename Range>
      using iterator_t = decltype(::ket::utility::meta::ranges_detail::adl_begin(std::declval<Range&>()));

      template <typename Range>
      using const_iterator_t = decltype(::ket::utility::meta::ranges_detail::adl_cbegin(std::declval<std::remove_reference_t<Range> const&>()));

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
