#ifndef KET_UTILITY_CONTAINS_HPP
# define KET_UTILITY_CONTAINS_HPP

# include <algorithm>

namespace ket
{
  namespace utility
  {
    template <typename InputIterator, typename Value>
    inline auto contains(InputIterator const first, InputIterator const last, Value const& value) -> bool
    { return std::find(first, last, value) != last; }
  } // namespace utility
} // namespace ket

#endif // KET_UTILITY_CONTAINS_HPP

