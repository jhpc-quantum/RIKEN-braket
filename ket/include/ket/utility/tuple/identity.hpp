#ifndef KET_UTILITY_TUPLE_IDENTITY_HPP
# define KET_UTILITY_TUPLE_IDENTITY_HPP

# include <utility>


namespace ket
{
  namespace utility
  {
    namespace tuple
    {
      struct identity
      {
        template <typename Value>
        constexpr auto operator()(Value&& value) const noexcept -> Value&&
        { return std::forward<Value>(value); }
      }; // struct identity
    } // namespace tuple
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_TUPLE_IDENTITY_HPP
