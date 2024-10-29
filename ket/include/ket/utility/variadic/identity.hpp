#ifndef KET_UTILITY_VARIADIC_IDENTITY_HPP
# define KET_UTILITY_VARIADIC_IDENTITY_HPP

# include <utility>


namespace ket
{
  namespace utility
  {
    namespace variadic
    {
      struct identity
      {
        template <typename Value>
        constexpr auto operator()(Value&& value) const noexcept -> Value&&
        { return std::forward<Value>(value); }
      }; // struct identity
    } // namespace variadic
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_VARIADIC_IDENTITY_HPP
