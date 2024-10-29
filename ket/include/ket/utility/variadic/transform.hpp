#ifndef KET_UTILITY_VARIADIC_TRANSFORM_HPP
# define KET_UTILITY_VARIADIC_TRANSFORM_HPP

# include <cstddef>
# include <tuple>
# include <utility>

# include <ket/utility/variadic/identity.hpp>


namespace ket
{
  namespace utility
  {
    namespace variadic
    {
      namespace proj
      {
        template <typename UnaryFunction, typename Projection, typename... Arguments>
        inline constexpr auto transform(UnaryFunction&& unary_function, Projection&& projection, Arguments&&... arguments)
        { return std::make_tuple(unary_function(projection(std::forward<Arguments>(arguments)))...); }
      } // namespace proj

      template <typename UnaryFunction, typename... Arguments>
      inline constexpr auto transform(UnaryFunction&& unary_function, Arguments&&... arguments)
      { return ::ket::utility::variadic::proj::transform(std::forward<UnaryFunction>(unary_function), ::ket::utility::variadic::identity{}, std::forward<Arguments>(arguments)...); }
    } // namespace variadic
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_VARIADIC_TRANSFORM_HPP

