#ifndef KET_UTILITY_VARIADIC_ALL_OF_HPP
# define KET_UTILITY_VARIADIC_ALL_OF_HPP

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
        namespace impl
        {
          template <typename Predicate, typename Projection>
          inline constexpr auto all_of(Predicate&&, Projection&&) -> bool
          { return true; }

          template <typename Predicate, typename Projection, typename Argument, typename... Arguments>
          inline constexpr auto all_of(Predicate&& predicate, Projection&& projection, Argument&& argument, Arguments&&... arguments) -> bool
          {
            return predicate(projection(std::forward<Argument>(argument))) and ::ket::utility::variadic::proj::impl::all_of(
              std::forward<Predicate>(predicate), std::forward<Projection>(projection), std::forward<Arguments>(arguments)...);
          }
        } // namespace impl

        template <typename Predicate, typename Projection, typename... Arguments>
        inline constexpr auto all_of(Predicate&& predicate, Projection&& projection, Arguments&&... arguments) -> bool
        {
          return ::ket::utility::variadic::proj::impl::all_of(
            std::forward<Predicate>(predicate), std::forward<Projection>(projection), std::forward<Arguments>(arguments)...);
        }
      } // namespace proj

      template <typename Predicate, typename... Arguments>
      inline constexpr auto all_of(Predicate&& predicate, Arguments&&... arguments) -> bool
      { return ::ket::utility::variadic::proj::all_of(std::forward<Predicate>(predicate), ::ket::utility::variadic::identity{}, std::forward<Arguments>(arguments)...); }
    } // namespace variadic
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_VARIADIC_ALL_OF_HPP

