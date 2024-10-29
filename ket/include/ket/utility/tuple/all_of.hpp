#ifndef KET_UTILITY_TUPLE_ALL_OF_HPP
# define KET_UTILITY_TUPLE_ALL_OF_HPP

# include <cstddef>
# include <tuple>
# include <utility>

# include <ket/utility/tuple/identity.hpp>


namespace ket
{
  namespace utility
  {
    namespace tuple
    {
      namespace proj
      {
        namespace impl
        {
# if __cpp_fold_expressions >= 201603L
          template <typename Input, typename Predicate, typename Projection, std::size_t... indices>
          inline constexpr auto all_of(Input const& input, Predicate&& predicate, Projection&& projection, std::index_sequence<indices...>) -> bool
          { return (predicate(projection(std::get<indices>(input))) and...); }
# else // __cpp_fold_expressions >= 201603L
          template <std::size_t index, std::size_t count>
          struct all_of
          {
            template <typename Input, typename Predicate, typename Projection>
            static constexpr auto call(Input const& input, Predicate&& predicate, Projection&& projection) -> bool
            {
              return predicate(projection(std::get<index>(input))) and ::ket::utility::tuple::proj::impl::all_of<index + 1u, count>::call(
                input, std::forward<Predicate>(predicate), std::forward<Projection>(projection));
            }
          }; // struct all_of<index, count>

          template <std::size_t count>
          struct all_of<count, count>
          {
            template <typename Input, typename Predicate, typename Projection>
            static constexpr auto call(Input const&, Predicate&&, Projection&&) -> bool
            { return true; }
          }; // struct all_of<count, count>
# endif // __cpp_fold_expressions >= 201603L
        } // namespace impl

# if __cpp_fold_expressions >= 201603L
        template <typename Input, typename Predicate, typename Projection>
        inline constexpr auto all_of(Input const& input, Predicate&& predicate, Projection&& projection) -> bool
        {
          return ::ket::utility::tuple::proj::impl::all_of(
            input, std::forward<Predicate>(predicate), std::forward<Projection>(projection), std::make_index_sequence<std::tuple_size<Input>::value>{});
        }
# else // __cpp_fold_expressions >= 201603L
        template <typename Input, typename Predicate, typename Projection>
        inline constexpr auto all_of(Input const& input, Predicate&& predicate, Projection&& projection) -> bool
        {
          return ::ket::utility::tuple::proj::impl::all_of<0u, std::tuple_size<Input>::value>::call(
            input, std::forward<Predicate>(predicate), std::forward<Projection>(projection));
        }
# endif // __cpp_fold_expressions >= 201603L
      } // namespace proj

      template <typename Input, typename Predicate>
      inline constexpr auto all_of(Input const& input, Predicate&& predicate) -> bool
      { return ::ket::utility::tuple::proj::all_of(input, std::forward<Predicate>(predicate), ::ket::utility::tuple::identity{}); }
    } // namespace tuple
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_TUPLE_ALL_OF_HPP

