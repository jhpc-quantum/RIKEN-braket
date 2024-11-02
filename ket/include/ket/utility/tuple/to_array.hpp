#ifndef KET_UTILITY_TUPLE_TO_ARRAY_HPP
# define KET_UTILITY_TUPLE_TO_ARRAY_HPP

# include <tuple>
# include <array>
# include <type_traits>

namespace ket
{
  namespace utility
  {
    namespace tuple
    {
      namespace impl
      {
# if __cpp_fold_expressions >= 201603L
        template <typename Tuple, typename T, std::size_t... indices>
        inline constexpr auto to_array(
          Tuple&& tuple, std::array<T, std::tuple_size<std::remove_cv_t<std::remove_reference_t<Tuple>>>::value>& array, std::index_sequence<indices...>)
        -> std::array<T, std::tuple_size<std::remove_cv_t<std::remove_reference_t<Tuple>>>::value>&
        {
          ((array[indices] = static_cast<T>(std::get<indices>(tuple))), ...);
          return array;
        }
# else // __cpp_fold_expressions >= 201603L
        template <std::size_t index, std::size_t count>
        struct to_array
        {
          template <typename Tuple, typename T>
          static constexpr auto call(Tuple&& tuple, std::array<T, count>& array) -> std::array<T, count>&
          {
            static_assert(
              std::is_convertible<std::tuple_element_t<index, std::remove_cv_t<std::remove_reference_t<Tuple>>>, T>::value,
              "Tuple's elements should be convertible to T");
            array[index] = static_cast<T>(std::get<index>(tuple));
            return ::ket::utility::tuple::impl::to_array<index + 1u, count>::call(std::forward<Tuple>(tuple), array);
          }
        }; // struct to_array<index, count>

        template <std::size_t count>
        struct to_array<count, count>
        {
          template <typename Tuple, typename T>
          static constexpr auto call(Tuple&&, std::array<T, count>& array) -> std::array<T, count>&
          { return array; }
        }; // struct to_array<count, count>
# endif // __cpp_fold_expressions >= 201603L
      } // namespace impl

# if __cpp_fold_expressions >= 201603L
      template <typename Tuple, typename T>
      inline constexpr auto to_array(
        Tuple&& tuple, std::array<T, std::tuple_size<std::remove_cv_t<std::remove_reference_t<Tuple>>>::value>& array)
      -> std::array<T, std::tuple_size<std::remove_cv_t<std::remove_reference_t<Tuple>>>::value>&
      { return ::ket::utility::tuple::impl::to_array(tuple, array, std::make_index_sequence<std::tuple_size<std::remove_cv_t<std::remove_reference_t<Tuple>>>::value>{}); }
# else // __cpp_fold_expressions >= 201603L
      template <typename Tuple, typename T>
      inline constexpr auto to_array(
        Tuple&& tuple, std::array<T, std::tuple_size<std::remove_cv_t<std::remove_reference_t<Tuple>>>::value>& array)
      -> std::array<T, std::tuple_size<std::remove_cv_t<std::remove_reference_t<Tuple>>>::value>&
      { return ::ket::utility::tuple::impl::to_array<0u, std::tuple_size<std::remove_cv_t<std::remove_reference_t<Tuple>>>::value>::call(tuple, array); }
# endif // __cpp_fold_expressions >= 201603L
    } // namespace tuple
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_TUPLE_TO_ARRAY_HPP

