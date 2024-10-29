#ifndef KET_UTILITY_TUPLE_TRANSFORM_HPP
# define KET_UTILITY_TUPLE_TRANSFORM_HPP

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
          template <typename Input, typename UnaryFunction, typename Projection, std::size_t... indices>
          inline constexpr auto transform(
            Input const& input, UnaryFunction&& unary_function, Projection&& projection, std::index_sequence<indices...>)
          { return std::make_tuple(unary_function(projection(std::get<indices>(input)))...); }

          template <typename Input1, typename Input2, typename BinaryFunction, typename Projection, std::size_t... indices>
          inline constexpr auto transform(
            Input1 const& input1, Input2 const& input2, BinaryFunction&& binary_function, Projection&& projection, std::index_sequence<indices...>)
          { return std::make_tuple(binary_function(projection(std::get<indices>(input1)), projection(std::get<indices>(input2)))...); }
        } // namespace impl

        template <typename Input, typename UnaryFunction, typename Projection>
        inline constexpr auto transform(Input const& input, UnaryFunction&& unary_function, Projection&& projection)
        {
          return ::ket::utility::tuple::proj::impl::transform(
            input, std::forward<UnaryFunction>(unary_function), std::forward<Projection>(projection),
            std::make_index_sequence<std::tuple_size<Input>::value>{});
        }

        template <typename Input1, typename Input2, typename BinaryFunction, typename Projection>
        inline constexpr auto transform(Input1 const& input1, Input2 const& input2, BinaryFunction&& binary_function, Projection&& projection)
        {
          return ::ket::utility::tuple::proj::impl::transform(
            input1, input2, std::forward<BinaryFunction>(binary_function), std::forward<Projection>(projection),
            std::make_index_sequence<std::min(std::tuple_size<Input1>::value, std::tuple_size<Input2>::value)>{});
        }
      } // namespace proj

      template <typename Input, typename UnaryFunction>
      inline constexpr auto transform(Input const& input, UnaryFunction&& unary_function)
      { return ::ket::utility::tuple::proj::transform(input, std::forward<UnaryFunction>(unary_function), ::ket::utility::tuple::identity{}); }

      template <typename Input1, typename Input2, typename BinaryFunction>
      inline constexpr auto transform(Input1 const& input1, Input2 const& input2, BinaryFunction&& binary_function)
      { return ::ket::utility::tuple::proj::transform(input1, input2, std::forward<BinaryFunction>(binary_function), ::ket::utility::tuple::identity{}); }
    } // namespace tuple
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_TUPLE_TRANSFORM_HPP

