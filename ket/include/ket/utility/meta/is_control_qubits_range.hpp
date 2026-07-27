#ifndef KET_UTILITY_META_IS_CONTROL_QUBITS_RANGE_HPP
# define KET_UTILITY_META_IS_CONTROL_QUBITS_RANGE_HPP

# include <type_traits>

# include <ket/control.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  namespace utility
  {
    namespace meta
    {
      template <typename Range, typename = void>
      struct is_control_qubits_range
        : std::false_type
      { };

      template <typename Range>
      struct is_control_qubits_range<
        Range,
        std::enable_if_t<
          ::ket::meta::is_control_cvref<
            ::ket::utility::meta::range_value_t<Range> >::value> >
        : std::true_type
      { };
    } // namespace meta
  } // namespace utility
} // namespace ket

#endif // KET_UTILITY_META_IS_CONTROL_QUBITS_RANGE_HPP
