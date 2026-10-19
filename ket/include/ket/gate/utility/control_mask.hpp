#ifndef KET_GATE_UTILITY_CONTROL_MASK_HPP
# define KET_GATE_UTILITY_CONTROL_MASK_HPP

# include <cstddef>
# include <iterator>
# include <limits>
# include <stdexcept>
# include <type_traits>
# include <utility>

# include <ket/control.hpp>
# include <ket/runtime_control.hpp>

namespace ket { namespace gate { namespace utility
{
  template <typename Qubit>
  struct is_default_control_operand : std::true_type { };

  template <typename Qubit, typename Qubit::state_integer_type control_value>
  struct is_default_control_operand< ::ket::control<Qubit, control_value> >
    : std::integral_constant<bool, control_value == ::ket::meta::base_v<Qubit> - 1u> { };

  template <typename Qubit>
  struct is_default_control_operand< ::ket::runtime_control<Qubit> > : std::false_type { };

  template <typename... Qubits>
  struct all_default_control_operands;
  template <>
  struct all_default_control_operands<> : std::true_type { };
  template <typename Qubit, typename... Qubits>
  struct all_default_control_operands<Qubit, Qubits...>
    : std::integral_constant<bool, is_default_control_operand<Qubit>::value
                                  and all_default_control_operands<Qubits...>::value> { };

  template <typename... Qubits>
  struct all_controls;

  template <>
  struct all_controls<> : std::true_type { };

  template <typename Qubit, typename... Qubits>
  struct all_controls<Qubit, Qubits...>
    : std::integral_constant<bool, ::ket::meta::is_control_cvref<Qubit>::value and all_controls<Qubits...>::value>
  { };

  template <typename Control, typename Qubit, bool = ::ket::meta::is_control_cvref<Control>::value>
  struct is_control_of : std::false_type { };

  template <typename Control, typename Qubit>
  struct is_control_of<Control, Qubit, true>
    : std::is_same<decltype(::ket::remove_control(std::declval<Control>())), Qubit>
  { };

  namespace control_mask_detail
  {
    template <typename Qubit>
    constexpr auto is_negative(Qubit const&, std::false_type) noexcept -> bool { return false; }

    template <typename Control>
    constexpr auto is_negative(Control const control, std::true_type) noexcept -> bool
    { return ::ket::control_value(control) == 0u; }
  }

  // Bit i refers to operand i in the unsorted/packed indexing convention.
  // XOR this mask into an all-positive kernel's packed index to select zero
  // controls directly, without modifying the state vector or target ordering.
  template <typename UnsignedInteger, typename... Qubits>
  constexpr auto negative_control_mask(Qubits const... qubits) noexcept -> UnsignedInteger
  {
    static_assert(std::is_unsigned<UnsignedInteger>::value, "Unsigned index required");
    static_assert(sizeof...(Qubits) < std::numeric_limits<UnsignedInteger>::digits, "Too many operands for packed index");
    auto result = UnsignedInteger{0u};
    auto bit = std::size_t{0u};
    using swallow = int[];
    (void)swallow{0, (result |= control_mask_detail::is_negative(qubits, ::ket::meta::is_control_cvref<Qubits>{})
                               ? (UnsignedInteger{1u} << bit) : UnsignedInteger{0u}, ++bit, 0)...};
    return result;
  }

  template <typename UnsignedInteger, typename ControlRange>
  inline auto negative_control_mask_from_range(ControlRange const& controls, std::size_t const num_targets)
  -> UnsignedInteger
  {
    static_assert(std::is_unsigned<UnsignedInteger>::value, "Unsigned index required");
    auto result = UnsignedInteger{0u};
    auto bit = num_targets;
    for (auto const control : controls)
    {
      if (bit >= std::numeric_limits<UnsignedInteger>::digits)
        throw std::out_of_range{"Too many operands for packed index"};
      using ::ket::control_value;
      if (control_value(control) == 0u)
        result |= UnsignedInteger{1u} << bit;
      ++bit;
    }
    return result;
  }
} } }

#endif // KET_GATE_UTILITY_CONTROL_MASK_HPP
