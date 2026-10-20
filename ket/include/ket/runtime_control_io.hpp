#ifndef KET_RUNTIME_CONTROL_IO_HPP
# define KET_RUNTIME_CONTROL_IO_HPP

# include <cstdint>
# include <istream>
# include <limits>
# include <ostream>

# include <ket/runtime_control.hpp>
# include <ket/detail/control_io.hpp>

namespace ket
{
  // Both fields are numeric, including for character-sized integer types.
  template <typename Character, typename CharacterTraits, typename Qubit>
  inline auto operator<<(
    std::basic_ostream<Character, CharacterTraits>& output_stream,
    ::ket::runtime_control<Qubit> const& control_qubit)
  -> std::basic_ostream<Character, CharacterTraits>&
  {
    using bit_integer_type = typename Qubit::bit_integer_type;
    return ::ket::detail::control_io::write(
      output_stream, static_cast<std::uintmax_t>(static_cast<bit_integer_type>(control_qubit.qubit())),
      static_cast<std::uintmax_t>(::ket::control_value(control_qubit)));
  }

  template <typename Character, typename CharacterTraits, typename Qubit>
  inline auto operator>>(
    std::basic_istream<Character, CharacterTraits>& input_stream,
    ::ket::runtime_control<Qubit>& control_qubit)
  -> std::basic_istream<Character, CharacterTraits>&
  {
    using bit_integer_type = typename Qubit::bit_integer_type;
    auto position = std::uintmax_t{};
    auto value = std::uintmax_t{};
    if (not ::ket::detail::control_io::read(input_stream, position, value))
      return input_stream;
    if (position > (std::numeric_limits<bit_integer_type>::max)()
        or value >= ::ket::meta::base_v<Qubit>)
    {
      input_stream.setstate(std::ios_base::failbit);
      return input_stream;
    }
    // Commit only after both fields have been read and validated. This also
    // preserves the destination if setstate throws under an exception mask.
    control_qubit = ::ket::runtime_control<Qubit>{Qubit{static_cast<bit_integer_type>(position)}, value};
    return input_stream;
  }
}

#endif // KET_RUNTIME_CONTROL_IO_HPP
