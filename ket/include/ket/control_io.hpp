#ifndef KET_CONTROL_IO_HPP
# define KET_CONTROL_IO_HPP

# include <cstdint>
# include <istream>
# include <limits>
# include <ostream>

# include <ket/control.hpp>
# include <ket/detail/control_io.hpp>


namespace ket
{
  template <typename Character, typename CharacterTraits, typename Qubit, ::ket::meta::state_integer_t<Qubit> control_value>
  inline auto operator<<(
    std::basic_ostream<Character, CharacterTraits>& output_stream,
    ::ket::control<Qubit, control_value> const control_qubit)
  -> std::basic_ostream<Character, CharacterTraits>&
  {
    using bit_integer_type = typename Qubit::bit_integer_type;
    // Widen both integer fields before insertion, including byte-sized types.
    return ::ket::detail::control_io::write(
      output_stream, static_cast<std::uintmax_t>(static_cast<bit_integer_type>(control_qubit.qubit())),
      static_cast<std::uintmax_t>(control_value));
  }

  template <typename Character, typename CharacterTraits, typename Qubit, ::ket::meta::state_integer_t<Qubit> control_value>
  inline auto operator>>(
    std::basic_istream<Character, CharacterTraits>& input_stream,
    ::ket::control<Qubit, control_value>& control_qubit)
  -> std::basic_istream<Character, CharacterTraits>&
  {
    using bit_integer_type = typename Qubit::bit_integer_type;
    auto position = std::uintmax_t{};
    auto value = std::uintmax_t{};
    if (not ::ket::detail::control_io::read(input_stream, position, value))
      return input_stream;
    if (position > (std::numeric_limits<bit_integer_type>::max)()
        or value != static_cast<std::uintmax_t>(control_value))
    {
      input_stream.setstate(std::ios_base::failbit);
      return input_stream;
    }
    // Commit only after both fields have been checked, including under stream
    // exception masks. A static control's required value never changes.
    control_qubit = ::ket::control<Qubit, control_value>{Qubit{static_cast<bit_integer_type>(position)}};
    return input_stream;
  }
} // namespace ket


#endif // KET_CONTROL_IO_HPP
