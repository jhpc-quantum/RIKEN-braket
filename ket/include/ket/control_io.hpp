#ifndef KET_CONTROL_IO_HPP
# define KET_CONTROL_IO_HPP

# include <cstdint>
# include <istream>
# include <limits>
# include <ostream>
# include <sstream>

# include <ket/control.hpp>


namespace ket
{
  template <typename Character, typename CharacterTraits, typename Qubit, ::ket::meta::state_integer_t<Qubit> control_value>
  inline auto operator<<(
    std::basic_ostream<Character, CharacterTraits>& output_stream,
    ::ket::control<Qubit, control_value> const control_qubit)
  -> std::basic_ostream<Character, CharacterTraits>&
  {
    using bit_integer_type = typename Qubit::bit_integer_type;
    // Format as one operand so width/fill apply to the complete wire#value.
    // Widen both integer fields before insertion, including byte-sized types.
    auto formatted = std::basic_ostringstream<Character, CharacterTraits>{};
    formatted.imbue(output_stream.getloc());
    formatted.flags(output_stream.flags());
    formatted.precision(output_stream.precision());
    formatted << static_cast<std::uintmax_t>(static_cast<bit_integer_type>(control_qubit.qubit()))
              << formatted.widen('#') << static_cast<std::uintmax_t>(control_value);
    if (not formatted)
    {
      output_stream.setstate(std::ios_base::badbit);
      return output_stream;
    }
    return output_stream << formatted.str();
  }

  template <typename Character, typename CharacterTraits, typename Qubit, ::ket::meta::state_integer_t<Qubit> control_value>
  inline auto operator>>(
    std::basic_istream<Character, CharacterTraits>& input_stream,
    ::ket::control<Qubit, control_value>& control_qubit)
  -> std::basic_istream<Character, CharacterTraits>&
  {
    using stream_type = std::basic_istream<Character, CharacterTraits>;
    using bit_integer_type = typename Qubit::bit_integer_type;
    auto const read_unsigned = [&input_stream](std::uintmax_t& integer) -> bool
    {
      typename stream_type::sentry const sentry{input_stream};
      if (not sentry)
        return false;
      // Formatted unsigned extraction otherwise accepts a minus sign and
      // can wrap negative input into a representable wire or control value.
      if (CharacterTraits::eq_int_type(
            input_stream.peek(), CharacterTraits::to_int_type(input_stream.widen('-'))))
      {
        input_stream.setstate(std::ios_base::failbit);
        return false;
      }
      return static_cast<bool>(input_stream >> integer);
    };

    auto position = std::uintmax_t{};
    auto value = std::uintmax_t{};
    auto separator = Character{};
    if (not read_unsigned(position) or not (input_stream >> separator))
      return input_stream;
    if (not CharacterTraits::eq(separator, input_stream.widen('#')))
    {
      input_stream.setstate(std::ios_base::failbit);
      return input_stream;
    }
    if (not read_unsigned(value))
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
