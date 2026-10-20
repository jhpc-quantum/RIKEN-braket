#ifndef KET_DETAIL_CONTROL_IO_HPP
# define KET_DETAIL_CONTROL_IO_HPP

# include <cstdint>
# include <istream>
# include <ostream>
# include <sstream>

namespace ket
{
  namespace detail
  {
    namespace control_io
    {
      template <typename Character, typename CharacterTraits>
      inline auto write(
        std::basic_ostream<Character, CharacterTraits>& output_stream,
        std::uintmax_t const position, std::uintmax_t const value)
      -> std::basic_ostream<Character, CharacterTraits>&
      {
        // Width/fill apply to the entire operand, while numeric fields retain
        // the caller's locale and flags (including hexadecimal formatting).
        auto formatted = std::basic_ostringstream<Character, CharacterTraits>{};
        formatted.imbue(output_stream.getloc());
        formatted.flags(output_stream.flags());
        formatted.precision(output_stream.precision());
        formatted << position << formatted.widen('#') << value;
        if (not formatted)
        {
          output_stream.setstate(std::ios_base::badbit);
          return output_stream;
        }
        return output_stream << formatted.str();
      }

      template <typename Character, typename CharacterTraits>
      inline auto read_unsigned(
        std::basic_istream<Character, CharacterTraits>& input_stream, std::uintmax_t& integer) -> bool
      {
        using stream_type = std::basic_istream<Character, CharacterTraits>;
        typename stream_type::sentry const sentry{input_stream};
        if (not sentry)
          return false;
        // Unsigned formatted extraction accepts minus signs and can wrap a
        // negative field into an otherwise valid wire index or control value.
        if (CharacterTraits::eq_int_type(
              input_stream.peek(), CharacterTraits::to_int_type(input_stream.widen('-'))))
        {
          input_stream.setstate(std::ios_base::failbit);
          return false;
        }
        return static_cast<bool>(input_stream >> integer);
      }

      template <typename Character, typename CharacterTraits>
      inline auto read(
        std::basic_istream<Character, CharacterTraits>& input_stream,
        std::uintmax_t& position, std::uintmax_t& value) -> bool
      {
        auto separator = Character{};
        if (not ::ket::detail::control_io::read_unsigned(input_stream, position)
            or not (input_stream >> separator))
          return false;
        if (not CharacterTraits::eq(separator, input_stream.widen('#')))
        {
          input_stream.setstate(std::ios_base::failbit);
          return false;
        }
        return ::ket::detail::control_io::read_unsigned(input_stream, value);
      }
    } // namespace control_io
  } // namespace detail
} // namespace ket

#endif // KET_DETAIL_CONTROL_IO_HPP
