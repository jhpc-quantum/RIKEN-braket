#ifndef BRA_UTILITY_TO_INTEGER_HPP
# define BRA_UTILITY_TO_INTEGER_HPP

# include <string>


namespace bra
{
  namespace utility
  {
    // Use std::from_chars and std::string_view instead if C++17 is available
    template <typename Character, typename CharacterTraits>
    int to_int(std::basic_string<Character, CharacterTraits> const& string)
    { return std::stoi(string); }

    template <typename Character, typename CharacterTraits>
    long to_long(std::basic_string<Character, CharacterTraits> const& string)
    { return std::stol(string); }

    template <typename Character, typename CharacterTraits>
    long long to_long_long(std::basic_string<Character, CharacterTraits> const& string)
    { return std::stoll(string); }

    template <typename Character, typename CharacterTraits>
    unsigned long to_unsigned_long(std::basic_string<Character, CharacterTraits> const& string)
    { return std::stoul(string); }

    template <typename Character, typename CharacterTraits>
    unsigned long long to_unsigned_long_long(std::basic_string<Character, CharacterTraits> const& string)
    { return std::stoull(string); }

    namespace to_integer_detail
    {
      template <typename Integer>
      struct to_integer;

      template <>
      struct to_integer<int>
      {
        template <typename Character, typename CharacterTraits>
        static int call(std::basic_string<Character, CharacterTraits> const& string)
        { return to_int(string); }
      }; // struct to_integer<int>

      template <>
      struct to_integer<long>
      {
        template <typename Character, typename CharacterTraits>
        static long call(std::basic_string<Character, CharacterTraits> const& string)
        { return to_long(string); }
      }; // struct to_integer<long>

      template <>
      struct to_integer<long long>
      {
        template <typename Character, typename CharacterTraits>
        static long long call(std::basic_string<Character, CharacterTraits> const& string)
        { return to_long_long(string); }
      }; // struct to_integer<long long>

      template <>
      struct to_integer<unsigned long>
      {
        template <typename Character, typename CharacterTraits>
        static unsigned long call(std::basic_string<Character, CharacterTraits> const& string)
        { return to_unsigned_long(string); }
      }; // struct to_integer<unsigned long>

      template <>
      struct to_integer<unsigned long long>
      {
        template <typename Character, typename CharacterTraits>
        static unsigned long long call(std::basic_string<Character, CharacterTraits> const& string)
        { return to_unsigned_long_long(string); }
      }; // struct to_integer<unsigned long long>
    } // namespace to_integer_detail

    template <typename Integer, typename Character, typename CharacterTraits>
    Integer to_integer(std::basic_string<Character, CharacterTraits> const& string)
    { return ::bra::utility::to_integer_detail::to_integer<Integer>::call(string); }

    template <typename Integer, typename InputIterator>
    Integer to_integer(InputIterator const first, InputIterator const last)
    { return ::bra::utility::to_integer<Integer>(std::string{first, last}); }
  } // namespace utility
} // namespace bra


#endif // BRA_UTILITY_TO_INTEGER_HPP
