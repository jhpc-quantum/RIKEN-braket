#ifndef BRA_UTILITY_TO_INTEGER_HPP
# define BRA_UTILITY_TO_INTEGER_HPP

# include <string>

# include <boost/utility/string_view.hpp>


namespace bra
{
  namespace utility
  {
    // Use std::from_chars and std::string_view instead if C++17 is available
    template <typename Character, typename CharacterTraits>
    int to_int(boost::basic_string_view<Character, CharacterTraits> const str)
    {
      auto const integer_string = std::basic_string<Character, CharacterTraits>{str};
      return std::stoi(integer_string);
    }

    template <typename Character, typename CharacterTraits>
    long to_long(boost::basic_string_view<Character, CharacterTraits> const str)
    {
      auto const integer_string = std::basic_string<Character, CharacterTraits>{str};
      return std::stol(integer_string);
    }

    template <typename Character, typename CharacterTraits>
    long long to_long_long(boost::basic_string_view<Character, CharacterTraits> const str)
    {
      auto const integer_string = std::basic_string<Character, CharacterTraits>{str};
      return std::stoll(integer_string);
    }

    template <typename Character, typename CharacterTraits>
    unsigned long to_unsigned_long(boost::basic_string_view<Character, CharacterTraits> const str)
    {
      auto const integer_string = std::basic_string<Character, CharacterTraits>{str};
      return std::stoul(integer_string);
    }

    template <typename Character, typename CharacterTraits>
    unsigned long long to_unsigned_long_long(boost::basic_string_view<Character, CharacterTraits> const str)
    {
      auto const integer_string = std::basic_string<Character, CharacterTraits>{str};
      return std::stoull(integer_string);
    }

    namespace to_integer_detail
    {
      template <typename Integer>
      struct to_integer;

      template <>
      struct to_integer<int>
      {
        template <typename Character, typename CharacterTraits>
        static int call(boost::basic_string_view<Character, CharacterTraits> const str)
        { return to_int(str); }
      }; // struct to_integer<int>

      template <>
      struct to_integer<long>
      {
        template <typename Character, typename CharacterTraits>
        static long call(boost::basic_string_view<Character, CharacterTraits> const str)
        { return to_long(str); }
      }; // struct to_integer<long>

      template <>
      struct to_integer<long long>
      {
        template <typename Character, typename CharacterTraits>
        static long long call(boost::basic_string_view<Character, CharacterTraits> const str)
        { return to_long_long(str); }
      }; // struct to_integer<long long>

      template <>
      struct to_integer<unsigned long>
      {
        template <typename Character, typename CharacterTraits>
        static unsigned long call(boost::basic_string_view<Character, CharacterTraits> const str)
        { return to_unsigned_long(str); }
      }; // struct to_integer<unsigned long>

      template <>
      struct to_integer<unsigned long long>
      {
        template <typename Character, typename CharacterTraits>
        static unsigned long long call(boost::basic_string_view<Character, CharacterTraits> const str)
        { return to_unsigned_long_long(str); }
      }; // struct to_integer<unsigned long long>
    } // namespace to_integer_detail

    template <typename Integer, typename Character, typename CharacterTraits>
    Integer to_integer(boost::basic_string_view<Character, CharacterTraits> const str)
    { return ::bra::utility::to_integer_detail::to_integer<Integer>::call(str); }
  } // namespace utility
} // namespace bra


#endif // BRA_UTILITY_TO_INTEGER_HPP
