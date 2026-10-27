// Implementation included by classical_function.cpp: no separate build target.
#ifndef BRA_SRC_ANGLEBITS2PHASE_DETAIL_HPP
# define BRA_SRC_ANGLEBITS2PHASE_DETAIL_HPP

# include <algorithm>
# include <cctype>
# include <memory>
# include <stdexcept>
# include <string>
# include <vector>
# include <boost/multiprecision/cpp_int.hpp>
# include <bra/anglebits2phase.hpp>

namespace bra
{
  namespace anglebits2phase_detail
  {
    using integer = boost::multiprecision::cpp_int;
    constexpr unsigned max_exact_bits = 65536u;
    constexpr std::size_t max_modifiers = 1024u;
    constexpr std::size_t max_literal_chars = 4096u;

    struct dyadic
    {
      integer numerator;
      unsigned denominator_power;
    };

    inline void normalize(dyadic& value)
    {
      if (value.numerator == 0) { value.denominator_power = 0u; return; }
      integer const magnitude = value.numerator < 0 ? -value.numerator : value.numerator;
      auto const shift = std::min(value.denominator_power, boost::multiprecision::lsb(magnitude));
      value.numerator /= integer{1} << shift;
      value.denominator_power -= shift;
    }

    inline void check_size(dyadic const& value)
    {
      integer const magnitude = value.numerator < 0 ? -value.numerator : value.numerator;
      if (value.denominator_power >= max_exact_bits
          or (magnitude != 0 and boost::multiprecision::msb(magnitude) >= max_exact_bits))
        throw std::length_error{"ANGLEBITS2PHASE exact arithmetic exceeds 65536 bits"};
    }

    inline void principal(dyadic& value)
    {
      integer const modulus = integer{1} << value.denominator_power;
      value.numerator %= modulus;
      if (value.numerator < 0) value.numerator += modulus;
      if ((value.numerator << 1) > modulus) value.numerator -= modulus;
      normalize(value);
      check_size(value);
    }

    // Round a positive exact rational to a binary format using integer work
    // only. The result may have a large integer part before normalization.
    inline auto rounded_binary(integer numerator, integer denominator, int const digits,
                               int const minimum_unit) -> dyadic
    {
      if (numerator == 0) return {0, 0u};
      auto exponent = static_cast<int>(boost::multiprecision::msb(numerator))
        - static_cast<int>(boost::multiprecision::msb(denominator));
      if (exponent >= 0 ? numerator < (denominator << exponent) : (numerator << -exponent) < denominator)
        --exponent;
      auto const unit = std::max(exponent - digits + 1, minimum_unit);
      if (unit >= 0) denominator <<= unit;
      else numerator <<= -unit;
      integer rounded = numerator / denominator;
      integer const twice_remainder = (numerator % denominator) << 1;
      if (twice_remainder > denominator or (twice_remainder == denominator and (rounded & 1) != 0))
        ++rounded;
      auto result = dyadic{rounded, unit < 0 ? static_cast<unsigned>(-unit) : 0u};
      if (unit > 0) result.numerator <<= unit;
      normalize(result);
      return result;
    }

    inline auto decimal_integer(std::string const& digits) -> integer
    {
      integer result = 0;
      for (auto const digit: digits) { result *= 10; result += digit - '0'; }
      return result;
    }

    inline auto decode_exponent(std::string const& token) -> dyadic
    {
      if (token.size() > max_literal_chars)
        throw std::length_error{"ANGLEBITS2PHASE exponent exceeds 4096 characters"};
      auto position = std::size_t{0u};
      auto negative = false;
      if (position < token.size() and (token[position] == '+' or token[position] == '-'))
        negative = token[position++] == '-';
      auto digits = std::string{};
      auto after_point = 0;
      auto point = false;
      while (position < token.size())
      {
        auto const character = token[position];
        if (character >= '0' and character <= '9')
        { digits += character; if (point) ++after_point; ++position; }
        else if (character == '.' and not point) { point = true; ++position; }
        else break;
      }
      auto scientific = false;
      auto decimal_exponent = 0;
      if (position < token.size() and (token[position] == 'E' or token[position] == 'e'))
      {
        scientific = true;
        ++position;
        auto exponent_negative = false;
        if (position < token.size() and (token[position] == '+' or token[position] == '-'))
          exponent_negative = token[position++] == '-';
        auto const start = position;
        while (position < token.size() and token[position] >= '0' and token[position] <= '9')
        {
          // Saturate only the decimal scale, after checking every character.
          decimal_exponent = std::min(100000, decimal_exponent * 10 + token[position++] - '0');
        }
        if (position == start) throw std::invalid_argument{"ANGLEBITS2PHASE invalid exponent literal: " + token};
        if (exponent_negative) decimal_exponent = -decimal_exponent;
      }
      if (digits.empty() or position != token.size())
        throw std::invalid_argument{"ANGLEBITS2PHASE invalid/nonfinite exponent literal: " + token};
      auto result = dyadic{0, 0u};
      if (not point and not scientific) result.numerator = decimal_integer(digits);
      else
      {
        auto const first = digits.find_first_not_of('0');
        if (first == std::string::npos) return result;
        decimal_exponent -= after_point;
        auto const decimal_size = decimal_exponent + static_cast<int>(digits.size() - first);
        if (decimal_size > 309) throw std::domain_error{"ANGLEBITS2PHASE nonfinite binary64 exponent"};
        if (decimal_size < -324) return result;
        auto numerator = decimal_integer(digits);
        integer denominator = 1;
        if (decimal_exponent > 0)
          for (auto index = 0; index < decimal_exponent; ++index) numerator *= 10;
        else
          for (auto index = 0; index < -decimal_exponent; ++index) denominator *= 10;
        result = rounded_binary(numerator, denominator, 53, -1074);
        if (result.numerator != 0
            and static_cast<int>(boost::multiprecision::msb(result.numerator))
                - static_cast<int>(result.denominator_power) > 1023)
          throw std::domain_error{"ANGLEBITS2PHASE nonfinite binary64 exponent"};
      }
      if (negative) result.numerator = -result.numerator;
      check_size(result);
      return result;
    }

    inline auto uppercase(std::string token) -> std::string
    {
      for (auto& character: token) character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
      return token;
    }
  }

  struct anglebits2phase_chain
  {
    // INV is multiplication by -1 followed by canonicalization. Keeping each
    // descriptor separate preserves the branch before every subsequent POW.
    std::vector<anglebits2phase_detail::dyadic> factors;
  };

  auto parse_anglebits2phase_chain(std::vector<std::string> const& descriptors) -> ::bra::anglebits2phase_chain_ptr
  {
    using namespace anglebits2phase_detail;
    auto chain = std::make_shared< ::bra::anglebits2phase_chain >();
    for (auto position = std::size_t{0u}; position < descriptors.size(); ++position)
    {
      if (chain->factors.size() >= max_modifiers)
        throw std::length_error{"ANGLEBITS2PHASE chain exceeds 1024 modifiers"};
      auto const name = uppercase(descriptors[position]);
      if (name == "INV") chain->factors.push_back({-1, 0u});
      else if (name == "POW")
      {
        if (++position == descriptors.size()) throw std::invalid_argument{"ANGLEBITS2PHASE POW requires an exponent"};
        chain->factors.push_back(decode_exponent(descriptors[position]));
      }
      else throw std::invalid_argument{"ANGLEBITS2PHASE unknown modifier: " + descriptors[position]};
    }
    return chain;
  }
}

#endif // BRA_SRC_ANGLEBITS2PHASE_DETAIL_HPP
