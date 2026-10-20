// g++ -std=c++14 -Wall -Wextra -pedantic -Iket/include ket/test/runtime_control_io.cpp -o /tmp/runtime_control_io
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

// Intentionally omit qubit_io.hpp: both control headers are self-contained.
#include <ket/runtime_control_io.hpp>
#include <ket/control_io.hpp>

namespace
{
  template <typename Character>
  auto text(std::string const& value) -> std::basic_string<Character>
  { return {value.begin(), value.end()}; }

  template <typename Qubit, typename Character>
  auto check_invalid(std::string const& input) -> void
  {
    using control = ket::runtime_control<Qubit>;
    for (auto const value : {0u, 1u})
      for (auto const exceptions : {false, true})
      {
        auto destination = control{Qubit{7u}, value};
        auto stream = std::basic_istringstream<Character>{text<Character>(input)};
        if (exceptions)
          stream.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        auto threw = false;
        try { stream >> destination; }
        catch (std::ios_base::failure const&) { threw = true; }
        assert(stream.fail() and threw == exceptions);
        assert(destination.qubit() == Qubit{7u} and destination.value() == value);
      }
  }

  template <typename Qubit, typename Character>
  auto check_qubit() -> void
  {
    using control = ket::runtime_control<Qubit>;
    using bit_integer = typename Qubit::bit_integer_type;
    using stream_type = std::basic_stringstream<Character>;
    auto const maximum = static_cast<std::uintmax_t>((std::numeric_limits<bit_integer>::max)());
    for (auto const value : {0u, 1u})
    {
      auto const suffix = "#" + std::to_string(value);
      for (auto const position : {std::uintmax_t{0u}, std::uintmax_t{3u}, maximum})
      {
        auto stream = stream_type{};
        auto const source = control{Qubit{static_cast<bit_integer>(position)}, value};
        assert(&(stream << source) == static_cast<std::basic_ostream<Character>*>(&stream));
        assert(stream.str() == text<Character>(std::to_string(position) + suffix));
        // A successful read must update both the position and the polarity.
        auto destination = control{Qubit{7u}, 1u - value};
        assert(&(stream >> destination) == static_cast<std::basic_istream<Character>*>(&stream));
        assert(not stream.fail() and destination.qubit() == source.qubit() and destination.value() == value);
      }
      {
        auto stream = stream_type{text<Character>("  5 # " + std::to_string(value) + " ")};
        auto destination = control{};
        stream >> destination;
        assert(not stream.fail() and destination.qubit() == Qubit{5u} and destination.value() == value);
      }
      {
        auto stream = stream_type{text<Character>("3" + suffix + ",rest")};
        auto destination = control{};
        auto comma = Character{};
        auto rest = std::basic_string<Character>{};
        stream >> destination >> comma >> rest;
        assert(not stream.fail() and destination.qubit() == Qubit{3u} and destination.value() == value);
        assert(comma == stream.widen(',') and rest == text<Character>("rest"));
      }
      {
        auto stream = stream_type{};
        stream << std::hex << std::uppercase << std::showbase << control{Qubit{31u}, value};
        assert(stream.str() == text<Character>(value == 0u ? "0X1F#0" : "0X1F#0X1"));
        auto destination = control{};
        stream >> std::hex >> destination;
        assert(not stream.fail() and destination.qubit() == Qubit{31u} and destination.value() == value);
      }
      {
        auto stream = stream_type{};
        stream << std::oct << control{Qubit{31u}, value};
        assert(stream.str() == text<Character>("37" + suffix));
        auto destination = control{};
        stream >> std::oct >> destination;
        assert(not stream.fail() and destination.qubit() == Qubit{31u} and destination.value() == value);
      }
      for (auto const left : {false, true})
      {
        auto stream = stream_type{};
        stream << std::setfill(stream.widen('_')) << (left ? std::left : std::right)
               << std::setw(7) << control{Qubit{3u}, value};
        auto const token = "3" + suffix;
        assert(stream.str() == text<Character>(left ? token + "____" : "____" + token));
        assert(stream.width() == 0);
      }
      {
        auto destination = control{Qubit{7u}, 1u - value};
        auto stream = stream_type{text<Character>("3" + suffix)};
        stream >> std::noskipws >> destination;
        assert(not stream.fail() and destination.qubit() == Qubit{3u} and destination.value() == value);
        stream.str(text<Character>(" 5" + suffix)); stream.clear();
        stream >> destination;
        assert(stream.fail() and destination.qubit() == Qubit{3u} and destination.value() == value);
      }
      for (auto const state : {std::ios_base::badbit, std::ios_base::failbit, std::ios_base::eofbit})
      {
        auto destination = control{Qubit{7u}, 1u - value};
        auto stream = stream_type{text<Character>("3" + suffix)};
        stream.setstate(state);
        stream >> destination;
        assert(stream.fail() and destination.qubit() == Qubit{7u} and destination.value() == 1u - value);
      }
      {
        auto destination = control{Qubit{7u}, 1u - value};
        auto stream = stream_type{text<Character>("3" + suffix)};
        stream.exceptions(std::ios_base::eofbit);
        auto threw = false;
        try { stream >> destination; }
        catch (std::ios_base::failure const&) { threw = true; }
        assert(threw and destination.qubit() == Qubit{7u} and destination.value() == 1u - value);
      }
    }
    {
      auto stream = stream_type{};
      stream << control{Qubit{3u}, 0u} << stream.widen(' ') << control{Qubit{4u}, 1u};
      auto first = control{}, second = control{};
      stream >> first >> second;
      assert(not stream.fail() and first.qubit() == Qubit{3u} and first.value() == 0u);
      assert(second.qubit() == Qubit{4u} and second.value() == 1u);
    }
    for (auto const& input : {
           std::string{}, std::string{"3"}, std::string{"3#"}, std::string{"#0"},
           std::string{"a#0"}, std::string{"3#a"}, std::string{"3/0"},
           std::string{"3:0"}, std::string{"3:1"}, std::string{"-1#0"}, std::string{"-0#1"},
           std::string{"3#-0"}, std::string{"3#-1"}, std::string{"3#2"},
           std::string{"3#256"}, std::string{"3#-256"}, std::string{"3#-18446744073709551615"},
           std::to_string((std::numeric_limits<std::uintmax_t>::max)()) + "0#0",
           "3#" + std::to_string((std::numeric_limits<std::uintmax_t>::max)()) + "0"})
      check_invalid<Qubit, Character>(input);
    if (maximum < (std::numeric_limits<std::uintmax_t>::max)())
      check_invalid<Qubit, Character>(std::to_string(maximum + 1u) + "#0");
  }

  template <typename Qubit, typename Character, typename Qubit::state_integer_type value>
  auto check_interoperability() -> void
  {
    using fixed_control = ket::control<Qubit, value>;
    using runtime_control = ket::runtime_control<Qubit>;
    using stream_type = std::basic_stringstream<Character>;
    auto fixed_stream = stream_type{}, runtime_stream = stream_type{};
    fixed_stream << fixed_control{Qubit{3u}};
    runtime_stream << runtime_control{Qubit{3u}, value};
    assert(fixed_stream.str() == runtime_stream.str());
    auto runtime_destination = runtime_control{Qubit{7u}, 1u - value};
    fixed_stream >> runtime_destination;
    assert(not fixed_stream.fail() and runtime_destination.qubit() == Qubit{3u}
           and runtime_destination.value() == value);
    auto fixed_destination = fixed_control{Qubit{7u}};
    runtime_stream >> fixed_destination;
    assert(not runtime_stream.fail() and fixed_destination.qubit() == Qubit{3u});
    runtime_stream.str(text<Character>("5#" + std::to_string(1u - value))); runtime_stream.clear();
    runtime_stream >> fixed_destination;
    assert(runtime_stream.fail() and fixed_destination.qubit() == Qubit{3u});
  }

  template <typename StateInteger, typename BitInteger>
  auto check_types() -> void
  {
    using qubit = ket::qubit<StateInteger, BitInteger>;
    check_qubit<qubit, char>();
    check_qubit<qubit, wchar_t>();
    check_interoperability<qubit, char, 0u>();
    check_interoperability<qubit, char, 1u>();
    check_interoperability<qubit, wchar_t, 0u>();
    check_interoperability<qubit, wchar_t, 1u>();
  }

  template <typename Character>
  struct grouped_numbers : std::numpunct<Character>
  {
    auto do_thousands_sep() const -> Character override { return static_cast<Character>('_'); }
    auto do_grouping() const -> std::string override { return "\3"; }
  };

  template <typename Character>
  auto check_locale() -> void
  {
    using qubit = ket::qubit<>;
    using control = ket::runtime_control<qubit>;
    auto stream = std::basic_stringstream<Character>{};
    stream.imbue(std::locale{std::locale::classic(), new grouped_numbers<Character>});
    stream << control{qubit{1234u}, 0u};
    assert(stream.str() == text<Character>("1_234#0"));
    auto destination = control{};
    stream >> destination;
    assert(not stream.fail() and destination.qubit() == qubit{1234u} and destination.value() == 0u);
  }

  template <typename Character>
  struct failing_output : std::basic_streambuf<Character>
  {
    auto xsputn(Character const*, std::streamsize) -> std::streamsize override { return 0; }
    auto overflow(typename std::char_traits<Character>::int_type)
      -> typename std::char_traits<Character>::int_type override
    { return std::char_traits<Character>::eof(); }
  };

  template <typename Character>
  auto check_output_failure() -> void
  {
    using qubit = ket::qubit<>;
    for (auto const exceptions : {false, true})
      for (auto const fixed : {false, true})
      {
        auto buffer = failing_output<Character>{};
        std::basic_ostream<Character> stream{&buffer};
        if (exceptions)
          stream.exceptions(std::ios_base::badbit);
        auto threw = false;
        try
        {
          if (fixed)
            stream << ket::control<qubit, 0u>{qubit{3u}};
          else
            stream << ket::runtime_control<qubit>{qubit{3u}, 0u};
        }
        catch (std::ios_base::failure const&) { threw = true; }
        assert(stream.bad() and threw == exceptions);
      }
  }
}

int main()
{
  check_types<unsigned char, unsigned char>();
  check_types<unsigned short, unsigned short>();
  check_types<unsigned int, unsigned int>();
  check_types<unsigned long, unsigned long>();
  check_types<unsigned long long, unsigned long long>();
  check_types<std::uint64_t, std::uint8_t>();
  check_types<std::uint64_t, std::uint16_t>();
  check_types<std::uint64_t, std::uint32_t>();
  check_types<std::uint64_t, std::uint64_t>();
  check_locale<char>();
  check_locale<wchar_t>();
  check_output_failure<char>();
  check_output_failure<wchar_t>();
}
