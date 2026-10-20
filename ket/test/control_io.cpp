// g++ -std=c++14 -Wall -Wextra -pedantic -Iket/include ket/test/control_io.cpp -o /tmp/control_io
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

// Intentionally omit qubit_io.hpp: control_io.hpp must be self-contained.
#include <ket/control_io.hpp>

namespace
{
  template <typename Character>
  auto text(std::string const& value) -> std::basic_string<Character>
  { return {value.begin(), value.end()}; }

  template <typename Control, typename Character>
  auto check_invalid(std::string const& input, bool const exceptions) -> void
  {
    using qubit = typename Control::qubit_type;
    auto destination = Control{qubit{7u}};
    auto stream = std::basic_istringstream<Character>{text<Character>(input)};
    if (exceptions)
      stream.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    auto threw = false;
    try { stream >> destination; }
    catch (std::ios_base::failure const&) { threw = true; }
    assert(stream.fail() and threw == exceptions);
    assert(destination.qubit() == qubit{7u});
    assert(ket::control_value(destination) == Control::value());
  }

  template <typename Control, typename Character>
  auto check_control() -> void
  {
    using qubit = typename Control::qubit_type;
    using bit_integer = typename qubit::bit_integer_type;
    using stream_type = std::basic_stringstream<Character>;
    auto const value = std::to_string(static_cast<std::uintmax_t>(Control::value()));
    auto const maximum = static_cast<std::uintmax_t>((std::numeric_limits<bit_integer>::max)());
    for (auto const position : {std::uintmax_t{0u}, std::uintmax_t{3u}, maximum})
    {
      auto stream = stream_type{};
      auto const source = Control{qubit{static_cast<bit_integer>(position)}};
      assert(&(stream << source) == static_cast<std::basic_ostream<Character>*>(&stream));
      assert(stream.str() == text<Character>(std::to_string(position) + "#" + value));
      auto destination = Control{qubit{7u}};
      assert(&(stream >> destination) == static_cast<std::basic_istream<Character>*>(&stream));
      assert(not stream.fail() and destination.qubit() == source.qubit());
    }
    {
      auto stream = stream_type{text<Character>("  5 # " + value + "  6#" + value)};
      auto first = Control{}, second = Control{};
      stream >> first >> second;
      assert(not stream.fail() and first.qubit() == qubit{5u} and second.qubit() == qubit{6u});
    }
    {
      auto stream = stream_type{text<Character>("3#" + value + ",rest")};
      auto destination = Control{};
      auto comma = Character{};
      auto rest = std::basic_string<Character>{};
      stream >> destination >> comma >> rest;
      assert(not stream.fail() and destination.qubit() == qubit{3u});
      assert(comma == stream.widen(',') and rest == text<Character>("rest"));
    }
    {
      auto stream = stream_type{};
      stream << std::hex << std::showbase << std::uppercase << Control{qubit{31u}};
      auto destination = Control{};
      stream >> std::hex >> destination;
      assert(not stream.fail() and destination.qubit() == qubit{31u});
      assert(stream.str() == text<Character>("0X1F#" + (Control::value() == 0u ? std::string{"0"} : std::string{"0X1"})));
    }
    {
      auto stream = stream_type{};
      stream << std::oct << Control{qubit{31u}};
      assert(stream.str() == text<Character>("37#" + value));
      auto destination = Control{};
      stream >> std::oct >> destination;
      assert(not stream.fail() and destination.qubit() == qubit{31u});
    }
    for (auto const left : {false, true})
    {
      auto stream = stream_type{};
      stream << std::setfill(stream.widen('_')) << (left ? std::left : std::right)
             << std::setw(7) << Control{qubit{3u}};
      auto const token = "3#" + value;
      assert(stream.str() == text<Character>(left ? token + "____" : "____" + token));
      assert(stream.width() == 0);
    }
    {
      auto destination = Control{qubit{7u}};
      auto stream = stream_type{text<Character>("3#" + value)};
      stream >> std::noskipws >> destination;
      assert(not stream.fail() and destination.qubit() == qubit{3u});
      stream.str(text<Character>(" 3#" + value)); stream.clear();
      stream >> destination;
      assert(stream.fail() and destination.qubit() == qubit{3u});
    }
    auto const mismatch = Control::value() == 0u ? "1" : "0";
    for (auto const exceptions : {false, true})
    {
      for (auto const& input : {std::string{}, std::string{"3"}, std::string{"3#"},
                               std::string{"#"} + value, "a#" + value, std::string{"3#a"},
                               "3:" + value, "3/" + value, "-1#" + value, "-0#" + value,
                               std::string{"3#-0"}, std::string{"3#-1"}, std::string{"3#2"},
                               std::string{"3#256"}, "3#" + std::string{mismatch},
                               "3#" + std::to_string((std::numeric_limits<std::uintmax_t>::max)()) + "0",
                               std::to_string((std::numeric_limits<std::uintmax_t>::max)()) + "0#" + value,
                               std::string{"3#-18446744073709551615"}})
        check_invalid<Control, Character>(input, exceptions);
      if (maximum < (std::numeric_limits<std::uintmax_t>::max)())
        check_invalid<Control, Character>(std::to_string(maximum + 1u) + "#" + value, exceptions);
    }
    {
      auto destination = Control{qubit{7u}};
      auto stream = stream_type{text<Character>("3#" + value)};
      stream.setstate(std::ios_base::badbit);
      stream >> destination;
      assert(stream.fail() and destination.qubit() == qubit{7u});
    }
    {
      // A valid numeric field ending at EOF can itself throw when eofbit is
      // in the exception mask. Do not commit before that extraction returns.
      auto destination = Control{qubit{7u}};
      auto stream = stream_type{text<Character>("3#" + value)};
      stream.exceptions(std::ios_base::eofbit);
      auto threw = false;
      try { stream >> destination; }
      catch (std::ios_base::failure const&) { threw = true; }
      assert(threw and destination.qubit() == qubit{7u});
    }
  }

  template <typename StateInteger, typename BitInteger>
  auto check_types() -> void
  {
    using qubit = ket::qubit<StateInteger, BitInteger>;
    check_control<ket::control<qubit, 0u>, char>();
    check_control<ket::control<qubit, 1u>, char>();
    check_control<ket::control<qubit, 0u>, wchar_t>();
    check_control<ket::control<qubit, 1u>, wchar_t>();
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
    using control = ket::control<qubit>;
    auto stream = std::basic_stringstream<Character>{};
    stream.imbue(std::locale{std::locale::classic(), new grouped_numbers<Character>});
    stream << control{qubit{1234u}};
    assert(stream.str() == text<Character>("1_234#1"));
    auto destination = control{};
    stream >> destination;
    assert(not stream.fail() and destination.qubit() == qubit{1234u});
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
  using qubit = ket::qubit<>;
  auto stream = std::stringstream{"3#0 4#1"};
  auto negative = ket::control<qubit, 0u>{};
  auto positive = ket::control<qubit>{};
  stream >> negative >> positive;
  assert(not stream.fail() and negative.qubit() == qubit{3u} and positive.qubit() == qubit{4u});
}
