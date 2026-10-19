// g++ -std=c++14 -Wall -Wextra -pedantic -Iket/include ket/test/runtime_control_io.cpp -o /tmp/runtime_control_io
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

// Intentionally do not include qubit_io.hpp: this header is self-contained.
#include <ket/runtime_control_io.hpp>

namespace
{
  using qubit = ket::qubit<>;
  using control = ket::runtime_control<qubit>;

  auto check_invalid(std::string const& text) -> void
  {
    auto destination = control{qubit{7u}, 1};
    auto stream = std::istringstream{text};
    stream >> destination;
    assert(stream.fail());
    assert(destination.qubit() == qubit{7u} and destination.value() == 1u);
  }
}

int main()
{
  for (auto const value : {0u, 1u})
  {
    auto stream = std::stringstream{};
    auto const source = control{qubit{3u}, value};
    stream << source;
    assert(stream.str() == "3:" + std::to_string(value));
    auto destination = control{};
    stream >> destination;
    assert(not stream.fail());
    assert(destination.qubit() == source.qubit() and destination.value() == value);
  }
  {
    auto stream = std::stringstream{};
    stream << control{qubit{3u}, 0} << ' ' << control{qubit{4u}, 1};
    auto first = control{}, second = control{};
    stream >> first >> second;
    assert(not stream.fail());
    assert(first.qubit() == qubit{3u} and first.value() == 0u);
    assert(second.qubit() == qubit{4u} and second.value() == 1u);
  }
  {
    auto stream = std::wstringstream{L"  5 : 0 "};
    auto destination = control{};
    stream >> destination;
    assert(not stream.fail());
    stream.str(L""); stream.clear();
    stream << destination;
    assert(stream.str() == L"5:0");
  }
  {
    using small_qubit = ket::qubit<unsigned char, unsigned char>;
    using small_control = ket::runtime_control<small_qubit>;
    auto stream = std::stringstream{};
    stream << small_control{small_qubit{255u}, 0};
    assert(stream.str() == "255:0");
    auto destination = small_control{};
    stream >> destination;
    assert(not stream.fail());
    assert(destination.qubit() == small_qubit{255u} and destination.value() == 0u);
    for (auto const text : {"256:0", "3:256", "-1:0", "3:-256"})
    {
      stream.str(text); stream.clear();
      stream >> destination;
      assert(stream.fail());
      assert(destination.qubit() == small_qubit{255u} and destination.value() == 0u);
    }
  }
  {
    auto const position = (std::numeric_limits<unsigned int>::max)();
    auto stream = std::stringstream{};
    stream << control{qubit{position}, 1};
    auto destination = control{};
    stream >> destination;
    assert(not stream.fail() and destination.qubit() == qubit{position} and destination.value() == 1u);
  }
  for (auto const text : {"", "3", "3:", ":0", "a:0", "3:a", "3/0",
                          "-1:0", "3:-1", "3:2", "3:18446744073709551616",
                          "18446744073709551616:0"})
    check_invalid(text);
  check_invalid(std::to_string(std::uintmax_t{(std::numeric_limits<unsigned int>::max)()} + 1u) + ":0");
  {
    auto destination = control{qubit{7u}, 0};
    auto stream = std::istringstream{"3:2"};
    stream.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    auto threw = false;
    try { stream >> destination; }
    catch (std::ios_base::failure const&) { threw = true; }
    assert(threw and destination.qubit() == qubit{7u} and destination.value() == 0u);
  }
  {
    auto stream = std::stringstream{};
    stream << std::hex << control{qubit{31u}, 0};
    assert(stream.str() == "1f:0");
    auto destination = control{};
    stream >> std::hex >> destination;
    assert(not stream.fail() and destination.qubit() == qubit{31u} and destination.value() == 0u);
  }
}
