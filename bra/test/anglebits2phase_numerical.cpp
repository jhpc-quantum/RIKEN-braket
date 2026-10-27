// The probe is compiled without fast-math, independently of the evaluator.
#include <cfenv>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
# include <xmmintrin.h>
#endif
#include <boost/math/constants/constants.hpp>
#include <bra/classical_function_call.hpp>

namespace
{
  auto storage(bra::real_type const& value) -> std::string
  {
    unsigned char bytes[sizeof(value)];
    std::memcpy(bytes, &value, sizeof(value));
    auto result = std::ostringstream{};
    for (auto byte: bytes) result << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(byte);
    return result.str();
  }
}

int main()
{
  using limits = std::numeric_limits<bra::real_type>;
  std::cout << limits::digits << ' ' << limits::min_exponent << ' '
            << storage(boost::math::constants::two_pi<bra::real_type>()) << '\n';
  for (auto line = std::string{}; std::getline(std::cin, line);)
  {
    auto input = std::istringstream{line};
    auto mode = 0, flush = 0, width = 0;
    auto bits = bra::uint_type{0u};
    input >> mode >> flush >> bits >> width;
    int const modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    if (mode < 0 or mode >= 4 or std::fesetround(modes[mode]) != 0) return 2;
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
    _mm_setcsr((_mm_getcsr() & ~0x8040u) | (flush ? 0x8040u : 0u));
    auto const controls = _mm_getcsr() & 0xe040u;
#endif
    auto descriptors = std::vector<std::string>{};
    for (auto token = std::string{}; input >> token;) descriptors.push_back(token);
    auto result = bra::real_type{77};
    auto output = std::string{};
    try
    {
      auto const chain = bra::parse_anglebits2phase_chain(descriptors);
      auto const& signature = bra::resolve_classical_function(bra::classical_function_id::anglebits2phase,
        {bra::variable_type::unsigned_integer, bra::variable_type::integer});
      auto const call = bra::classical_function_call{signature, {&result}, {bits, width}, chain};
      call.commit_results(bra::evaluate_classical_function(call));
      output = "OK " + storage(result);
    }
    catch (std::exception const& error)
    {
      if (result != bra::real_type{77}) throw std::runtime_error{"failed call changed destination"};
      output = "ERR " + std::string{error.what()};
    }
    if (std::fegetround() != modes[mode]) throw std::runtime_error{"changed rounding mode"};
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
    if ((_mm_getcsr() & 0xe040u) != controls) throw std::runtime_error{"changed floating-point controls"};
#endif
    std::cout << output << '\n';
  }
}
