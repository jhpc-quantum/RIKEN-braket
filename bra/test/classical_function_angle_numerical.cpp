#define BOOST_MATH_DISABLE_STD_FPCLASSIFY
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
# include <xmmintrin.h>
#endif
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <bra/classical_function_call.hpp>

namespace
{
  using real = bra::real_type;
  using uint = bra::uint_type;
  using integer = boost::multiprecision::cpp_int;
  using rational = boost::multiprecision::cpp_rational;

  void require(bool const condition, char const* message)
  { if (not condition) throw std::runtime_error{message}; }

  auto invoke(real const value, int const width) -> uint
  {
    auto result = uint{77u};
    auto const& signature = bra::resolve_classical_function(
      bra::classical_function_id::anglebits, {bra::variable_type::real, bra::variable_type::integer});
    auto const call = bra::classical_function_call{signature, {&result}, {value, width}};
    try { call.commit_results(bra::evaluate_classical_function(call)); }
    catch (...) { require(result == 77u, "failed ANGLEBITS changed its destination"); throw; }
    return result;
  }

  auto reference(real const value, int const width) -> uint
  {
    // Boost's independent floating-to-rational constructor is exact. The
    // oracle is compiled without fast-math; no production decomposition code
    // is reused. Use floor division here, including for negative inputs.
    integer const modulus = integer{1} << width;
    rational const scaled = rational{value} * modulus
      / rational{boost::math::constants::two_pi<real>()};
    integer const numerator = boost::multiprecision::numerator(scaled);
    integer const denominator = boost::multiprecision::denominator(scaled);
    integer rounded = numerator / denominator;
    integer remainder = numerator % denominator;
    if (remainder < 0) { --rounded; remainder += denominator; }
    if (2 * remainder > denominator or (2 * remainder == denominator and rounded % 2 != 0)) ++rounded;
    rounded %= modulus;
    if (rounded < 0) rounded += modulus;
    return rounded.convert_to<uint>();
  }

  auto checked = 0u;
  void check(real const value, int const width)
  {
    auto const rounding = std::fegetround();
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
    auto const controls = _mm_getcsr() & 0xe040u;
#endif
    auto const actual = invoke(value, width);
    require(std::fegetround() == rounding, "ANGLEBITS changed rounding mode");
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
    require((_mm_getcsr() & 0xe040u) == controls, "ANGLEBITS changed floating-point controls");
#endif
    auto const expected = reference(value, width);
    if (actual != expected)
    {
      auto message = std::ostringstream{};
      message << "ANGLEBITS(" << std::hexfloat << value << ", " << width << ") = "
              << actual << ", expected " << expected;
      throw std::runtime_error{message.str()};
    }
    ++checked;
  }
}

int main()
{
  using limits = std::numeric_limits<real>;
  auto const width_limit = std::numeric_limits<uint>::digits;
  auto const turn = boost::math::constants::two_pi<real>();
  auto const infinity = limits::infinity();
  for (auto const mode: {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
  {
    require(std::fesetround(mode) == 0, "cannot select test rounding mode");
    for (auto width = 1; width <= width_limit; ++width)
    {
      for (auto value: {real{0}, -real{0}, real{1}, real{-1}, real{0.1}, real{-0.1},
                        turn, -turn, turn / real{2}, -turn / real{2},
                        limits::max(), -limits::max(), limits::min(), -limits::min(),
                        std::nextafter(turn, real{0}), std::nextafter(turn, infinity)})
        check(value, width);
      auto const tie = std::ldexp(turn, -width - 1);
      for (auto value: {tie, -tie, std::nextafter(tie, real{0}), std::nextafter(tie, infinity),
                        std::nextafter(-tie, real{0}), std::nextafter(-tie, -infinity),
                        real{3} * tie, real{5} * tie, turn - tie, -turn + tie})
        check(value, width);
      require(invoke(tie, width) == 0u and invoke(-tie, width) == 0u, "half-grid ties must round to even zero");
      // All supported native UINT widths here are much smaller than the
      // reciprocal subnormal scale. Both signs must quantize to zero, even
      // when release startup enables DAZ/FTZ. Do not feed DAZ-sensitive
      // subnormals into the independent rational constructor.
      if (width_limit < -limits::min_exponent)
      {
        require(invoke(limits::denorm_min(), width) == 0u, "positive subnormal angle");
        require(invoke(-limits::denorm_min(), width) == 0u, "negative subnormal angle");
      }
    }
  }
  require(std::fesetround(FE_TONEAREST) == 0, "cannot restore test rounding mode");
  for (auto const invalid: {0, -1, width_limit + 1})
  {
    auto rejected = false;
    try { invoke(real{1}, invalid); }
    catch (std::domain_error const&) { rejected = true; }
    require(rejected, "invalid ANGLEBITS width accepted");
  }
  for (auto const invalid: {infinity, -infinity, limits::quiet_NaN()})
  {
    auto rejected = false;
    try { invoke(invalid, 4); }
    catch (std::domain_error const&) { rejected = true; }
    require(rejected, "nonfinite ANGLEBITS input accepted");
  }
  std::cout << checked << " exact ANGLEBITS comparisons passed\n";
}
