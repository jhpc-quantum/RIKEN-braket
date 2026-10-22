// Standalone: compile with src/classical_function.cpp and BRA_NO_MPI, using
// the same BRA_REAL_TYPE for both translation units. No simulator objects.
// Inspect stored representations even when the test oracle is compiled without
// fast-math but runs in a process with DAZ enabled by release startup.
#define BOOST_MATH_DISABLE_STD_FPCLASSIFY
#include <cfenv>
#include <cmath>
#include <complex>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
# include <xmmintrin.h>
#endif

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/sign.hpp>

#include <ket/utility/exp_i.hpp>
#include <bra/classical_function_call.hpp>

namespace
{
  using real = bra::real_type;
  using complex = bra::complex_type;
  using function = bra::classical_function_id;
  using type = bra::variable_type;

  void require(bool const condition, std::string const& reason)
  { if (not condition) throw std::runtime_error{reason}; }

  bool close(real const actual, real const expected)
  {
    return boost::math::isfinite(actual)
      and std::abs(actual - expected) <= real{32} * std::numeric_limits<real>::epsilon()
                                       * (real{1} + std::abs(expected));
  }

  auto real_mod(real const dividend, real const divisor) -> real
  {
    auto result = real{-77};
    auto const& signature = bra::resolve_classical_function(function::mod, {type::real, type::real});
    auto const call = bra::classical_function_call{signature, {&result}, {dividend, divisor}};
    call.commit_results(bra::evaluate_classical_function(call));
    return result;
  }

  auto real_exp(real const input) -> real
  {
    auto result = real{-77};
    auto const& signature = bra::resolve_classical_function(function::exp, {type::real});
    auto const call = bra::classical_function_call{signature, {&result}, {input}};
    try { call.commit_results(bra::evaluate_classical_function(call)); }
    catch (std::exception const& error)
    {
      auto diagnostic = std::ostringstream{};
      diagnostic << error.what() << " for EXP(" << std::hexfloat << input << ')';
      throw std::runtime_error{diagnostic.str()};
    }
    return result;
  }

  auto complex_exp(complex const input) -> complex
  {
    auto result = complex{-77, 19};
    auto const& signature = bra::resolve_classical_function(function::exp, {type::complex_});
    auto const call = bra::classical_function_call{signature, {&result}, {input}};
    try { call.commit_results(bra::evaluate_classical_function(call)); }
    catch (std::exception const& error)
    {
      auto diagnostic = std::ostringstream{};
      diagnostic << error.what() << " for complex EXP(" << std::hexfloat << input.real() << ", " << input.imag() << ')';
      throw std::runtime_error{diagnostic.str()};
    }
    return result;
  }

  auto argument(complex const input) -> real
  {
    auto result = real{-77};
    auto const& signature = bra::resolve_classical_function(function::arg, {type::complex_});
    auto const call = bra::classical_function_call{signature, {&result}, {input}};
    try { call.commit_results(bra::evaluate_classical_function(call)); }
    catch (std::exception const& error)
    {
      auto diagnostic = std::ostringstream{};
      diagnostic << error.what() << " for ARG(" << std::hexfloat << input.real() << ", " << input.imag()
                 << "), classes " << boost::math::fpclassify(input.real()) << ", " << boost::math::fpclassify(input.imag());
      throw std::runtime_error{diagnostic.str()};
    }
    return result;
  }

  auto magnitude(complex const input) -> real
  {
    auto result = real{-77};
    auto const& signature = bra::resolve_classical_function(function::abs, {type::complex_});
    auto const call = bra::classical_function_call{signature, {&result}, {input}};
    call.commit_results(bra::evaluate_classical_function(call));
    require(boost::math::isfinite(result) and not boost::math::signbit(result),
            "ABS result must be finite and nonnegative, including positive zero");
    return result;
  }

  template <typename T>
  auto square_root(T const input) -> T
  {
    auto result = T{-77};
    auto const operand_type = bra::classical_function_detail::value_type(bra::classical_function_value{input}.which());
    auto const& signature = bra::resolve_classical_function(function::sqrt, {operand_type});
    auto const call = bra::classical_function_call{signature, {&result}, {input}};
    call.commit_results(bra::evaluate_classical_function(call));
    return result;
  }

  auto trigonometric(function const id, real const input) -> real
  {
    auto result = real{-77};
    auto const& signature = bra::resolve_classical_function(id, {type::real});
    auto const call = bra::classical_function_call{signature, {&result}, {input}};
    try { call.commit_results(bra::evaluate_classical_function(call)); }
    catch (std::exception const& error)
    {
      auto diagnostic = std::ostringstream{};
      diagnostic << error.what() << " for " << bra::classical_function_name(id)
                 << '(' << std::hexfloat << input << ')';
      throw std::runtime_error{diagnostic.str()};
    }
    return result;
  }

  auto reference_trigonometric(function const id, real const input) -> real
  {
    // This translation unit is compiled independently, without fast-math.
    switch (id)
    {
    case function::sin: return std::sin(input);
    case function::cos: return std::cos(input);
    case function::tan: return std::tan(input);
    case function::arcsin: return std::asin(input);
    case function::arccos: return std::acos(input);
    case function::arctan: return std::atan(input);
    default: throw std::invalid_argument{"not a trigonometric function"};
    }
  }

  auto rounded(function const id, real const input) -> real
  {
    auto result = real{-77};
    auto const& signature = bra::resolve_classical_function(id, {type::real});
    auto const call = bra::classical_function_call{signature, {&result}, {input}};
    call.commit_results(bra::evaluate_classical_function(call));
    return result;
  }

  auto natural_log(real const input) -> real
  {
    auto result = real{-77};
    auto const& signature = bra::resolve_classical_function(function::log, {type::real});
    auto const call = bra::classical_function_call{signature, {&result}, {input}};
    call.commit_results(bra::evaluate_classical_function(call));
    return result;
  }

  template <typename Exception, typename T>
  void expect_failure(function const id, T const input, std::string const& reason)
  {
    auto result = real{-77};
    auto complex_result = complex{-77, 19};
    auto const operand_type = bra::classical_function_detail::value_type(bra::classical_function_value{input}.which());
    auto const& signature = bra::resolve_classical_function(id, {operand_type});
    auto const destination = signature.result_types.front() == type::real
      ? bra::classical_function_destination{&result} : bra::classical_function_destination{&complex_result};
    auto const call = bra::classical_function_call{signature, {destination}, {input}};
    auto caught = false;
    try { call.commit_results(bra::evaluate_classical_function(call)); }
    catch (Exception const& error)
    {
      require(std::string{error.what()}.find(reason) != std::string::npos, "wrong numerical error diagnostic");
      caught = true;
    }
    require(caught, "missing numerical error: " + reason);
    require(result == -77 and complex_result == complex{-77, 19}, "numerical error changed destination");
  }
}

int main()
{
  auto const rounding = std::fegetround();
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
  auto const initial_control = _mm_getcsr();
  // Ignore accrued exception-status bits; FTZ, DAZ, rounding, and masks must
  // remain exactly as selected by the caller/build's startup environment.
  auto const control_mode = initial_control & ~0x3fu;
#endif
  for (auto const dividend: {real{-7.5}, real{-6}, real{-0.0}, real{0.0}, real{6}, real{7.5},
                              std::numeric_limits<real>::max(), std::numeric_limits<real>::min()})
    for (auto const divisor: {real{-3}, real{-1}, real{1}, real{3},
                             std::numeric_limits<real>::max(), std::numeric_limits<real>::min()})
    {
      auto const expected = std::fmod(dividend, divisor);
      auto const result = real_mod(dividend, divisor);
      require(result == expected and boost::math::signbit(result) == boost::math::signbit(expected),
              "MOD changed remainder or zero sign");
    }
  for (auto const dividend: {real{1}, std::numeric_limits<real>::infinity(),
                             -std::numeric_limits<real>::infinity(), std::numeric_limits<real>::quiet_NaN()})
    for (auto const divisor: {real{3}, real{0}, real{-0.0}, std::numeric_limits<real>::infinity(),
                              -std::numeric_limits<real>::infinity(), std::numeric_limits<real>::quiet_NaN()})
    {
      if (dividend == real{1} and divisor == real{3}) continue;
      auto result = real{-77};
      auto const& signature = bra::resolve_classical_function(function::mod, {type::real, type::real});
      auto const call = bra::classical_function_call{signature, {&result}, {dividend, divisor}};
      auto rejected = false;
      try { call.commit_results(bra::evaluate_classical_function(call)); }
      catch (std::domain_error const&) { rejected = true; }
      require(rejected and result == real{-77}, "MOD accepted invalid input or changed destination");
    }
  // Audit subnormal operands separately: the build's arithmetic environment
  // may flush a subnormal result, but a stored nonzero divisor is not zero.
  if (std::numeric_limits<real>::has_denorm == std::denorm_present)
  {
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
    // The oracle alone uses gradual underflow, then restores the selected
    // environment before calling the evaluator. Production never changes it.
    _mm_setcsr(initial_control & ~0x8040u);
#endif
    auto const tiny = std::numeric_limits<real>::denorm_min();
    auto const near_normal = std::nextafter(std::numeric_limits<real>::min(), real{0});
    auto const values = std::vector<real>{tiny, tiny * 3, tiny * 7, near_normal, -tiny, -tiny * 3, -near_normal, real{1},
                                        std::numeric_limits<real>::max()};
    for (auto const dividend: values)
      for (auto const divisor: values)
        for (auto const negative: {false, true})
        {
          auto const input = negative ? -dividend : dividend;
          auto const expected = std::fmod(input, divisor);
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
          _mm_setcsr(initial_control);
#endif
          auto const result = real_mod(input, divisor);
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
          require((_mm_getcsr() & ~0x3fu) == control_mode, "MOD changed floating-point control mode");
          _mm_setcsr(initial_control & ~0x8040u);
#endif
          auto flushed = false;
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
          flushed = (initial_control & 0x8000u) != 0u
                 and std::numeric_limits<real>::min_exponent >= std::numeric_limits<double>::min_exponent
                 and boost::math::fpclassify(result) == FP_ZERO
                 and boost::math::fpclassify(expected) == FP_SUBNORMAL;
#endif
          require((result == expected or flushed)
                    and boost::math::signbit(result) == boost::math::signbit(expected),
                  "MOD changed subnormal remainder or zero sign");
        }
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
    _mm_setcsr(initial_control);
#endif
  }
  auto const pi = boost::math::constants::pi<real>();
  auto const positive_zero = real{0};
  auto const negative_zero = std::copysign(real{0}, real{-1});
  auto const maximum = std::numeric_limits<real>::max();
  for (auto const id: {function::ceiling, function::floor})
  {
    for (auto const input: {positive_zero, negative_zero, real{0.25}, real{-0.25}, real{1.5}, real{-1.5},
                           real{1}, real{-1}, real{2}, real{-2},
                           std::nextafter(real{1}, real{0}), std::nextafter(real{1}, maximum),
                           std::nextafter(real{-1}, real{0}), std::nextafter(real{-1}, -maximum),
                           real{2147483648.5}, real{-2147483648.5}, real{1e20}, real{-1e20}, maximum, -maximum,
                           std::numeric_limits<real>::min(), -std::numeric_limits<real>::min()})
    {
      auto const expected = id == function::ceiling ? std::ceil(input) : std::floor(input);
      auto const result = rounded(id, input);
      require(result == expected and boost::math::signbit(result) == boost::math::signbit(expected),
              std::string{bra::classical_function_name(id)} + " changed rounding value or zero sign");
    }
    auto const smallest = std::numeric_limits<real>::denorm_min();
    if (boost::math::fpclassify(smallest) != FP_ZERO)
    {
      constexpr auto half_minimum = std::numeric_limits<real>::min() / real{2};
      for (auto const tiny: {smallest, half_minimum, std::nextafter(std::numeric_limits<real>::min(), real{0})})
        for (auto const negative: {false, true})
        {
          auto const input = negative ? boost::math::changesign(tiny) : tiny;
          auto const result = rounded(id, input);
          auto const expected = id == function::ceiling ? (negative ? negative_zero : real{1})
            : (negative ? real{-1} : positive_zero);
          require(result == expected and boost::math::signbit(result) == boost::math::signbit(expected),
                  "rounding lost stored subnormal sign or treated it as exact zero");
        }
    }
    for (auto const invalid: {std::numeric_limits<real>::infinity(), -std::numeric_limits<real>::infinity(),
                             std::numeric_limits<real>::quiet_NaN()})
      expect_failure<std::domain_error>(id, invalid, "nonfinite input");
    auto in_place = real{-0.25};
    auto const captured = bra::classical_function_call{
      bra::resolve_classical_function(id, {type::real}), {&in_place}, {in_place}};
    in_place = real{100};
    captured.commit_results(bra::evaluate_classical_function(captured));
    auto const expected = id == function::ceiling ? negative_zero : real{-1};
    require(in_place == expected and boost::math::signbit(in_place) == boost::math::signbit(expected),
            "rounding did not use captured input or lost its zero sign");
  }
  for (auto const mode: {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    if (std::fesetround(mode) == 0)
    {
      require(rounded(function::ceiling, real{0.25}) == 1, "CEILING depends on ambient rounding direction");
      require(rounded(function::floor, real{-0.25}) == -1, "FLOOR depends on ambient rounding direction");
      require(std::fegetround() == mode, "rounding function changed ambient rounding direction");
    }
  require(std::fesetround(rounding) == 0, "could not restore test rounding direction");
  for (auto const id: {function::sin, function::cos, function::tan})
  {
    for (auto const input: {positive_zero, negative_zero})
    {
      auto const result = trigonometric(id, input);
      if (id == function::cos) require(result == 1, "COS of zero must be one");
      else
        require(boost::math::fpclassify(result) == FP_ZERO
                and boost::math::signbit(result) == boost::math::signbit(input),
                std::string{bra::classical_function_name(id)} + " lost signed zero: input sign "
                  + std::to_string(boost::math::signbit(input)) + ", result sign "
                  + std::to_string(boost::math::signbit(result)));
    }
    auto const half_pi = pi / real{2};
    for (auto const input: {real{0.25}, real{-0.5}, real{1}, real{-2}, pi, -pi,
                           half_pi, -half_pi, std::nextafter(half_pi, real{0}),
                           std::nextafter(half_pi, maximum), std::nextafter(pi, real{0}),
                           std::nextafter(pi, maximum), real{1e10}, real{-1e10},
                           real{1e20}, real{-1e20}, real{1e30}, real{-1e30}, maximum, -maximum,
                           std::numeric_limits<real>::min(), -std::numeric_limits<real>::min()})
    {
      auto const result = trigonometric(id, input);
      auto const expected = reference_trigonometric(id, input);
      require(close(result, expected), std::string{bra::classical_function_name(id)} + " differs from native reference");
      // Near zeros, an absolute tolerance alone could hide snapping to zero.
      if (expected != 0)
      {
        auto diagnostic = std::ostringstream{};
        diagnostic << bra::classical_function_name(id) << " lost relative accuracy at " << std::hexfloat
                   << input << ": " << result << " != " << expected;
        require(close(result / expected, real{1}), diagnostic.str());
      }
    }
    for (auto const input: {std::numeric_limits<real>::denorm_min(), -std::numeric_limits<real>::denorm_min()})
    {
      auto const result = trigonometric(id, input);
      auto const expected = reference_trigonometric(id, input);
      // Subnormal arithmetic follows the existing library/environment. Check
      // stored classes and signs, not comparisons that DAZ treats as zero.
      require(boost::math::fpclassify(result) == boost::math::fpclassify(expected)
              and boost::math::signbit(result) == boost::math::signbit(expected),
              "trigonometric subnormal behavior differs from native library");
    }
    for (auto const invalid: {std::numeric_limits<real>::infinity(), -std::numeric_limits<real>::infinity(),
                             std::numeric_limits<real>::quiet_NaN()})
      expect_failure<std::domain_error>(id, invalid, "nonfinite input");
    auto in_place = real{0.5};
    auto const captured = bra::classical_function_call{
      bra::resolve_classical_function(id, {type::real}), {&in_place}, {in_place}};
    in_place = real{2};
    captured.commit_results(bra::evaluate_classical_function(captured));
    require(close(in_place, reference_trigonometric(id, real{0.5})), "trigonometric function lost captured input");
  }
  require(close(trigonometric(function::sin, pi / real{6}), real{0.5}), "wrong known SIN");
  require(close(trigonometric(function::cos, pi / real{3}), real{0.5}), "wrong known COS");
  require(close(trigonometric(function::tan, pi / real{4}), real{1}), "wrong known TAN");
  for (auto const id: {function::arcsin, function::arccos, function::arctan})
  {
    for (auto const input: {positive_zero, negative_zero, real{-1}, real{1}, real{-0.5}, real{0.5},
                           std::nextafter(real{1}, real{0}), std::nextafter(real{-1}, real{0}),
                           std::numeric_limits<real>::min(), -std::numeric_limits<real>::min()})
    {
      auto const result = trigonometric(id, input);
      auto const expected = reference_trigonometric(id, input);
      require(close(result, expected), "inverse trigonometric result differs from native reference");
      if (expected != 0)
        require(close(result / expected, real{1}), "inverse trigonometric function lost relative accuracy");
      if (input == 0 and id != function::arccos)
        require(boost::math::fpclassify(result) == FP_ZERO
                and boost::math::signbit(result) == boost::math::signbit(input),
                "inverse trigonometric function lost signed zero");
      if (id == function::arccos)
        require(result >= 0 and result <= pi, "ARCCOS outside principal range");
      else
        require(result >= -pi / real{2} and result <= pi / real{2}, "inverse trigonometric result outside principal range");
    }
    for (auto const input: {std::numeric_limits<real>::denorm_min(), -std::numeric_limits<real>::denorm_min()})
    {
      auto const result = trigonometric(id, input);
      auto const expected = reference_trigonometric(id, input);
      require(close(result, expected)
              and boost::math::fpclassify(result) == boost::math::fpclassify(expected)
              and boost::math::signbit(result) == boost::math::signbit(expected),
              "inverse trigonometric subnormal behavior differs from native library");
    }
    for (auto const invalid: {std::numeric_limits<real>::infinity(), -std::numeric_limits<real>::infinity(),
                             std::numeric_limits<real>::quiet_NaN()})
      expect_failure<std::domain_error>(id, invalid, "nonfinite input");
    if (id != function::arctan)
      for (auto const invalid: {std::nextafter(real{1}, maximum), std::nextafter(real{-1}, -maximum),
                               real{2}, real{-2}, maximum, -maximum})
        expect_failure<std::domain_error>(id, invalid, "real input outside [-1, 1]");
    auto in_place = real{0.5};
    auto const captured = bra::classical_function_call{
      bra::resolve_classical_function(id, {type::real}), {&in_place}, {in_place}};
    in_place = real{0.25};
    captured.commit_results(bra::evaluate_classical_function(captured));
    require(close(in_place, reference_trigonometric(id, real{0.5})), "inverse trigonometric function lost captured input");
  }
  for (auto const input: {real{2}, real{-2}, real{1e20}, real{-1e20}, maximum, -maximum})
  {
    auto const result = trigonometric(function::arctan, input);
    require(close(result, std::atan(input)), "ARCTAN lost large finite input");
    require(result >= -pi / real{2} and result <= pi / real{2}, "ARCTAN outside rounded principal range");
  }
  require(close(trigonometric(function::arcsin, real{0.5}), pi / real{6}), "wrong known ARCSIN");
  require(close(trigonometric(function::arccos, real{0.5}), pi / real{3}), "wrong known ARCCOS");
  require(close(trigonometric(function::arctan, real{1}), pi / real{4}), "wrong known ARCTAN");
  require(close(trigonometric(function::arcsin, real{-1}), -pi / real{2}), "wrong ARCSIN lower endpoint");
  require(close(trigonometric(function::arcsin, real{1}), pi / real{2}), "wrong ARCSIN upper endpoint");
  require(close(trigonometric(function::arccos, real{-1}), pi), "wrong ARCCOS upper endpoint");
  auto const arccos_zero = trigonometric(function::arccos, real{1});
  require(boost::math::fpclassify(arccos_zero) == FP_ZERO and not boost::math::signbit(arccos_zero),
          "ARCCOS(1) must be positive zero");
  for (auto const input: {real{0.25}, real{0.5}, real{1}, real{2}, real{10},
                         std::numeric_limits<real>::min(), maximum})
    require(close(natural_log(input), std::log(input)), "wrong real LOG");
  auto const logarithm_zero = natural_log(real{1});
  require(boost::math::fpclassify(logarithm_zero) == FP_ZERO and not boost::math::signbit(logarithm_zero),
          "LOG(1) must be positive zero");
  for (auto const input: {std::nextafter(real{1}, real{0}), std::nextafter(real{1}, real{2})})
    require(close(natural_log(input) / (input - real{1}), real{1}), "LOG lost precision near one");
  auto const log_two = std::log(real{2});
  for (auto const exponent: {std::numeric_limits<real>::min_exponent - 1, -4, 0, 4,
                            std::numeric_limits<real>::max_exponent - 1})
    require(close(natural_log(std::ldexp(real{1}, exponent)), exponent * log_two), "LOG lost a power of two");
  auto const smallest = std::numeric_limits<real>::denorm_min();
  if (boost::math::fpclassify(smallest) != FP_ZERO)
  {
    // Analytic references do not pass a subnormal through the same libm LOG
    // path as the evaluator: DAZ could make both return the same wrong value.
    auto const expected = (std::numeric_limits<real>::min_exponent - std::numeric_limits<real>::digits) * log_two;
    require(close(natural_log(smallest), expected), "LOG lost the smallest positive subnormal");
    constexpr auto half_minimum = std::numeric_limits<real>::min() / real{2};
    constexpr auto three_quarters = std::numeric_limits<real>::min() * real{0.75};
    require(close(natural_log(half_minimum), (std::numeric_limits<real>::min_exponent - 2) * log_two),
            "LOG lost a subnormal power of two");
    require(close(natural_log(three_quarters), (std::numeric_limits<real>::min_exponent - 1) * log_two + std::log(real{0.75})),
            "LOG lost a subnormal significand");
    auto const largest = std::nextafter(std::numeric_limits<real>::min(), real{0});
    auto const expected_largest = std::log(std::numeric_limits<real>::min()) - std::numeric_limits<real>::epsilon();
    require(close(natural_log(largest), expected_largest), "LOG lost the largest positive subnormal");
    expect_failure<std::domain_error>(function::log, -smallest, "nonpositive real input");
  }
  for (auto const invalid: {positive_zero, negative_zero, real{-1}, -std::numeric_limits<real>::min(), -maximum})
    expect_failure<std::domain_error>(function::log, invalid, "nonpositive real input");
  auto const& constructor = bra::resolve_classical_function(function::complex_, {type::real, type::real});
  for (auto const re: {positive_zero, negative_zero, real{3}, real{-3}, maximum, -maximum,
                      std::numeric_limits<real>::min(), std::numeric_limits<real>::denorm_min()})
    for (auto const im: {positive_zero, negative_zero, real{4}, real{-4}, maximum, -maximum,
                        std::numeric_limits<real>::min(), std::numeric_limits<real>::denorm_min()})
    {
      auto result = complex{-77, 19};
      auto const call = bra::classical_function_call{constructor, {&result}, {re, im}};
      call.commit_results(bra::evaluate_classical_function(call));
      require(result.real() == re and result.imag() == im, "COMPLEX changed a component value");
      require(boost::math::fpclassify(result.real()) == boost::math::fpclassify(re)
              and boost::math::fpclassify(result.imag()) == boost::math::fpclassify(im),
              "COMPLEX flushed a stored component");
      require(boost::math::signbit(result.real()) == boost::math::signbit(re)
              and boost::math::signbit(result.imag()) == boost::math::signbit(im),
              "COMPLEX changed a component sign");
    }
  for (auto const input: {real{0.25}, real{1}, real{4}, std::numeric_limits<real>::min(), maximum})
    require(close(square_root(input) / std::sqrt(input), real{1}), "wrong real SQRT");
  for (auto const re: {positive_zero, negative_zero})
  {
    auto const root = square_root(re);
    require(boost::math::fpclassify(root) == FP_ZERO and boost::math::signbit(root) == boost::math::signbit(re),
            "real SQRT lost signed zero");
    for (auto const im: {positive_zero, negative_zero})
    {
      auto const result = square_root(complex{re, im});
      require(boost::math::fpclassify(result.real()) == FP_ZERO and not boost::math::signbit(result.real()),
              "complex SQRT zero real component must be positive zero");
      require(boost::math::fpclassify(result.imag()) == FP_ZERO and boost::math::signbit(result.imag()) == boost::math::signbit(im),
              "complex SQRT lost imaginary zero sign");
    }
  }
  for (auto const scale: {std::numeric_limits<real>::min(), real{1}, maximum / real{8}})
    for (auto const re_sign: {real{1}, real{-1}})
      for (auto const im_sign: {real{1}, real{-1}})
      {
        auto const root = square_root(complex{re_sign * real{3} * scale, im_sign * real{4} * scale});
        auto const expected_re = re_sign > 0 ? real{2} : real{1};
        auto const expected_im = im_sign * (re_sign > 0 ? real{1} : real{2});
        auto const unit = std::sqrt(scale);
        require(close(root.real() / unit, expected_re) and close(root.imag() / unit, expected_im),
                "SQRT lost a scaled 3-4-5 root");
      }
  for (auto const re_sign: {real{1}, real{-1}})
    for (auto const im_sign: {real{1}, real{-1}})
    {
      auto const root = square_root(complex{re_sign * maximum, im_sign * maximum});
      auto const unit = std::sqrt(maximum);
      auto const diagonal = std::sqrt(real{2});
      require(close(root.real() / unit, std::sqrt((diagonal + re_sign) / real{2}))
              and close(root.imag() / unit, im_sign * std::sqrt((diagonal - re_sign) / real{2})),
              "SQRT unnecessarily overflowed for maximum finite components");
    }
  for (auto const magnitude: {std::numeric_limits<real>::min(), real{4}, maximum})
    for (auto const imaginary: {positive_zero, negative_zero})
    {
      auto const positive = square_root(complex{magnitude, imaginary});
      auto const negative = square_root(complex{-magnitude, imaginary});
      require(close(positive.real() / std::sqrt(magnitude), real{1})
              and boost::math::fpclassify(positive.imag()) == FP_ZERO
              and boost::math::signbit(positive.imag()) == boost::math::signbit(imaginary),
              "SQRT lost positive-axis root or imaginary zero sign");
      require(boost::math::fpclassify(negative.real()) == FP_ZERO and not boost::math::signbit(negative.real())
              and close(std::abs(negative.imag()) / std::sqrt(magnitude), real{1})
              and boost::math::signbit(negative.imag()) == boost::math::signbit(imaginary),
              "SQRT lost negative-axis branch endpoint");
    }
  for (auto const tiny: {std::numeric_limits<real>::min(), std::numeric_limits<real>::denorm_min()})
    if (boost::math::fpclassify(tiny) != FP_ZERO)
    {
      expect_failure<std::domain_error>(function::sqrt, -tiny, "negative real input");
      for (auto const input: {complex{-1, tiny}, complex{-1, -tiny}, complex{tiny, tiny}, complex{-tiny, -tiny}})
      {
        using sqrt_function = complex (*)(complex const&);
        sqrt_function volatile native_sqrt = &std::sqrt<real>;
        auto const expected = native_sqrt(input);
        auto const actual = square_root(input);
        require(close(actual.real(), expected.real()) and close(actual.imag(), expected.imag()),
                "SQRT differs from native tiny-component behavior");
        require(boost::math::signbit(actual.imag()) == boost::math::signbit(input.imag()),
                "SQRT lost the side of the branch cut");
      }
      using sqrt_function = real (*)(real);
      sqrt_function volatile native_sqrt = &std::sqrt;
      auto const expected = native_sqrt(tiny);
      require(square_root(tiny) == expected, "SQRT differs from native subnormal behavior");
    }
  for (auto const input: {real{-1}, -maximum})
    expect_failure<std::domain_error>(function::sqrt, input, "negative real input");
  for (auto const scale: {real{1}, std::numeric_limits<real>::min(), maximum / real{8}})
    for (auto const sign_re: {real{1}, real{-1}})
      for (auto const sign_im: {real{1}, real{-1}})
      {
        // Relative checks at both extremes catch naive squaring, even when
        // an absolute tolerance would accept a spurious zero tiny magnitude.
        require(close(magnitude(complex{sign_re * real{3} * scale, sign_im * real{4} * scale}) / scale, real{5}),
                "ABS lost a scaled 3-4-5 magnitude");
        require(close(magnitude(complex{sign_re * scale, sign_im * scale}) / scale, std::sqrt(real{2})),
                "ABS lost a scaled equal-component magnitude");
      }
  for (auto const scale: {std::numeric_limits<real>::min(), real{1}, maximum})
    for (auto const sign: {real{1}, real{-1}})
    {
      require(magnitude(complex{sign * scale, 0}) == scale, "ABS lost real-axis magnitude");
      require(magnitude(complex{0, sign * scale}) == scale, "ABS lost imaginary-axis magnitude");
    }
  require(close(magnitude(complex{maximum / real{2}, maximum / real{2}}) / (maximum / real{2}), std::sqrt(real{2})),
          "ABS unnecessarily overflowed near maximum");
  require(magnitude(complex{maximum, std::numeric_limits<real>::min()}) == maximum,
          "ABS lost widely separated components");
  for (auto const re: {positive_zero, negative_zero})
    for (auto const im: {positive_zero, negative_zero})
      require(boost::math::fpclassify(magnitude(complex{re, im})) == FP_ZERO, "ABS rejected or changed zero input");
  auto const denormal = std::numeric_limits<real>::denorm_min();
  if (boost::math::fpclassify(denormal) != FP_ZERO)
    for (auto const z: {complex{denormal, 0}, complex{0, -denormal}, complex{denormal, denormal}})
    {
      // Match the existing environment on subnormal arithmetic; do not require
      // gradual underflow or classify a stored subnormal as an invalid input.
      using hypot_function = real (*)(real, real);
      hypot_function volatile native_hypot = &std::hypot;
      auto const expected = native_hypot(z.real(), z.imag());
      auto const actual = magnitude(z);
      require(boost::math::fpclassify(actual) == boost::math::fpclassify(expected),
              "ABS differs from native subnormal magnitude behavior");
      require(actual == expected, "wrong ABS subnormal magnitude");
    }
  for (auto const x: {real{0}, real{1}, real{-1}, real{10}, real{-10}})
    require(close(real_exp(x), std::exp(x)), "wrong real EXP");
  for (auto const z: {complex{0, 0}, complex{1, 2}, complex{-1, -2}, complex{0, pi}})
  {
    auto const actual = complex_exp(z), expected = std::exp(z);
    require(close(actual.real(), expected.real()) and close(actual.imag(), expected.imag()), "wrong complex EXP");
  }
  // exp_i is compiled in this independent, non-fast-math oracle to check the
  // intended exponential. Actual native gate equivalence is audited separately
  // in apply_numerical.py with the default-double executable and in
  // ket/test/exp_i_numerical.py across precisions and optimization modes.
  for (auto const angle: {static_cast<real>(1e20L), static_cast<real>(-1e20L), maximum, -maximum})
  {
    auto const actual = complex_exp(complex{0, angle});
    auto const expected = ket::utility::exp_i<complex>(angle);
    require(close(actual.real(), expected.real()) and close(actual.imag(), expected.imag()), "EXP differs from reference exponential for huge angle");
  }
  require(argument(complex{-1, positive_zero}) == pi, "ARG lost upper branch endpoint");
  require(argument(complex{-1, negative_zero}) == -pi, "ARG lost lower branch endpoint");
  auto const upper_zero = argument(complex{1, positive_zero});
  auto const lower_zero = argument(complex{1, negative_zero});
  require(upper_zero == 0 and not boost::math::signbit(upper_zero), "ARG lost positive zero");
  require(lower_zero == 0 and boost::math::signbit(lower_zero), "ARG lost negative zero");
  for (auto const tiny: {std::numeric_limits<real>::min(), std::numeric_limits<real>::denorm_min()})
    if (boost::math::fpclassify(tiny) != FP_ZERO)
    {
      require(argument(complex{-1, tiny}) > 0 and argument(complex{-1, -tiny}) < 0, "ARG snapped branch-cut neighbors");
      volatile auto native_tiny = tiny;
      auto const native_angle = std::atan2(native_tiny, native_tiny);
      if (boost::math::isfinite(native_angle))
        require(close(argument(complex{tiny, tiny}), native_angle), "ARG differs from library on tiny input");
      else
        // On some libm implementations, DAZ makes the internal ratio of two
        // nonzero subnormals become 0/0. Reject that nonfinite result, without
        // misreporting a stored nonzero operand as an exact-zero domain error.
        expect_failure<std::overflow_error>(function::arg, complex{tiny, tiny}, "nonfinite result");
      auto const axis_zero = argument(complex{tiny, negative_zero});
      require(boost::math::fpclassify(axis_zero) == FP_ZERO and boost::math::signbit(axis_zero),
              "ARG lost signed zero on a subnormal positive real axis");
      require(argument(complex{-tiny, negative_zero}) == -pi, "ARG lost subnormal negative real axis");
      require(argument(complex{positive_zero, tiny}) == pi / real{2}, "ARG lost subnormal imaginary axis");
      require(argument(complex{negative_zero, -tiny}) == -pi / real{2}, "ARG lost negative subnormal imaginary axis");
      require(not boost::math::signbit(argument(complex{1, tiny}))
              and boost::math::signbit(argument(complex{1, -tiny})), "ARG lost tiny imaginary sign near positive axis");
      auto const actual = complex_exp(complex{0, tiny});
      auto const expected = ket::utility::exp_i<complex>(tiny);
      require(boost::math::fpclassify(actual.imag()) == boost::math::fpclassify(expected.imag()),
              "EXP tiny imaginary result differs from backend environment behavior");
    }
  for (auto const z: {complex{1, 1}, complex{-1, 1}, complex{-1, -1}, complex{1, -1},
                      complex{maximum, maximum}, complex{-maximum, -maximum}})
    require(close(argument(z), std::atan2(z.imag(), z.real())), "wrong ARG quadrant or extreme input");

  auto const infinity = std::numeric_limits<real>::infinity();
  auto const nan = std::numeric_limits<real>::quiet_NaN();
  for (auto const invalid: {infinity, -infinity, nan})
  {
    for (auto const& inputs: {std::vector<bra::classical_function_value>{invalid, real{0}},
                            std::vector<bra::classical_function_value>{real{0}, invalid},
                            std::vector<bra::classical_function_value>{invalid, invalid}})
    {
      auto result = complex{-77, 19};
      auto const call = bra::classical_function_call{constructor, {&result}, inputs};
      auto caught = false;
      try { call.commit_results(bra::evaluate_classical_function(call)); }
      catch (std::domain_error const& error)
      {
        require(std::string{error.what()} == "nonfinite input to COMPLEX", "wrong COMPLEX diagnostic");
        caught = true;
      }
      require(caught and result == complex{-77, 19}, "failed COMPLEX call modified destination");
    }
    expect_failure<std::domain_error>(function::exp, invalid, "nonfinite input");
    expect_failure<std::domain_error>(function::sqrt, invalid, "nonfinite input");
    expect_failure<std::domain_error>(function::log, invalid, "nonfinite input");
    for (auto const z: {complex{invalid, 0}, complex{0, invalid}, complex{invalid, invalid}})
    {
      expect_failure<std::domain_error>(function::exp, z, "nonfinite input");
      expect_failure<std::domain_error>(function::arg, z, "nonfinite input");
      expect_failure<std::domain_error>(function::abs, z, "nonfinite input");
      expect_failure<std::domain_error>(function::sqrt, z, "nonfinite input");
    }
  }
  for (auto const re: {positive_zero, negative_zero})
    for (auto const im: {positive_zero, negative_zero})
      expect_failure<std::domain_error>(function::arg, complex{re, im}, "zero complex input");
  expect_failure<std::overflow_error>(function::exp, maximum, "overflow/nonfinite result");
  expect_failure<std::overflow_error>(function::abs, complex{maximum, maximum}, "overflow/nonfinite result");
  expect_failure<std::overflow_error>(function::abs, complex{-maximum, -maximum}, "overflow/nonfinite result");
  expect_failure<std::domain_error>(function::abs, complex{infinity, nan}, "nonfinite input");
  expect_failure<std::domain_error>(function::abs, complex{nan, infinity}, "nonfinite input");
  expect_failure<std::overflow_error>(function::exp, complex{maximum, 0}, "overflow/nonfinite result");
  expect_failure<std::overflow_error>(function::exp, complex{maximum, 1}, "overflow/nonfinite result");
  require(real_exp(-maximum) == 0, "real EXP underflow should be allowed");
  require(complex_exp(complex{-maximum, 1}) == complex{0, 0}, "complex EXP underflow should be allowed");
  // Underflow is not an error. Match native real EXP in the existing floating-
  // point environment; release startup may flush subnormals. A volatile input
  // prevents the reference call from being folded at compile time.
  auto const near_underflow = std::log(std::numeric_limits<real>::min()) - real{1};
  volatile auto native_input = near_underflow;
  auto const native_result = std::exp(native_input);
  require(boost::math::fpclassify(real_exp(near_underflow)) == boost::math::fpclassify(native_result),
          "real EXP differs from native underflow behavior");
  auto const subnormal = complex_exp(complex{near_underflow, 1});
  for (auto const component: {subnormal.real(), subnormal.imag()})
    require(boost::math::fpclassify(component) == FP_SUBNORMAL or boost::math::fpclassify(component) == FP_ZERO,
            "unexpected complex EXP underflow result");

  auto const imaginary_zero = complex_exp(complex{0, negative_zero}).imag();
  require(imaginary_zero == 0 and boost::math::signbit(imaginary_zero), "complex EXP lost imaginary signed zero");
  // In-place calls use the captured input, not the current destination value.
  auto x = real{1};
  auto divisor = real{3};
  auto const mod_call = bra::classical_function_call{
    bra::resolve_classical_function(function::mod, {type::real, type::real}), {&divisor}, {real{7}, divisor}};
  divisor = 0;
  mod_call.commit_results(bra::evaluate_classical_function(mod_call));
  require(divisor == real{1}, "MOD did not use captured divisor for aliased destination");
  auto const call = bra::classical_function_call{
    bra::resolve_classical_function(function::exp, {type::real}), {&x}, {x}};
  x = 10;
  call.commit_results(bra::evaluate_classical_function(call));
  require(close(x, std::exp(real{1})), "EXP did not use captured input");
  auto z = complex{3, 4};
  auto const abs_call = bra::classical_function_call{
    bra::resolve_classical_function(function::abs, {type::complex_}), {&x}, {z}};
  z = complex{0, 0};
  abs_call.commit_results(bra::evaluate_classical_function(abs_call));
  require(x == 5, "ABS did not use captured input");
  z = complex{-3, -4};
  auto const sqrt_call = bra::classical_function_call{
    bra::resolve_classical_function(function::sqrt, {type::complex_}), {&z}, {z}};
  z = complex{0, 0};
  sqrt_call.commit_results(bra::evaluate_classical_function(sqrt_call));
  require(z == complex{1, -2}, "SQRT did not use captured input for in-place evaluation");
  x = real{2};
  auto const log_call = bra::classical_function_call{
    bra::resolve_classical_function(function::log, {type::real}), {&x}, {x}};
  x = real{7};
  log_call.commit_results(bra::evaluate_classical_function(log_call));
  require(close(x, log_two), "LOG did not use captured input for in-place evaluation");
  require(std::fegetround() == rounding, "APPLY changed rounding mode");
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
  require((_mm_getcsr() & ~0x3fu) == control_mode, "APPLY changed floating-point control mode");
#endif
}
