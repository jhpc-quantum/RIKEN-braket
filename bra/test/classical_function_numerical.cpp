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
  // Ignore accrued exception-status bits; FTZ, DAZ, rounding, and masks must
  // remain exactly as selected by the caller/build's startup environment.
  auto const control_mode = _mm_getcsr() & ~0x3fu;
#endif
  auto const pi = boost::math::constants::pi<real>();
  auto const positive_zero = real{0};
  auto const negative_zero = std::copysign(real{0}, real{-1});
  auto const maximum = std::numeric_limits<real>::max();
  for (auto const x: {real{0}, real{1}, real{-1}, real{10}, real{-10}})
    require(close(real_exp(x), std::exp(x)), "wrong real EXP");
  for (auto const z: {complex{0, 0}, complex{1, 2}, complex{-1, -2}, complex{0, pi}})
  {
    auto const actual = complex_exp(z), expected = std::exp(z);
    require(close(actual.real(), expected.real()) and close(actual.imag(), expected.imag()), "wrong complex EXP");
  }
  for (auto const angle: {static_cast<real>(1e20L), static_cast<real>(-1e20L), maximum, -maximum})
  {
    auto const actual = complex_exp(complex{0, angle});
    auto const expected = ket::utility::exp_i<complex>(angle);
    require(close(actual.real(), expected.real()) and close(actual.imag(), expected.imag()), "EXP differs from backend for huge angle");
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
    expect_failure<std::domain_error>(function::exp, invalid, "nonfinite input");
    for (auto const z: {complex{invalid, 0}, complex{0, invalid}, complex{invalid, invalid}})
    {
      expect_failure<std::domain_error>(function::exp, z, "nonfinite input");
      expect_failure<std::domain_error>(function::arg, z, "nonfinite input");
    }
  }
  for (auto const re: {positive_zero, negative_zero})
    for (auto const im: {positive_zero, negative_zero})
      expect_failure<std::domain_error>(function::arg, complex{re, im}, "zero complex input");
  expect_failure<std::overflow_error>(function::exp, maximum, "overflow/nonfinite result");
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
  auto const call = bra::classical_function_call{
    bra::resolve_classical_function(function::exp, {type::real}), {&x}, {x}};
  x = 10;
  call.commit_results(bra::evaluate_classical_function(call));
  require(close(x, std::exp(real{1})), "EXP did not use captured input");
  require(std::fegetround() == rounding, "APPLY changed rounding mode");
#if defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
  require((_mm_getcsr() & ~0x3fu) == control_mode, "APPLY changed floating-point control mode");
#endif
}
