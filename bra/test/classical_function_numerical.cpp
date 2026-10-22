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
  // Ignore accrued exception-status bits; FTZ, DAZ, rounding, and masks must
  // remain exactly as selected by the caller/build's startup environment.
  auto const control_mode = _mm_getcsr() & ~0x3fu;
#endif
  auto const pi = boost::math::constants::pi<real>();
  auto const positive_zero = real{0};
  auto const negative_zero = std::copysign(real{0}, real{-1});
  auto const maximum = std::numeric_limits<real>::max();
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
