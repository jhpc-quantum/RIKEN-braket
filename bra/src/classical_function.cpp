#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/sign.hpp>
#include <boost/math/constants/constants.hpp>

#include <bra/classical_function_call.hpp>


namespace bra
{
  namespace
  {
    bool finite(::bra::complex_type const& value)
    { return boost::math::isfinite(value.real()) and boost::math::isfinite(value.imag()); }

    bool zero(::bra::real_type const value)
    {
      // Representation-based classification must distinguish a stored
      // subnormal from zero, even when the hardware treats it as zero in
      // arithmetic. Do not use value == 0 under fast-math/DAZ.
      return boost::math::fpclassify(value) == FP_ZERO;
    }

    auto bounded_exp_argument(::bra::real_type const value) -> ::bra::real_type
    {
      // Values below this conservative bound already round below the smallest
      // representable magnitude. Keeping the argument finite and modest avoids
      // an intermediate overflow in optimized x87 EXP's range reduction for
      // inputs such as -LDBL_MAX. Leave rounding/underflow to the library and
      // current environment rather than manufacturing or flushing the result.
      using limits = std::numeric_limits< ::bra::real_type >;
      auto const bound = static_cast< ::bra::real_type >(limits::min_exponent - limits::digits - 2)
        * std::log(static_cast< ::bra::real_type >(limits::radix));
      return value < bound ? bound : value;
    }

    // Select C entry points, not C++ overload wrappers: an address-taken
    // std::sin(long double) wrapper can itself contain an x87 FSIN under
    // fast-math, defeating the indirect-call boundary.
    inline auto trigonometric_functions(float) -> std::array<float (*)(float), 6u>
    { return {{&::sinf, &::cosf, &::tanf, &::asinf, &::acosf, &::atanf}}; }

    inline auto trigonometric_functions(double) -> std::array<double (*)(double), 6u>
    { return {{&::sin, &::cos, &::tan, &::asin, &::acos, &::atan}}; }

    inline auto trigonometric_functions(long double) -> std::array<long double (*)(long double), 6u>
    { return {{&::sinl, &::cosl, &::tanl, &::asinl, &::acosl, &::atanl}}; }

    auto library_real_trigonometric(::bra::classical_function_id const id, ::bra::real_type const value)
      -> ::bra::real_type
    {
      // Keep the library boundary: optimized inline x87 trigonometric
      // instructions can fail range reduction for large finite operands.
      // The library also retains signed-zero semantics. Do not pre-reduce
      // the argument or change floating-point control modes.
      auto const functions = trigonometric_functions(::bra::real_type{});
      using function_type = ::bra::real_type (*)(::bra::real_type);
      function_type volatile function = nullptr;
      switch (id)
      {
      case ::bra::classical_function_id::sin: function = functions[0u]; break;
      case ::bra::classical_function_id::cos: function = functions[1u]; break;
      case ::bra::classical_function_id::tan: function = functions[2u]; break;
      case ::bra::classical_function_id::arcsin: function = functions[3u]; break;
      case ::bra::classical_function_id::arccos: function = functions[4u]; break;
      case ::bra::classical_function_id::arctan: function = functions[5u]; break;
      default: throw std::invalid_argument{"unsupported trigonometric function evaluation"};
      }
      return function(value);
    }

    inline auto rounding_functions(float) -> std::array<float (*)(float), 2u>
    { return {{&::ceilf, &::floorf}}; }

    inline auto rounding_functions(double) -> std::array<double (*)(double), 2u>
    { return {{&::ceil, &::floor}}; }

    inline auto rounding_functions(long double) -> std::array<long double (*)(long double), 2u>
    { return {{&::ceill, &::floorl}}; }

    auto library_real_rounding(::bra::classical_function_id const id, ::bra::real_type const value)
      -> ::bra::real_type
    {
      auto argument = value;
      // A stored nonzero subnormal must not be rounded as an exact zero
      // under DAZ. Every value in (0,1), or (-1,0), has the same directed
      // rounding result: substitute an exact normal representative without
      // doing arithmetic on the original subnormal or changing control modes.
      volatile auto const negative = boost::math::signbit(value);
      if (boost::math::fpclassify(value) == FP_SUBNORMAL)
        argument = negative ? ::bra::real_type{-0.5} : ::bra::real_type{0.5};
      auto const functions = rounding_functions(::bra::real_type{});
      using function_type = ::bra::real_type (*)(::bra::real_type);
      // Use C entry points to retain signed zeros under fast-math and avoid
      // compiler expansions that round through native integer conversions.
      function_type volatile function = id == ::bra::classical_function_id::ceiling ? functions[0u] : functions[1u];
      return function(argument);
    }

    auto library_real_log(::bra::real_type const value) -> ::bra::real_type
    {
      auto argument = value;
      auto offset = ::bra::real_type{0};
      if (boost::math::fpclassify(value) == FP_SUBNORMAL)
      {
        using limits = std::numeric_limits< ::bra::real_type >;
        if (not limits::is_iec559 or limits::radix != 2)
          throw std::domain_error{"LOG cannot normalize subnormal input with this REAL representation"};
        // An IEEE binary subnormal is f * 2^(min_exponent-1). OR its
        // representation with 1.0 to obtain the normal value 1+f (also
        // setting the explicit integer bit on x87). Subtracting 1 is exact
        // and yields a normal f; unlike multiplication/frexp on the original
        // subnormal, none of this arithmetic can lose the input under DAZ.
        // Bytewise copies avoid aliasing, alignment, and byte-order assumptions.
        auto const unit = ::bra::real_type{1};
        unsigned char value_bits[sizeof(value)], unit_bits[sizeof(unit)];
        std::memcpy(value_bits, &value, sizeof(value));
        std::memcpy(unit_bits, &unit, sizeof(unit));
        for (auto index = std::size_t{0u}; index < sizeof(value); ++index)
          value_bits[index] |= unit_bits[index];
        std::memcpy(&argument, value_bits, sizeof(argument));
        argument -= unit;
        offset = static_cast< ::bra::real_type >(limits::min_exponent - 1)
          * boost::math::constants::ln_two< ::bra::real_type >();
      }
      using function_type = ::bra::real_type (*)(::bra::real_type);
      function_type volatile function = &std::log;
      return function(argument) + offset;
    }

    auto library_complex_exp(::bra::complex_type const& value) -> ::bra::complex_type
    {
      // Keep the library function boundary: propagating the bounded real
      // component into complex builtins can select a restricted-range x87
      // expansion under fast-math. The indirect call prevents that propagation
      // without changing optimization options or the floating-point environment.
      using function_type = ::bra::complex_type (*)(::bra::complex_type const&);
      function_type volatile function = &std::exp< ::bra::real_type >;
      return function(value);
    }

    auto library_complex_abs(::bra::complex_type const& value) -> ::bra::real_type
    {
      // Preserve the scaled library algorithm. Under fast-math, inlining a
      // complex absolute value can instead square its components directly,
      // overflowing/underflowing even when the magnitude is representable.
      // As with EXP, keep the function boundary without changing build flags
      // or the caller's floating-point environment.
      using function_type = ::bra::real_type (*)(::bra::real_type, ::bra::real_type);
      function_type volatile function = &std::hypot;
      return function(value.real(), value.imag());
    }

    auto principal_sqrt(::bra::complex_type const& value) -> ::bra::complex_type
    {
      // Observe signs before zero classification, as for ARG below. Explicit
      // axis handling preserves the two sides of the cut even under fast-math.
      volatile auto const real_negative = boost::math::signbit(value.real());
      volatile auto const imag_negative = boost::math::signbit(value.imag());
      volatile auto const imaginary_component = value.imag();
      if (zero(value.imag()))
      {
        if (zero(value.real()))
          return {::bra::real_type{0}, static_cast< ::bra::real_type >(imaginary_component)};
        if (real_negative)
        {
          auto const magnitude = std::sqrt(boost::math::changesign(value.real()));
          return {::bra::real_type{0}, imag_negative ? boost::math::changesign(magnitude) : magnitude};
        }
        return {std::sqrt(value.real()), static_cast< ::bra::real_type >(imaginary_component)};
      }
      // Keep the scaled complex library algorithm across the optimization
      // boundary; directly forming abs(z) or squaring components can overflow
      // even though every finite complex input has a representable root.
      using function_type = ::bra::complex_type (*)(::bra::complex_type const&);
      function_type volatile function = &std::sqrt< ::bra::real_type >;
      auto const result = function(value);
      auto const imaginary_magnitude = std::abs(result.imag());
      return {std::abs(result.real()), imag_negative ? boost::math::changesign(imaginary_magnitude) : imaginary_magnitude};
    }
  }

  auto evaluate_classical_function(::bra::classical_function_call const& call)
    -> std::vector< ::bra::classical_function_value >
  {
    auto const& signature = call.signature();
    // Only registered signatures are executable. The generic capture/storage
    // machinery can also be exercised with synthetic multi-result signatures.
    auto const& registered = ::bra::resolve_classical_function(signature.function, signature.operand_types);
    if (signature.result_types != registered.result_types)
      throw std::invalid_argument{"unsupported result signature"};
    if (signature.function == ::bra::classical_function_id::complex_)
    {
      auto const real = boost::get< ::bra::real_type >(call.operands()[0u]);
      auto const imaginary = boost::get< ::bra::real_type >(call.operands()[1u]);
      if (not boost::math::isfinite(real) or not boost::math::isfinite(imaginary))
        throw std::domain_error{"nonfinite input to COMPLEX"};
      // Construct components directly: arithmetic assembly loses signed zeros
      // and can flush subnormals even though no mathematical work is needed.
      return {::bra::complex_type{real, imaginary}};
    }
    auto const& operand = call.operands().front();
    if (auto const value = boost::get< ::bra::real_type >(&operand))
    {
      if (not boost::math::isfinite(*value))
        throw std::domain_error{"nonfinite input to " + std::string{::bra::classical_function_name(signature.function)}};
      if (signature.function == ::bra::classical_function_id::ceiling
          or signature.function == ::bra::classical_function_id::floor)
      {
        auto const result = library_real_rounding(signature.function, *value);
        if (not boost::math::isfinite(result))
          throw std::overflow_error{"overflow/nonfinite result from "
            + std::string{::bra::classical_function_name(signature.function)}};
        return {result};
      }
      if (signature.function == ::bra::classical_function_id::sin
          or signature.function == ::bra::classical_function_id::cos
          or signature.function == ::bra::classical_function_id::tan
          or signature.function == ::bra::classical_function_id::arcsin
          or signature.function == ::bra::classical_function_id::arccos
          or signature.function == ::bra::classical_function_id::arctan)
      {
        if ((signature.function == ::bra::classical_function_id::arcsin
             or signature.function == ::bra::classical_function_id::arccos)
            and (*value < ::bra::real_type{-1} or *value > ::bra::real_type{1}))
          throw std::domain_error{std::string{::bra::classical_function_name(signature.function)}
            + " domain error: real input outside [-1, 1]"};
        auto const result = library_real_trigonometric(signature.function, *value);
        if (not boost::math::isfinite(result))
          throw std::overflow_error{"overflow/nonfinite result from "
            + std::string{::bra::classical_function_name(signature.function)}};
        return {result};
      }
      if (signature.function == ::bra::classical_function_id::log)
      {
        // Test the representation: DAZ must not turn a stored positive
        // subnormal into an exact-zero domain error.
        volatile auto const negative = boost::math::signbit(*value);
        if (zero(*value) or negative)
          throw std::domain_error{"LOG domain error: nonpositive real input"};
        auto const result = library_real_log(*value);
        if (not boost::math::isfinite(result))
          throw std::overflow_error{"overflow/nonfinite result from LOG"};
        return {result};
      }
      if (signature.function == ::bra::classical_function_id::sqrt)
      {
        // -0 is a valid input and retains its sign. Negative stored subnormals
        // are invalid even when DAZ would compare them equal to zero.
        volatile auto const stored_value = *value;
        volatile auto const negative = boost::math::signbit(*value);
        if (zero(*value)) return {static_cast< ::bra::real_type >(stored_value)};
        if (negative)
          throw std::domain_error{"SQRT domain error: negative real input"};
        auto const result = std::sqrt(*value);
        if (not boost::math::isfinite(result))
          throw std::overflow_error{"overflow/nonfinite result from SQRT"};
        return {result};
      }
      if (signature.function != ::bra::classical_function_id::exp)
        throw std::invalid_argument{"unsupported real function evaluation"};
      auto const result = std::exp(bounded_exp_argument(*value));
      if (not boost::math::isfinite(result))
        throw std::overflow_error{"overflow/nonfinite result from EXP"};
      return {result};
    }
    auto const value = boost::get< ::bra::complex_type >(operand);
    if (not finite(value))
      throw std::domain_error{"nonfinite input to " + std::string{::bra::classical_function_name(signature.function)}};
    if (signature.function == ::bra::classical_function_id::sqrt)
    {
      auto const result = principal_sqrt(value);
      if (not finite(result))
        throw std::overflow_error{"overflow/nonfinite result from SQRT"};
      return {result};
    }
    if (signature.function == ::bra::classical_function_id::abs)
    {
      auto const result = library_complex_abs(value);
      if (not boost::math::isfinite(result))
        throw std::overflow_error{"overflow/nonfinite result from ABS"};
      return {result};
    }
    // Snapshot the representation-derived signs as integers before classifying
    // zeros. GCC fast-math can otherwise replace signbit(x) by false on a path
    // where x was classified as zero (observed for x87 long double). Volatile
    // snapshots keep these bit observations independent of that inference,
    // including the imaginary zero payload returned on the positive real axis.
    volatile auto const real_negative = boost::math::signbit(value.real());
    volatile auto const imag_negative = boost::math::signbit(value.imag());
    volatile auto const imaginary_component = value.imag();
    if (signature.function == ::bra::classical_function_id::exp)
    {
      // The same complex exponential as ket::utility::exp_i, without reducing
      // imaginary angles modulo a floating-point approximation of 2*pi.
      auto const result = library_complex_exp(::bra::complex_type{bounded_exp_argument(value.real()), value.imag()});
      if (not finite(result))
        throw std::overflow_error{"overflow/nonfinite result from EXP"};
      if (zero(value.imag()))
        // Preserve an exact imaginary signed zero without relying on
        // arithmetic identities that fast-math is allowed to simplify.
        return {::bra::complex_type{result.real(), static_cast< ::bra::real_type >(imaginary_component)}};
      return {result};
    }
    if (signature.function != ::bra::classical_function_id::arg)
      throw std::invalid_argument{"unsupported complex function evaluation"};
    auto const real_zero = zero(value.real());
    auto const imag_zero = zero(value.imag());
    if (real_zero and imag_zero)
      throw std::domain_error{"ARG domain error: zero complex input"};
    if (imag_zero)
    {
      if (not real_negative)
        return {static_cast< ::bra::real_type >(imaginary_component)};
      auto const pi = boost::math::constants::pi< ::bra::real_type >();
      return {imag_negative ? boost::math::changesign(pi) : pi};
    }
    if (real_zero)
    {
      auto const half_pi = boost::math::constants::half_pi< ::bra::real_type >();
      return {imag_negative ? boost::math::changesign(half_pi) : half_pi};
    }
    // The quadrant-aware library operation inherits the existing environment,
    // including its behavior on subnormal operands. Check the result as usual.
    auto const result = std::atan2(value.imag(), value.real());
    if (not boost::math::isfinite(result))
      throw std::overflow_error{"nonfinite result from ARG"};
    // The argument's sign is the stored imaginary sign. Some library paths
    // compare tiny inputs with zero under DAZ and can lose that sign near the
    // cut. Keep the library magnitude, but restore the observed sign explicitly.
    auto const magnitude = std::abs(result);
    return {imag_negative ? boost::math::changesign(magnitude) : magnitude};
  }
} // namespace bra
