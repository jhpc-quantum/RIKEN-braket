#include <cmath>
#include <complex>
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
    auto const& operand = call.operands().front();
    if (auto const value = boost::get< ::bra::real_type >(&operand))
    {
      if (signature.function != ::bra::classical_function_id::exp)
        throw std::invalid_argument{"unsupported real function evaluation"};
      if (not boost::math::isfinite(*value))
        throw std::domain_error{"nonfinite input to EXP"};
      auto const result = std::exp(bounded_exp_argument(*value));
      if (not boost::math::isfinite(result))
        throw std::overflow_error{"overflow/nonfinite result from EXP"};
      return {result};
    }
    auto const value = boost::get< ::bra::complex_type >(operand);
    if (not finite(value))
      throw std::domain_error{"nonfinite input to " + std::string{::bra::classical_function_name(signature.function)}};
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
