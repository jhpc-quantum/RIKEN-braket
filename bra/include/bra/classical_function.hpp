#ifndef BRA_CLASSICAL_FUNCTION_HPP
# define BRA_CLASSICAL_FUNCTION_HPP

# include <array>
# include <stdexcept>
# include <vector>

# include <bra/variable_type.hpp>


namespace bra
{
  enum class classical_function_id : int
  { exp = 0, arg = 1 };

  struct classical_function_signature
  {
    ::bra::classical_function_id function;
    std::vector< ::bra::variable_type > operand_types;
    // A sequence, not a scalar: future functions may return multiple results.
    std::vector< ::bra::variable_type > result_types;
  };

  inline auto classical_function_name(::bra::classical_function_id const function) -> char const*
  {
    switch (function)
    {
    case ::bra::classical_function_id::exp: return "EXP";
    case ::bra::classical_function_id::arg: return "ARG";
    }
    throw std::invalid_argument{"unknown classical function identifier"};
  }

  // Overload lookup must compare operand types only. Result types are checked
  // against destinations after selecting the signature, never used to select it.
  inline auto classical_function_signatures()
    -> std::array< ::bra::classical_function_signature, 3u > const&
  {
    static auto const signatures = std::array< ::bra::classical_function_signature, 3u >{{
      {::bra::classical_function_id::exp, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::exp, {::bra::variable_type::complex_}, {::bra::variable_type::complex_}},
      {::bra::classical_function_id::arg, {::bra::variable_type::complex_}, {::bra::variable_type::real}}
    }};
    return signatures;
  }
} // namespace bra


#endif // BRA_CLASSICAL_FUNCTION_HPP
