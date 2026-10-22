#ifndef BRA_CLASSICAL_FUNCTION_HPP
# define BRA_CLASSICAL_FUNCTION_HPP

# include <array>
# include <stdexcept>
# include <string>
# include <vector>

# include <bra/variable_type.hpp>


namespace bra
{
  enum class classical_function_id : int
  { exp = 0, arg = 1, abs = 2, sqrt = 3, complex_ = 4 };

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
    case ::bra::classical_function_id::abs: return "ABS";
    case ::bra::classical_function_id::sqrt: return "SQRT";
    case ::bra::classical_function_id::complex_: return "COMPLEX";
    }
    throw std::invalid_argument{"unknown classical function identifier"};
  }

  // Overload lookup must compare operand types only. Result types are checked
  // against destinations after selecting the signature, never used to select it.
  inline auto classical_function_signatures()
    -> std::array< ::bra::classical_function_signature, 7u > const&
  {
    static auto const signatures = std::array< ::bra::classical_function_signature, 7u >{{
      {::bra::classical_function_id::exp, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::exp, {::bra::variable_type::complex_}, {::bra::variable_type::complex_}},
      {::bra::classical_function_id::arg, {::bra::variable_type::complex_}, {::bra::variable_type::real}},
      {::bra::classical_function_id::abs, {::bra::variable_type::complex_}, {::bra::variable_type::real}},
      {::bra::classical_function_id::sqrt, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::sqrt, {::bra::variable_type::complex_}, {::bra::variable_type::complex_}},
      {::bra::classical_function_id::complex_, {::bra::variable_type::real, ::bra::variable_type::real},
                                            {::bra::variable_type::complex_}}
    }};
    return signatures;
  }

  inline auto classical_function_from_name(std::string const& name) -> ::bra::classical_function_id
  {
    for (auto const& signature: classical_function_signatures())
      if (name == classical_function_name(signature.function))
        return signature.function;
    throw std::invalid_argument{"unknown function " + name};
  }

  inline auto resolve_classical_function(
    ::bra::classical_function_id const function, std::vector< ::bra::variable_type > const& operand_types)
    -> ::bra::classical_function_signature const&
  {
    auto matching_arity = false;
    for (auto const& signature: classical_function_signatures())
      if (signature.function == function)
      {
        matching_arity = matching_arity or signature.operand_types.size() == operand_types.size();
        if (signature.operand_types == operand_types)
          return signature;
      }
    if (not matching_arity)
      throw std::invalid_argument{"wrong operand count for " + std::string{classical_function_name(function)}};
    throw std::invalid_argument{"unsupported operand signature for " + std::string{classical_function_name(function)}};
  }
} // namespace bra


#endif // BRA_CLASSICAL_FUNCTION_HPP
