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
  {
    exp = 0, arg = 1, abs = 2, sqrt = 3, complex_ = 4, log = 5, sin = 6, cos = 7, tan = 8,
    arcsin = 9, arccos = 10, arctan = 11, ceiling = 12, floor = 13, mod = 14, anglebits = 15
  };

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
    case ::bra::classical_function_id::log: return "LOG";
    case ::bra::classical_function_id::sin: return "SIN";
    case ::bra::classical_function_id::cos: return "COS";
    case ::bra::classical_function_id::tan: return "TAN";
    case ::bra::classical_function_id::arcsin: return "ARCSIN";
    case ::bra::classical_function_id::arccos: return "ARCCOS";
    case ::bra::classical_function_id::arctan: return "ARCTAN";
    case ::bra::classical_function_id::ceiling: return "CEILING";
    case ::bra::classical_function_id::floor: return "FLOOR";
    case ::bra::classical_function_id::mod: return "MOD";
    case ::bra::classical_function_id::anglebits: return "ANGLEBITS";
    }
    throw std::invalid_argument{"unknown classical function identifier"};
  }

  // Overload lookup must compare operand types only. Result types are checked
  // against destinations after selecting the signature, never used to select it.
  inline auto classical_function_signatures()
    -> std::array< ::bra::classical_function_signature, 18u > const&
  {
    static auto const signatures = std::array< ::bra::classical_function_signature, 18u >{{
      {::bra::classical_function_id::exp, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::exp, {::bra::variable_type::complex_}, {::bra::variable_type::complex_}},
      {::bra::classical_function_id::arg, {::bra::variable_type::complex_}, {::bra::variable_type::real}},
      {::bra::classical_function_id::abs, {::bra::variable_type::complex_}, {::bra::variable_type::real}},
      {::bra::classical_function_id::sqrt, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::sqrt, {::bra::variable_type::complex_}, {::bra::variable_type::complex_}},
      {::bra::classical_function_id::complex_, {::bra::variable_type::real, ::bra::variable_type::real},
                                            {::bra::variable_type::complex_}},
      {::bra::classical_function_id::log, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::sin, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::cos, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::tan, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::arcsin, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::arccos, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::arctan, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::ceiling, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::floor, {::bra::variable_type::real}, {::bra::variable_type::real}},
      {::bra::classical_function_id::mod, {::bra::variable_type::real, ::bra::variable_type::real},
                                       {::bra::variable_type::real}},
      {::bra::classical_function_id::anglebits, {::bra::variable_type::real, ::bra::variable_type::integer},
                                             {::bra::variable_type::unsigned_integer}}
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
