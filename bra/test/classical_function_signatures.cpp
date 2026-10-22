// Standalone C++14 test; only bra/include is required.
#include <stdexcept>
#include <string>
#include <vector>

#include <bra/classical_function.hpp>

namespace
{
  void require(bool const condition)
  {
    if (not condition)
      throw std::runtime_error{"incorrect classical function signature contract"};
  }
}

int main()
{
  using type = bra::variable_type;
  using function = bra::classical_function_id;
  auto const& signatures = bra::classical_function_signatures();
  require(signatures.size() == 8u);
  require(signatures[0u].function == function::exp);
  require(signatures[0u].operand_types == std::vector<type>{type::real});
  require(signatures[0u].result_types == std::vector<type>{type::real});
  require(signatures[1u].function == function::exp);
  require(signatures[1u].operand_types == std::vector<type>{type::complex_});
  require(signatures[1u].result_types == std::vector<type>{type::complex_});
  require(signatures[2u].function == function::arg);
  require(signatures[2u].operand_types == std::vector<type>{type::complex_});
  require(signatures[2u].result_types == std::vector<type>{type::real});
  require(signatures[3u].function == function::abs);
  require(signatures[3u].operand_types == std::vector<type>{type::complex_});
  require(signatures[3u].result_types == std::vector<type>{type::real});
  require(std::string{bra::classical_function_name(function::exp)} == "EXP");
  require(std::string{bra::classical_function_name(function::arg)} == "ARG");
  require(std::string{bra::classical_function_name(function::abs)} == "ABS");
  require(bra::classical_function_from_name("ABS") == function::abs);
  require(&bra::resolve_classical_function(function::abs, {type::complex_}) == &signatures[3u]);
  require(std::string{bra::classical_function_name(function::sqrt)} == "SQRT");
  require(bra::classical_function_from_name("SQRT") == function::sqrt);
  require(bra::classical_function_from_name("COMPLEX") == function::complex_);
  require(bra::classical_function_from_name("LOG") == function::log);
  require(std::string{bra::classical_function_name(function::log)} == "LOG");
  require(bra::resolve_classical_function(function::log, {type::real}).result_types == std::vector<type>{type::real});
  for (auto const& operands: {std::vector<type>{}, std::vector<type>{type::integer},
                             std::vector<type>{type::unsigned_integer}, std::vector<type>{type::complex_},
                             std::vector<type>{type::real, type::real}})
  {
    auto rejected = false;
    try { bra::resolve_classical_function(function::log, operands); }
    catch (std::invalid_argument const&) { rejected = true; }
    require(rejected);
  }
  auto const& constructor = bra::resolve_classical_function(function::complex_, {type::real, type::real});
  require(constructor.result_types == std::vector<type>{type::complex_});
  for (auto const& operands: {std::vector<type>{type::real}, std::vector<type>{type::real, type::integer},
                             std::vector<type>{type::complex_, type::real},
                             std::vector<type>{type::real, type::real, type::real}})
  {
    auto rejected = false;
    try { bra::resolve_classical_function(function::complex_, operands); }
    catch (std::invalid_argument const&) { rejected = true; }
    require(rejected);
  }
  for (auto const operand: {type::real, type::complex_})
  {
    auto const& signature = bra::resolve_classical_function(function::sqrt, {operand});
    require(signature.result_types == std::vector<type>{operand});
  }
  for (auto const& operands: {std::vector<type>{}, std::vector<type>{type::integer},
                            std::vector<type>{type::unsigned_integer}, std::vector<type>{type::real, type::real}})
  {
    auto rejected = false;
    try { bra::resolve_classical_function(function::sqrt, operands); }
    catch (std::invalid_argument const&) { rejected = true; }
    require(rejected);
  }
  for (auto first = signatures.begin(); first != signatures.end(); ++first)
    for (auto second = first + 1; second != signatures.end(); ++second)
      require(first->function != second->function or first->operand_types != second->operand_types);

  // The shared type extraction must preserve the existing enum values.
  static_assert(static_cast<int>(type::real) == 0, "REAL type changed");
  static_assert(static_cast<int>(type::complex_) == 1, "COMPLEX type changed");
  static_assert(static_cast<int>(type::integer) == 2, "INT type changed");
  static_assert(static_cast<int>(type::pauli_string_space) == 3, "PAULISS type changed");
  static_assert(static_cast<int>(type::unsigned_integer) == 4, "UINT type changed");

  // Multiple and differently typed results require no change to the schema.
  auto const future_signature = bra::classical_function_signature{
    function::exp, {type::real}, {type::real, type::integer}};
  require(future_signature.result_types.size() == 2u);
  require(future_signature.result_types[1u] == type::integer);
}
