// Compile with bra's macros and link with non-MPI objects excluding bra.o.
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <bra/classical_function_call.hpp>
#include <bra/interpreter.hpp>
#include <bra/nompi_state.hpp>

namespace
{
  void require(bool const condition)
  {
    if (not condition) throw std::runtime_error{"APPLY infrastructure test failed"};
  }

  template <typename Function>
  void expect_failure(Function const& function, std::string const& reason)
  {
    try { function(); }
    catch (std::exception const& error)
    {
      if (std::string{error.what()}.find(reason) == std::string::npos)
        throw std::runtime_error{"expected diagnostic '" + reason + "', got '" + error.what() + "'"};
      return;
    }
    throw std::runtime_error{"expected APPLY failure: " + reason};
  }

  auto tokens(std::string const& text) -> std::vector<std::string>
  {
    auto stream = std::istringstream{text};
    auto result = std::vector<std::string>{};
    for (auto token = std::string{}; stream >> token;) result.push_back(token);
    return result;
  }
}

int main()
{
  using type = bra::variable_type;
  using function = bra::classical_function_id;
  require(bra::classical_function_from_name("EXP") == function::exp);
  require(bra::resolve_classical_function(function::arg, {type::complex_}).result_types == std::vector<type>{type::real});
  expect_failure([] { bra::classical_function_from_name("BOGUS"); }, "unknown function");
  expect_failure([] { bra::resolve_classical_function(function::exp, {}); }, "wrong operand count");
  expect_failure([] { bra::resolve_classical_function(function::exp, {type::integer}); }, "unsupported operand signature");

  auto x = bra::real_type{2}, y = bra::real_type{7};
  auto n = bra::int_type{1};
  auto const& exp = bra::resolve_classical_function(function::exp, {type::real});
  auto const call = bra::classical_function_call{exp, {&x}, {x}};
  x = 99;
  require(boost::get<bra::real_type>(call.operands().front()) == 2);
  call.commit_results({bra::real_type{3}});
  require(x == 3);
  expect_failure([&] { call.commit_results({n}); }, "wrong result type");
  expect_failure([&] { call.commit_results({}); }, "wrong result count");
  require(x == 3);
  expect_failure([&] { bra::classical_function_call{exp, {&n}, {x}}; }, "wrong destination type");
  expect_failure([&] { bra::classical_function_call{exp, {}, {x}}; }, "wrong destination count");
  expect_failure([&] { bra::classical_function_call{exp, {&x}, {n}}; }, "wrong operand type");
  expect_failure([&] { bra::classical_function_call{exp, {static_cast<bra::real_type*>(nullptr)}, {x}}; }, "null destination");

  // Synthetic multi-result signature exercises generic storage, not a new
  // registered mathematical function or overload.
  auto const multiple = bra::classical_function_signature{function::exp, {type::real}, {type::real, type::real}};
  expect_failure([&] { bra::classical_function_call{multiple, {&x, &x}, {x}}; }, "duplicate destination");
  auto const pair = bra::classical_function_call{multiple, {&x, &y}, {x}};
  expect_failure([&] { pair.commit_results({bra::real_type{4}, n}); }, "wrong result type");
  require(x == 3 and y == 7);
  pair.commit_results({bra::real_type{4}, bra::real_type{5}});
  require(x == 4 and y == 5);
  auto const mixed = bra::classical_function_signature{function::exp, {type::real}, {type::integer, type::real}};
  bra::classical_function_call{mixed, {&n, &x}, {x}}.commit_results({bra::int_type{2}, bra::real_type{6}});
  require(n == 2 and x == 6);

  auto state = bra::nompi_state{0u, 0u, 1u, 16u, 1u, false, 0.0, 0.0, 0.0, false, 0u, 0};
  state.generate_new_variable("R", type::real, 2);
  state.generate_new_variable("Z", type::complex_, 2);
  state.generate_new_variable("N", type::integer, 2);
  state.generate_new_variable("U", type::unsigned_integer, 1);
  state.generate_new_variable("P", type::pauli_string_space, 1);
  state.invoke_assign_operation("R:0", bra::assign_operation_type::assign, "3");
  state.invoke_assign_operation("R:1", bra::assign_operation_type::assign, "7");
  state.invoke_assign_operation("N:0", bra::assign_operation_type::assign, "1");
  state.invoke_assign_operation("N:1", bra::assign_operation_type::assign, "0");
  state.invoke_jump_operation("PENDING");

  auto check_call = [&](std::vector<std::string> const& destinations, function const fn,
                        std::vector<std::string> const& operands, std::string const& reason)
  {
    auto const instruction = bra::apply_instruction(destinations, fn, operands);
    expect_failure([&] { state.invoke_apply_operation(destinations, fn, operands); }, instruction + ": " + reason);
    state.invoke_assert_operation("R:0", bra::compare_operation_type::equal_to, "3");
    state.invoke_assert_operation("R:1", bra::compare_operation_type::equal_to, "7");
    require(state.maybe_label() and *state.maybe_label() == "PENDING");
  };
  for (auto const& operand: {"R", "R:N", "R:N:N", ":REAL:N", ":REAL:U", ":REAL:Z:1", ":IMAG:Z:N",
                            ":REAL:4294967295", ":PI", "1.0", ".5", "-1E+2", ":REAL:R::INT:1.9"})
    check_call({"R:N"}, function::exp, {operand}, "numerical function evaluation is not implemented yet");
  for (auto const& operand: {"Z", "Z:N:N", ":COMPLEX:R:N", ":COMPLEX:4294967295", ":I"})
    check_call({"Z:1"}, function::exp, {operand}, "numerical function evaluation is not implemented yet");
  check_call({"R"}, function::arg, {"Z:N"}, "numerical function evaluation is not implemented yet");
  check_call({"R"}, function::exp, {}, "wrong operand count");
  check_call({}, function::exp, {"R"}, "wrong destination count");
  check_call({"R", "R:0"}, function::exp, {"R"}, "wrong destination count");
  check_call({"N"}, function::exp, {"R"}, "wrong destination type");
  check_call({"Z"}, function::exp, {"R"}, "wrong destination type");
  check_call({":REAL:0"}, function::exp, {"R"}, "destination must be a mutable variable");
  check_call({"R"}, function::exp, {"N"}, "unsupported operand signature");
  check_call({"R"}, function::exp, {"U"}, "unsupported operand signature");
  check_call({"R"}, function::exp, {"1"}, "unsupported operand signature");
  check_call({"R"}, function::exp, {"P"}, "unsupported operand signature");
  check_call({"R"}, function::arg, {"R"}, "unsupported operand signature");
  check_call({"MISSING"}, function::exp, {"R"}, "unknown variable");
  check_call({"R"}, function::exp, {"MISSING"}, "unknown variable");
  check_call({"R:-1"}, function::exp, {"R"}, "negative index");
  check_call({"R:U"}, function::exp, {"R"}, "index requires INT");
  check_call({"R"}, function::exp, {":REAL:R:-1"}, "negative index");
  check_call({"R"}, function::exp, {":REAL:R:N:99"}, "index out of range");
  check_call({"R"}, function::exp, {":REAL:"}, "empty symbol argument");
  check_call({"R"}, function::exp, {"1e"}, "invalid numeric literal");
  check_call({"R"}, function::exp, {"1.0x"}, "invalid numeric literal");
  check_call({"R"}, function::exp, {":REAL:R::INT:1e100"}, "REAL to INT conversion out of range");
  state.generate_new_variable("R", type::integer, 1);
  expect_failure([&] { state.invoke_apply_operation({"Z"}, function::exp, {"R"}); }, "ambiguous variable type");

  auto stream = std::istringstream{"QUBITS 1\napply very_long_destination:index <- exp :complex:source:index\n"};
  auto const interpreter = bra::interpreter{stream};
  require(tokens(interpreter.back_circuit().back()->representation()) == std::vector<std::string>{
    "APPLY", "VERY_LONG_DESTINATION:INDEX", "<-", "EXP", ":COMPLEX:SOURCE:INDEX"});
  for (auto const& malformed: {"APPLY", "APPLY R", "APPLY <- EXP R", "APPLY R <-", "APPLY R <- EXP", "APPLY R := EXP R",
                               "APPLY R <- EXP <- R", "APPLY R<-EXP R"})
    expect_failure([&] { auto input = std::istringstream{std::string{"QUBITS 1\n"} + malformed + "\n"};
                        bra::interpreter parsed{input}; }, "APPLY");
  expect_failure([] { auto input = std::istringstream{"QUBITS 1\nAPPLY R <- BOGUS X\n"};
                      bra::interpreter parsed{input}; }, "unknown function BOGUS");
}
