// Compile with bra's macros and link with non-MPI objects excluding bra.o.
#include <limits>
#include <cmath>
#include <iomanip>
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
  require(bra::classical_function_from_name("ABS") == function::abs);
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
  state.generate_new_variable("RESULT", type::real, 1);
  state.generate_new_variable("CRESULT", type::complex_, 1);
  state.invoke_assign_operation("R:0", bra::assign_operation_type::assign, "3");
  state.invoke_assign_operation("R:1", bra::assign_operation_type::assign, "7");
  state.invoke_assign_operation("N:0", bra::assign_operation_type::assign, "1");
  state.invoke_assign_operation("N:1", bra::assign_operation_type::assign, "0");
  state.invoke_assign_operation("Z:0", bra::assign_operation_type::assign, ":I");
  state.invoke_assign_operation("Z:1", bra::assign_operation_type::assign, ":COMPLEX:1");
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
  auto check_close = [&](std::string const& name, bra::real_type const expected)
  {
    auto const tolerance = 32 * std::numeric_limits<bra::real_type>::epsilon() * (1 + std::abs(expected));
    auto lower = std::ostringstream{}, upper = std::ostringstream{};
    lower << std::setprecision(std::numeric_limits<bra::real_type>::max_digits10) << expected - tolerance;
    upper << std::setprecision(std::numeric_limits<bra::real_type>::max_digits10) << expected + tolerance;
    state.invoke_assert_operation(name, bra::compare_operation_type::greater_equal, lower.str());
    state.invoke_assert_operation(name, bra::compare_operation_type::less_equal, upper.str());
  };
  state.invoke_apply_operation({"R:N"}, function::mod, {"R:N", "R:N:N"});
  check_close("R:1", 1);
  state.invoke_assign_operation("R:1", bra::assign_operation_type::assign, "7");
  state.invoke_apply_operation({"RESULT"}, function::mod, {"R:N:N", "R:N"});
  check_close("RESULT", 3);
  for (auto const& operands: {std::vector<std::string>{"R", "0.0"},
                             std::vector<std::string>{"R", "-0.0"}})
    check_call({"R:N"}, function::mod, operands, "MOD domain error: zero divisor");
  for (auto const& operands: {std::vector<std::string>{"R"},
                             std::vector<std::string>{"R", "R", "R"}})
    check_call({"R:N"}, function::mod, operands, "wrong operand count");
  for (auto const& operands: {std::vector<std::string>{"N", "R"},
                             std::vector<std::string>{"R", "U"},
                             std::vector<std::string>{"Z", "R"}})
    check_call({"R:N"}, function::mod, operands, "unsupported operand signature");
  check_call({"N"}, function::mod, {"R", "R"}, "wrong destination type");
  check_call({"R:N"}, function::mod, {"R", "R:N:999"}, "index out of range");
  for (auto const& entry: std::vector<std::pair<std::string, bra::real_type>>{
      {"R", 3}, {"R:N", 7}, {"R:N:N", 3}, {":REAL:N", 1}, {":REAL:U", 0}, {":REAL:Z:1", 1},
      {":IMAG:Z:N", 0}, {"1.0", 1}, {".5", 0.5}, {"-1E+2", -100}, {":REAL:R::INT:1.9", 7}})
  {
    state.invoke_apply_operation({"RESULT"}, function::exp, {entry.first});
    check_close("RESULT", std::exp(entry.second));
  }
  for (auto const& operand: {"Z", "Z:N:N", ":COMPLEX:R:N", ":I"})
    state.invoke_apply_operation({"CRESULT"}, function::exp, {operand});
  state.invoke_apply_operation({"RESULT"}, function::arg, {"Z:N"});
  check_close("RESULT", 0);
  state.invoke_apply_operation({"R:N"}, function::abs, {"Z:N:N"});
  check_close("R:1", 1);
  state.invoke_assign_operation("R:1", bra::assign_operation_type::assign, "7");
  state.invoke_apply_operation({"RESULT"}, function::abs, {":COMPLEX:R:N"});
  check_close("RESULT", 7);
  state.invoke_apply_operation({"RESULT"}, function::sqrt, {"R:N"});
  check_close("RESULT", std::sqrt(bra::real_type{7}));
  for (auto const id: {function::sin, function::cos, function::tan})
  {
    state.invoke_apply_operation({"RESULT"}, id, {"R:N"});
    auto const input = bra::real_type{7};
    auto const expected = id == function::sin ? std::sin(input)
      : id == function::cos ? std::cos(input) : std::tan(input);
    check_close("RESULT", expected);
    check_call({"R:N"}, id, {"N"}, "unsupported operand signature");
    check_call({"R:N"}, id, {"Z"}, "unsupported operand signature");
    check_call({"CRESULT"}, id, {"R"}, "wrong destination type");
  }
  for (auto const id: {function::arcsin, function::arccos, function::arctan})
  {
    state.invoke_apply_operation({"RESULT"}, id, {":REAL:0.5"});
    auto const input = bra::real_type{0.5};
    auto const expected = id == function::arcsin ? std::asin(input)
      : id == function::arccos ? std::acos(input) : std::atan(input);
    check_close("RESULT", expected);
    check_call({"R:N"}, id, {"N"}, "unsupported operand signature");
    check_call({"R:N"}, id, {"Z"}, "unsupported operand signature");
    check_call({"CRESULT"}, id, {"R"}, "wrong destination type");
    if (id != function::arctan)
      check_call({"R:N"}, id, {"R:N"}, std::string{bra::classical_function_name(id)}
                 + " domain error: real input outside [-1, 1]");
  }
  for (auto const id: {function::ceiling, function::floor})
  {
    state.invoke_apply_operation({"RESULT"}, id, {":REAL:0.5"});
    check_close("RESULT", id == function::ceiling ? 1 : 0);
    state.invoke_apply_operation({"RESULT"}, id, {"RESULT"});
    check_close("RESULT", id == function::ceiling ? 1 : 0);
    check_call({"R:N"}, id, {"N"}, "unsupported operand signature");
    check_call({"R:N"}, id, {"Z"}, "unsupported operand signature");
    check_call({"CRESULT"}, id, {"R"}, "wrong destination type");
  }
  state.invoke_apply_operation({"RESULT"}, function::log, {"R:N"});
  check_close("RESULT", std::log(bra::real_type{7}));
  state.invoke_apply_operation({"RESULT"}, function::log, {"RESULT"});
  check_close("RESULT", std::log(std::log(bra::real_type{7})));
  check_call({"R:N"}, function::log, {":REAL:-1"}, "LOG domain error: nonpositive real input");
  check_call({"R:N"}, function::log, {":REAL:-0.0"}, "LOG domain error: nonpositive real input");
  check_call({"R:N"}, function::log, {"N"}, "unsupported operand signature");
  check_call({"R:N"}, function::log, {"Z"}, "unsupported operand signature");
  check_call({"CRESULT"}, function::log, {"R"}, "wrong destination type");
  state.invoke_apply_operation({"CRESULT"}, function::sqrt, {":COMPLEX:-4.0"});
  state.invoke_assign_operation("RESULT", bra::assign_operation_type::assign, ":REAL:CRESULT");
  check_close("RESULT", 0);
  state.invoke_assign_operation("RESULT", bra::assign_operation_type::assign, ":IMAG:CRESULT");
  check_close("RESULT", 2);
  state.invoke_apply_operation({"CRESULT"}, function::complex_, {"R:N:N", ":REAL:CRESULT"});
  state.invoke_assign_operation("RESULT", bra::assign_operation_type::assign, ":REAL:CRESULT");
  check_close("RESULT", 3);
  state.invoke_assign_operation("RESULT", bra::assign_operation_type::assign, ":IMAG:CRESULT");
  check_close("RESULT", 0);
  check_call({"RESULT"}, function::complex_, {"R", "R"}, "wrong destination type");
  check_call({"CRESULT"}, function::complex_, {"N", "R"}, "unsupported operand signature");
  check_call({"CRESULT"}, function::complex_, {"R"}, "wrong operand count");
  check_call({"CRESULT"}, function::complex_, {"R", "R:999"}, "index out of range");
  check_call({"RESULT"}, function::sqrt, {"N"}, "unsupported operand signature");
  check_call({"CRESULT"}, function::sqrt, {"R:N"}, "wrong destination type");
  check_call({"RESULT"}, function::sqrt, {"Z"}, "wrong destination type");
  check_call({"R:N"}, function::sqrt, {":REAL:-1"}, "SQRT domain error: negative real input");
  state.invoke_assign_operation("RESULT", bra::assign_operation_type::assign, "-7");
  state.invoke_apply_operation({"RESULT"}, function::abs, {":COMPLEX:RESULT"});
  check_close("RESULT", 7);
  check_call({"R"}, function::abs, {"R"}, "unsupported operand signature");
  check_call({"R"}, function::abs, {"N"}, "unsupported operand signature");
  check_call({"R"}, function::abs, {"U"}, "unsupported operand signature");
  check_call({"Z"}, function::abs, {"Z"}, "wrong destination type");
  check_call({"R", "R:1"}, function::abs, {"Z"}, "wrong destination count");
  check_call({"R"}, function::abs, {}, "wrong operand count");
  check_call({"R"}, function::abs, {"Z", "Z"}, "wrong operand count");
  check_call({"R:N:999"}, function::abs, {"Z"}, "index out of range");
  check_call({"R"}, function::abs, {"Z:N:999"}, "index out of range");
  state.invoke_apply_operation({"R:N:N"}, function::exp, {"R:N:N"});
  check_close("R:0", std::exp(bra::real_type{3}));
  state.invoke_assign_operation("R:0", bra::assign_operation_type::assign, "3");
  check_call({"R:N"}, function::exp, {":REAL:4294967295"}, "overflow/nonfinite result");
  check_call({"Z:1"}, function::exp, {":COMPLEX:4294967295"}, "overflow/nonfinite result");
  check_call({"R:N"}, function::arg, {":COMPLEX:0"}, "ARG domain error");
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
  // Exercise numeric failures through the state/interpreter-facing entry point,
  // including separate complex destinations and an in-place failed invocation.
  state.invoke_assign_operation("RESULT", bra::assign_operation_type::assign, "1.0");
  state.invoke_assign_operation("RESULT", bra::assign_operation_type::divides_assign, "0.0");
  check_call({"R:N"}, function::mod, {"RESULT", "R"}, "MOD domain error: nonfinite input");
  check_call({"R:N"}, function::mod, {"R", "RESULT"}, "MOD domain error: nonfinite input");
  state.invoke_assign_operation("CRESULT", bra::assign_operation_type::assign, ":COMPLEX:RESULT");
  check_call({"R:N"}, function::exp, {"RESULT"}, "nonfinite input");
  check_call({"R:N"}, function::arg, {"CRESULT"}, "nonfinite input");
  check_call({"R:N"}, function::abs, {"CRESULT"}, "nonfinite input");
  check_call({"Z:1"}, function::exp, {"CRESULT"}, "nonfinite input");
  state.generate_new_variable("INSPECT", type::real, 1);
  state.invoke_assign_operation("INSPECT", bra::assign_operation_type::assign, ":REAL:Z:1");
  state.invoke_assert_operation("INSPECT", bra::compare_operation_type::equal_to, "1");
  state.invoke_assign_operation("INSPECT", bra::assign_operation_type::assign, ":IMAG:Z:1");
  state.invoke_assert_operation("INSPECT", bra::compare_operation_type::equal_to, "0");
  state.invoke_assign_operation("RESULT", bra::assign_operation_type::assign, "1000000");
  expect_failure([&] { state.invoke_apply_operation({"RESULT"}, function::exp, {"RESULT"}); },
                 "APPLY RESULT <- EXP RESULT: overflow/nonfinite result");
  state.invoke_assert_operation("RESULT", bra::compare_operation_type::equal_to, "1000000");
  state.invoke_assign_operation("RESULT", bra::assign_operation_type::assign, "-4");
  expect_failure([&] { state.invoke_apply_operation({"RESULT"}, function::sqrt, {"RESULT"}); },
                 "APPLY RESULT <- SQRT RESULT: SQRT domain error");
  state.invoke_assert_operation("RESULT", bra::compare_operation_type::equal_to, "-4");
  expect_failure([&] { state.invoke_apply_operation({"RESULT"}, function::log, {"RESULT"}); },
                 "APPLY RESULT <- LOG RESULT: LOG domain error: nonpositive real input");
  state.invoke_assert_operation("RESULT", bra::compare_operation_type::equal_to, "-4");
  auto maximum = std::ostringstream{};
  maximum << std::setprecision(std::numeric_limits<bra::real_type>::max_digits10)
          << std::numeric_limits<bra::real_type>::max();
  state.invoke_assign_operation("RESULT", bra::assign_operation_type::assign, maximum.str());
  state.invoke_assign_operation("CRESULT", bra::assign_operation_type::assign, ":I");
  state.invoke_assign_operation("CRESULT", bra::assign_operation_type::multiplies_assign, ":COMPLEX:RESULT");
  state.invoke_assign_operation("CRESULT", bra::assign_operation_type::plus_assign, ":COMPLEX:RESULT");
  check_call({"R:N"}, function::abs, {"CRESULT"}, "overflow/nonfinite result");
  state.generate_new_variable("R", type::integer, 1);
  expect_failure([&] { state.invoke_apply_operation({"Z"}, function::exp, {"R"}); }, "ambiguous variable type");

  auto stream = std::istringstream{"QUBITS 1\napply very_long_destination:index <- exp :complex:source:index\n"};
  auto const interpreter = bra::interpreter{stream};
  require(tokens(interpreter.back_circuit().back()->representation()) == std::vector<std::string>{
    "APPLY", "VERY_LONG_DESTINATION:INDEX", "<-", "EXP", ":COMPLEX:SOURCE:INDEX"});
  auto abs_stream = std::istringstream{"QUBITS 1\napply result:index <- abs source:index\n"};
  auto const abs_interpreter = bra::interpreter{abs_stream};
  require(tokens(abs_interpreter.back_circuit().back()->representation()) == std::vector<std::string>{
    "APPLY", "RESULT:INDEX", "<-", "ABS", "SOURCE:INDEX"});
  auto sqrt_stream = std::istringstream{"QUBITS 1\napply result:index <- sqrt source:index\n"};
  auto const sqrt_interpreter = bra::interpreter{sqrt_stream};
  require(tokens(sqrt_interpreter.back_circuit().back()->representation()) == std::vector<std::string>{
    "APPLY", "RESULT:INDEX", "<-", "SQRT", "SOURCE:INDEX"});
  auto log_stream = std::istringstream{"QUBITS 1\napply result:index <- log source:index\n"};
  auto const log_interpreter = bra::interpreter{log_stream};
  require(tokens(log_interpreter.back_circuit().back()->representation()) == std::vector<std::string>{
    "APPLY", "RESULT:INDEX", "<-", "LOG", "SOURCE:INDEX"});
  for (auto const& malformed: {"APPLY", "APPLY R", "APPLY <- EXP R", "APPLY R <-", "APPLY R <- EXP", "APPLY R := EXP R",
                               "APPLY R <- EXP <- R", "APPLY R<-EXP R"})
    expect_failure([&] { auto input = std::istringstream{std::string{"QUBITS 1\n"} + malformed + "\n"};
                        bra::interpreter parsed{input}; }, "APPLY");
  expect_failure([] { auto input = std::istringstream{"QUBITS 1\nAPPLY R <- BOGUS X\n"};
                      bra::interpreter parsed{input}; }, "unknown function BOGUS");
}
