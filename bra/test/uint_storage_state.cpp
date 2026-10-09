// Compile with bra's macros and link with non-MPI objects excluding bra.o.
#include <limits>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <bra/nompi_state.hpp>
#include <bra/state.hpp>
#include <bra/types.hpp>

namespace
{
  void check_value(bra::state& state, std::string const& operand, std::string const& expected)
  {
    auto stream = std::ostringstream{};
    auto const original = std::cout.rdbuf(stream.rdbuf());
    try { state.invoke_print_operation({operand}); }
    catch (...) { std::cout.rdbuf(original); throw; }
    std::cout.rdbuf(original);
    if (stream.str() != expected)
      throw std::runtime_error{"incorrect UINT value: " + operand};
  }

  template <typename Function>
  void expect_failure(Function const& function)
  {
    auto caught = false;
    try { function(); }
    catch (...) { caught = true; }
    if (not caught)
      throw std::runtime_error{"invalid UINT operation accepted"};
  }
}

int main()
{
  static_assert(static_cast<int>(bra::variable_type::integer) == 2, "INT enum changed");
  static_assert(static_cast<int>(bra::variable_type::pauli_string_space) == 3, "PAULISS enum changed");
  auto state = bra::nompi_state{0u, 0u, 1u, 16u, 1u, false, 0.0, 0.0, 0.0, false, 0u, 0};
  using type = bra::variable_type;
  using operation = bra::assign_operation_type;
  auto const maximum = std::numeric_limits<bra::uint_type>::max();
  state.generate_new_variable("A", type::unsigned_integer, 2);
  state.generate_new_variable("INDEX", type::integer, 1);
  state.generate_new_variable("SIGNED", type::integer, 1);
  state.invoke_assign_operation("INDEX", operation::assign, "1");
  state.invoke_assign_operation("SIGNED", operation::assign, "-1");
  check_value(state, "A:0", "0");
  check_value(state, "A:INDEX", "0");
  state.invoke_assign_operation("A:INDEX", operation::assign, std::to_string(maximum));
  state.invoke_jump_operation("PENDING");
  for (auto const& operand: {"-1", "1.5", "SIGNED", "MISSING", "A:-1", "A:2"})
  {
    expect_failure([&] { state.invoke_assign_operation("A:INDEX", operation::assign, operand); });
    check_value(state, "A:0", "0");
    check_value(state, "A:INDEX", std::to_string(maximum));
  }
  for (auto const& literal: {"", "+", "++1", " 1", "1 ", "1.0", "-0"})
    expect_failure([&] { state.invoke_assign_operation("A:INDEX", operation::assign, literal); });
  expect_failure([&] { state.invoke_assign_operation("A:INDEX", operation::assign, std::to_string(maximum) + "0"); });
  check_value(state, "A:INDEX", std::to_string(maximum));
  expect_failure([&] { state.invoke_assign_operation("A:2", operation::assign, "0"); });
  expect_failure([&] { state.generate_new_variable("A", type::unsigned_integer, 1); });
  expect_failure([&] { state.generate_new_variable("A", type::integer, 1); });
  expect_failure([&] { state.generate_new_variable("SIGNED", type::unsigned_integer, 1); });
  expect_failure([&] { state.generate_new_variable("EMPTY", type::unsigned_integer, 0); });
  state.generate_new_variable("EMPTY", type::unsigned_integer, 1);
  state.invoke_assign_operation("A:0", operation::assign, "A:INDEX");
  state.invoke_assign_operation("A:INDEX", operation::assign, "A:INDEX");
  check_value(state, "A:0", std::to_string(maximum));
  check_value(state, "A:1", std::to_string(maximum));
  state.invoke_assert_operation("SIGNED", bra::compare_operation_type::equal_to, "-1");
  if (not state.maybe_label() or *state.maybe_label() != "PENDING")
    throw std::runtime_error{"UINT assignment changed pending jump state"};
}
