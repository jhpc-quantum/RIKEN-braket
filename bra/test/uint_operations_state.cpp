// Compile with bra's macros and link with non-MPI objects excluding bra.o.
#include <limits>
#include <stdexcept>
#include <string>

#include <bra/nompi_state.hpp>
#include <bra/state.hpp>
#include <bra/types.hpp>

namespace
{
  template <typename Function>
  void expect_failure(Function const& function)
  {
    auto caught = false;
    try { function(); }
    catch (...) { caught = true; }
    if (not caught)
      throw std::runtime_error{"invalid UINT operation accepted"};
  }

  template <typename Exception, typename Function>
  void expect_error(Function const& function, std::string const& diagnostic)
  {
    auto caught = false;
    try { function(); }
    catch (Exception const& error)
    {
      caught = true;
      if (std::string{error.what()}.find(diagnostic) == std::string::npos)
        throw std::runtime_error{"incorrect UINT error diagnostic"};
    }
    if (not caught)
      throw std::runtime_error{"UINT error not thrown"};
  }
}

int main()
{
  auto state = bra::nompi_state{0u, 0u, 1u, 16u, 1u, false, 0.0, 0.0, 0.0, false, 0u, 0};
  using operation = bra::assign_operation_type;
  using comparison = bra::compare_operation_type;
  using type = bra::variable_type;
  auto const maximum = std::to_string(std::numeric_limits<bra::uint_type>::max());
  state.generate_new_variable("A", type::unsigned_integer, 2);
  state.generate_new_variable("INDEX", type::integer, 1);
  state.generate_new_variable("SIGNED", type::integer, 1);
  state.generate_new_variable("R", type::real, 1);
  state.generate_new_variable("Z", type::complex_, 1);
  state.generate_new_variable("P", type::pauli_string_space, 1);
  state.invoke_assign_operation("INDEX", operation::assign, "1");
  state.invoke_assign_operation("A:0", operation::assign, maximum);
  state.invoke_assign_operation("SIGNED", operation::assign, "17");
  state.invoke_jump_operation("PENDING");
  for (auto const op: {operation::assign, operation::plus_assign, operation::minus_assign,
                      operation::multiplies_assign, operation::divides_assign,
                      operation::bit_and_assign, operation::bit_or_assign, operation::bit_xor_assign})
  {
    state.invoke_assign_operation("A:1", operation::assign, "7");
    for (auto const& operand: {"SIGNED", "R", "Z", "P", "-1", "1.5", "MISSING",
                              "A:2", ":INT:1", ":UINT::REAL:-1.0"})
    {
      expect_failure([&] { state.invoke_assign_operation("A:INDEX", op, operand); });
      state.invoke_assert_operation("A:0", comparison::equal_to, maximum);
      state.invoke_assert_operation("A:INDEX", comparison::equal_to, "7");
    }
  }
  expect_error<bra::integer_zero_divisor_error>(
    [&] { state.invoke_assign_operation("A:INDEX", operation::divides_assign, ":UINT:0.5"); },
    "integer division by zero in LET A:INDEX /= :UINT:0.5");
  state.invoke_assert_operation("A:INDEX", comparison::equal_to, "7");
  expect_error<std::out_of_range>(
    [&] { state.invoke_assign_operation("SIGNED", operation::assign, ":INT:A:0"); },
    "UINT to INT conversion out of range");
  state.invoke_assert_operation("SIGNED", comparison::equal_to, "17");
  for (auto const& numerator: {"0", "1", "-1"})
  {
    state.invoke_assign_operation("R", operation::assign, numerator);
    state.invoke_assign_operation("R", operation::divides_assign, "0");
    expect_error<std::out_of_range>(
      [&] { state.invoke_assign_operation("A:INDEX", operation::assign, ":UINT:R"); },
      "REAL to UINT conversion out of range");
    state.invoke_assert_operation("A:INDEX", comparison::equal_to, "7");
  }
  state.invoke_jump_operation("WRONG", "A:0", comparison::less, "1");
  if (not state.maybe_label() or *state.maybe_label() != "PENDING")
    throw std::runtime_error{"UINT operations changed pending jump state"};
  state.invoke_jump_operation("MATCHED", "A:0", comparison::greater, "A:INDEX");
  expect_error<bra::assertion_error>(
    [&] { state.invoke_assert_operation("A:0", comparison::less, "1"); },
    "evaluated: " + maximum + " < 1");
  expect_failure([&] { state.invoke_jump_operation("WRONG", "A:0", comparison::equal_to, "SIGNED"); });
  if (not state.maybe_label() or *state.maybe_label() != "MATCHED")
    throw std::runtime_error{"failed UINT comparison changed pending jump state"};
  state.invoke_assign_operation("A:INDEX", operation::assign, ":UINT:-1");
  state.invoke_assign_operation("A:INDEX", operation::plus_assign, "1");
  state.invoke_assert_operation("A:INDEX", comparison::equal_to, "0");
}
