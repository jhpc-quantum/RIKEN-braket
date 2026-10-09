// Compile with bra's macros and link with non-MPI objects excluding bra.o.
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <bra/gate/let_op.hpp>
#include <bra/nompi_state.hpp>
#include <bra/state.hpp>

namespace
{
  template <typename Function>
  void expect_failure(Function const& function)
  {
    auto caught = false;
    try { function(); }
    catch (...) { caught = true; }
    if (not caught)
      throw std::runtime_error{"invalid shift operand accepted"};
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
        throw std::runtime_error{"incorrect shift diagnostic"};
    }
    if (not caught)
      throw std::runtime_error{"invalid shift accepted"};
  }
}

int main()
{
  auto state = bra::nompi_state{0u, 0u, 1u, 16u, 1u, false, 0.0, 0.0, 0.0, false, 0u, 0};
  using operation = bra::assign_operation_type;
  using comparison = bra::compare_operation_type;
  auto const width = std::numeric_limits<bra::uint_type>::digits;
  auto const minimum = std::to_string(std::numeric_limits<bra::int_type>::min());
  auto const maximum = std::to_string(std::numeric_limits<bra::uint_type>::max());
  state.generate_new_variable("A", bra::variable_type::integer, 2);
  state.generate_new_variable("U", bra::variable_type::unsigned_integer, 2);
  state.generate_new_variable("INDEX", bra::variable_type::integer, 1);
  state.generate_new_variable("R", bra::variable_type::real, 1);
  state.generate_new_variable("Z", bra::variable_type::complex_, 1);
  state.generate_new_variable("P", bra::variable_type::pauli_string_space, 1);
  state.invoke_assign_operation("INDEX", operation::assign, "1");
  state.invoke_assign_operation("A:0", operation::assign, "3");
  state.invoke_assign_operation("U:0", operation::assign, maximum);
  state.invoke_jump_operation("PENDING");
  for (auto const op: {operation::left_shift_assign, operation::right_shift_assign})
  {
    auto const symbol = op == operation::left_shift_assign ? "<<=" : ">>=";
    bra::gate::let_op const gate{"A", op, "3"};
    auto stream = std::istringstream{gate.representation()};
    auto mnemonic = std::string{}, lhs = std::string{}, rendered_op = std::string{}, rhs = std::string{};
    stream >> mnemonic >> lhs >> rendered_op >> rhs;
    if (mnemonic != "LET" or lhs != "A" or rendered_op != symbol or rhs != "3")
      throw std::runtime_error{"incorrect shift LET representation"};
    for (auto const& target: {"R", "Z", "P", "MISSING", ""})
      expect_error<bra::wrong_assignment_argument_error>(
        [&] { state.invoke_assign_operation(target, op, "1"); },
        std::string{"\""} + target + " " + symbol + " 1\" is a wrong argument");
    for (auto const& target: {"A:INDEX", "U:INDEX"})
    {
      state.invoke_assign_operation(target, operation::assign, "7");
      for (auto const& count: {std::string{"-1"}, std::to_string(width), maximum})
      {
        expect_error<std::out_of_range>([&] { state.invoke_assign_operation(target, op, count); }, "shift count");
        state.invoke_assert_operation(target, comparison::equal_to, "7");
      }
      for (auto const& count: {"R", "Z", "P", ":REAL", "1.5", "MISSING"})
      {
        expect_failure([&] { state.invoke_assign_operation(target, op, count); });
        state.invoke_assert_operation(target, comparison::equal_to, "7");
      }
    }
    state.invoke_assert_operation("A:0", comparison::equal_to, "3");
    state.invoke_assert_operation("U:0", comparison::equal_to, maximum);
    if (not state.maybe_label() or *state.maybe_label() != "PENDING")
      throw std::runtime_error{"shift changed pending jump state"};
  }
  for (auto const& value: {minimum, std::to_string(std::numeric_limits<bra::int_type>::max())})
  {
    state.invoke_assign_operation("A:INDEX", operation::assign, value);
    expect_error<std::overflow_error>(
      [&] { state.invoke_assign_operation("A:INDEX", operation::left_shift_assign, "1"); },
      "integer left shift overflow in LET A:INDEX <<= 1");
    state.invoke_assert_operation("A:INDEX", comparison::equal_to, value);
  }
  state.invoke_assign_operation("A:INDEX", operation::assign, "-1");
  state.invoke_assign_operation("A:INDEX", operation::left_shift_assign, std::to_string(width - 1));
  state.invoke_assert_operation("A:INDEX", comparison::equal_to, minimum);
  state.invoke_assign_operation("A:INDEX", operation::right_shift_assign, std::to_string(width - 1));
  state.invoke_assert_operation("A:INDEX", comparison::equal_to, "-1");
  state.invoke_assign_operation("U:INDEX", operation::assign, maximum);
  state.invoke_assign_operation("U:INDEX", operation::left_shift_assign, "1");
  state.invoke_assert_operation("U:INDEX", comparison::equal_to,
                                std::to_string(std::numeric_limits<bra::uint_type>::max() - 1u));
  state.invoke_assign_operation("U:INDEX", operation::right_shift_assign, std::to_string(width - 1));
  state.invoke_assert_operation("U:INDEX", comparison::equal_to, "1");
  state.invoke_assert_operation("A:0", comparison::equal_to, "3");
  state.invoke_assert_operation("U:0", comparison::equal_to, maximum);
  if (not state.maybe_label() or *state.maybe_label() != "PENDING")
    throw std::runtime_error{"successful shift changed pending jump state"};
}
