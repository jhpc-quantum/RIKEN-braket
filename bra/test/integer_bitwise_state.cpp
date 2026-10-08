// Compile with bra's macros and link with non-MPI objects excluding bra.o.
#include <sstream>
#include <stdexcept>
#include <string>

#include <bra/gate/let_op.hpp>
#include <bra/nompi_state.hpp>
#include <bra/state.hpp>

namespace
{
  void check_value(bra::state& state, std::string const& variable, std::string const& expected)
  { state.invoke_assert_operation(variable, bra::compare_operation_type::equal_to, expected); }
}

int main()
{
  auto state = bra::nompi_state{0u, 0u, 1u, 16u, 1u, false, 0.0, 0.0, 0.0, false, 0u, 0};
  state.generate_new_variable("VALUES", bra::variable_type::integer, 2);
  state.generate_new_variable("INDEX", bra::variable_type::integer, 1);
  state.generate_new_variable("R", bra::variable_type::real, 1);
  state.generate_new_variable("Z", bra::variable_type::complex_, 1);
  state.generate_new_variable("P", bra::variable_type::pauli_string_space, 1);
  state.invoke_assign_operation("INDEX", bra::assign_operation_type::assign, "1");
  state.invoke_assign_operation("VALUES:0", bra::assign_operation_type::assign, "3");
  state.invoke_assign_operation("R", bra::assign_operation_type::assign, "2.5");
  state.invoke_assign_operation("Z", bra::assign_operation_type::assign, ":COMPLEX:2.5");
  state.invoke_jump_operation("PENDING");

  using operation = bra::assign_operation_type;
  for (auto const op: {operation::bit_and_assign, operation::bit_or_assign, operation::bit_xor_assign})
  {
    auto const symbol = op == operation::bit_and_assign ? "&=" : op == operation::bit_or_assign ? "|=" : "^=";
    bra::gate::let_op const gate{"A", op, "3"};
    auto stream = std::istringstream{gate.representation()};
    auto mnemonic = std::string{}, lhs = std::string{}, rendered_op = std::string{}, rhs = std::string{};
    stream >> mnemonic >> lhs >> rendered_op >> rhs;
    if (mnemonic != "LET" or lhs != "A" or rendered_op != symbol or rhs != "3")
      throw std::runtime_error{"incorrect bitwise LET representation"};

    for (auto const& target: {"R", "Z", "P", "MISSING"})
    {
      auto caught = false;
      try { state.invoke_assign_operation(target, op, "1"); }
      catch (bra::wrong_assignment_argument_error const& error)
      {
        caught = true;
        if (std::string{error.what()} != std::string{"\""} + target + " " + symbol + " 1\" is a wrong argument")
          throw std::runtime_error{"incorrect bitwise LET diagnostic"};
      }
      if (not caught)
        throw std::runtime_error{"non-integer destination accepted"};
    }
    // Failed RHS conversions must not modify either indexed element.
    for (auto const& operand: {"R", "Z", "P", ":REAL", "1.5", "MISSING"})
    {
      state.invoke_assign_operation("VALUES:1", operation::assign, "7");
      auto caught = false;
      try { state.invoke_assign_operation("VALUES:INDEX", op, operand); }
      catch (...) { caught = true; }
      if (not caught)
        throw std::runtime_error{"non-integer RHS accepted"};
      check_value(state, "VALUES:0", "3");
      check_value(state, "VALUES:1", "7");
    }
    state.invoke_assign_operation("VALUES:1", operation::assign, "7");
    state.invoke_assign_operation("VALUES:INDEX", op, "VALUES:0");
    check_value(state, "VALUES:1", op == operation::bit_and_assign ? "3" : op == operation::bit_or_assign ? "7" : "4");
    check_value(state, "VALUES:0", "3");
    state.invoke_assert_operation("R", bra::compare_operation_type::equal_to, "2.5");
    if (not state.maybe_label() or *state.maybe_label() != "PENDING")
      throw std::runtime_error{"bitwise LET changed pending jump state"};
  }
}
