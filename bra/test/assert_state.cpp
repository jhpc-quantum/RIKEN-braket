// Compile with the same macros as bra, then link with non-MPI bra objects,
// excluding bra.o (which defines main).
#include <sstream>
#include <limits>
#include <stdexcept>
#include <string>

#include <bra/gate/assert_op.hpp>
#include <bra/nompi_state.hpp>
#include <bra/state.hpp>

namespace
{
  void check_failure(bra::state& state, std::string const& lhs, bra::compare_operation_type const op,
                     std::string const& rhs, std::string const& instruction, std::string const& values)
  {
    auto const previous_label = state.maybe_label();
    auto caught = false;
    try { state.invoke_assert_operation(lhs, op, rhs); }
    catch (bra::assertion_error const& error)
    {
      caught = true;
      if (std::string{error.what()} != "assertion failed in ASSERT " + instruction + " (evaluated: " + values + ")")
        throw std::runtime_error{"unexpected assertion diagnostic: " + std::string{error.what()}};
    }
    if (not caught or state.maybe_label() != previous_label)
      throw std::runtime_error{"failed assertion changed jump state or did not throw"};
  }
}

int main()
{
  auto state = bra::nompi_state{0u, 0u, 1u, 16u, 1u, false, 0.0, 0.0, 0.0, false, 0u, 0};
  state.generate_new_variable("VALUES", bra::variable_type::integer, 2);
  state.generate_new_variable("INDEX", bra::variable_type::integer, 1);
  state.generate_new_variable("REALS", bra::variable_type::real, 2);
  state.generate_new_variable("COMPLEX", bra::variable_type::complex_, 1);
  state.invoke_assign_operation("INDEX", bra::assign_operation_type::assign, "1");
  state.invoke_assign_operation("VALUES:0", bra::assign_operation_type::assign, "3");
  state.invoke_assign_operation("VALUES:1", bra::assign_operation_type::assign, "7");
  state.invoke_assign_operation("REALS:0", bra::assign_operation_type::assign, "0.5");
  state.invoke_assign_operation("REALS:1", bra::assign_operation_type::assign, "-0.5");
  state.invoke_jump_operation("PENDING");

  using operation = bra::compare_operation_type;
  state.invoke_assert_operation("VALUES:INDEX", operation::equal_to, "7");
  state.invoke_assert_operation("VALUES:INDEX", operation::not_equal_to, "VALUES:0");
  state.invoke_assert_operation("VALUES:INDEX", operation::greater, "3");
  state.invoke_assert_operation("VALUES:0", operation::less, "VALUES:INDEX");
  state.invoke_assert_operation("VALUES:INDEX", operation::greater_equal, "7");
  state.invoke_assert_operation("VALUES:INDEX", operation::less_equal, "7");
  state.invoke_assert_operation("REALS:0", operation::equal_to, ":REAL:0.5");
  state.invoke_assert_operation("REALS:0", operation::not_equal_to, "REALS:INDEX");
  state.invoke_assert_operation("REALS:0", operation::greater, "REALS:1");
  state.invoke_assert_operation("REALS:1", operation::less, ":PI");
  state.invoke_assert_operation("REALS:0", operation::greater_equal, "0.5");
  state.invoke_assert_operation("REALS:INDEX", operation::less_equal, "-0.5");
  if (not state.maybe_label() or *state.maybe_label() != "PENDING")
    throw std::runtime_error{"successful assertion changed jump state"};

  check_failure(state, "VALUES:INDEX", operation::equal_to, "3", "VALUES:INDEX == 3", "7 == 3");
  check_failure(state, "VALUES:INDEX", operation::not_equal_to, "7", "VALUES:INDEX \\= 7", "7 \\= 7");
  check_failure(state, "VALUES:INDEX", operation::greater, "7", "VALUES:INDEX > 7", "7 > 7");
  check_failure(state, "VALUES:INDEX", operation::less, "7", "VALUES:INDEX < 7", "7 < 7");
  check_failure(state, "VALUES:INDEX", operation::greater_equal, "8", "VALUES:INDEX >= 8", "7 >= 8");
  check_failure(state, "VALUES:INDEX", operation::less_equal, "6", "VALUES:INDEX <= 6", "7 <= 6");
  check_failure(state, "REALS:0", operation::equal_to, "1", "REALS:0 == 1", "0.5 == 1");
  check_failure(state, "REALS:0", operation::not_equal_to, "0.5", "REALS:0 \\= 0.5", "0.5 \\= 0.5");
  check_failure(state, "REALS:0", operation::greater, "0.5", "REALS:0 > 0.5", "0.5 > 0.5");
  check_failure(state, "REALS:0", operation::less, "0.5", "REALS:0 < 0.5", "0.5 < 0.5");
  check_failure(state, "REALS:0", operation::greater_equal, "1", "REALS:0 >= 1", "0.5 >= 1");
  check_failure(state, "REALS:0", operation::less_equal, "0", "REALS:0 <= 0", "0.5 <= 0");
  state.invoke_assert_operation("VALUES:INDEX", operation::equal_to, "7");
  state.invoke_assert_operation("REALS:0", operation::equal_to, "0.5");

  for (auto const& lhs: {"", "0", ":INT:1", "MISSING", "COMPLEX"})
  {
    auto caught = false;
    try { state.invoke_assert_operation(lhs, operation::equal_to, "0"); }
    catch (bra::wrong_comparison_argument_error const&) { caught = true; }
    if (not caught)
      throw std::runtime_error{"invalid assertion operand accepted"};
  }
  state.delete_label();
  state.invoke_assert_operation("INDEX", operation::equal_to, "1");
  check_failure(state, "INDEX", operation::equal_to, "0", "INDEX == 0", "1 == 0");

  // Both signed endpoints are valid nonzero steps; comparisons need no negation.
  auto const minimum = std::to_string(std::numeric_limits<bra::int_type>::min());
  auto const maximum = std::to_string(std::numeric_limits<bra::int_type>::max());
  state.invoke_assign_operation("VALUES:0", bra::assign_operation_type::assign, minimum);
  state.invoke_assign_operation("VALUES:1", bra::assign_operation_type::assign, maximum);
  state.invoke_assert_operation("VALUES:0", operation::not_equal_to, "0");
  state.invoke_assert_operation("VALUES:1", operation::not_equal_to, "0");
  state.invoke_assert_operation("VALUES:0", operation::less, "VALUES:1");
  state.invoke_assert_operation("VALUES:1", operation::greater, "VALUES:0");

  ::bra::gate::assert_op gate{"INDEX", operation::not_equal_to, "0"};
  gate.apply(state);
  auto stream = std::istringstream{gate.representation()};
  auto mnemonic = std::string{}, lhs = std::string{}, op = std::string{}, rhs = std::string{}, extra = std::string{};
  stream >> mnemonic >> lhs >> op >> rhs;
  if (gate.name() != "ASSERT" or mnemonic != "ASSERT" or lhs != "INDEX" or op != "\\=" or rhs != "0" or stream >> extra)
    throw std::runtime_error{"assertion representation is not valid QCX"};
}
