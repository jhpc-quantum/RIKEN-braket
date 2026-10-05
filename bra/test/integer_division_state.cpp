// Link with non-MPI bra objects, excluding bra.o (which defines main).
#include <stdexcept>
#include <string>

#include <bra/nompi_state.hpp>
#include <bra/state.hpp>

namespace
{
  void check_value(bra::state& state, std::string const& variable, std::string const& expected)
  {
    state.delete_label();
    state.invoke_jump_operation("MATCH", variable, bra::compare_operation_type::equal_to, expected);
    if (not state.maybe_label())
      throw std::runtime_error{"unexpected integer value in " + variable};
  }
}

int main()
{
  auto state = bra::nompi_state{0u, 0u, 1u, 16u, 1u, false, 0.0, 0.0, 0.0, false, 0u, 0};
  state.generate_new_variable("VALUES", bra::variable_type::integer, 2);
  state.generate_new_variable("INDEX", bra::variable_type::integer, 1);
  state.invoke_assign_operation("INDEX", bra::assign_operation_type::assign, "1");
  state.invoke_assign_operation("VALUES:0", bra::assign_operation_type::assign, "3");

  for (auto const& divisor: {"0", ":INT:0.5", "INDEX"})
  {
    state.invoke_assign_operation("VALUES:1", bra::assign_operation_type::assign, "7");
    if (std::string{divisor} == "INDEX")
      state.invoke_assign_operation("INDEX", bra::assign_operation_type::assign, "0");
    auto caught = false;
    try
    {
      state.invoke_assign_operation("VALUES:1", bra::assign_operation_type::divides_assign, divisor);
    }
    catch (bra::integer_zero_divisor_error const& error)
    {
      caught = true;
      if (std::string{error.what()} != std::string{"integer division by zero in LET VALUES:1 /= "} + divisor)
        throw std::runtime_error{"unexpected integer division diagnostic"};
    }
    if (not caught)
      throw std::runtime_error{"zero-divisor check failed"};
    check_value(state, "VALUES:1", "7");
    check_value(state, "VALUES:0", "3");
  }

  state.invoke_assign_operation("INDEX", bra::assign_operation_type::assign, "1");
  state.invoke_assign_operation("VALUES:INDEX", bra::assign_operation_type::divides_assign, "VALUES:0");
  check_value(state, "VALUES:1", "2");
}
