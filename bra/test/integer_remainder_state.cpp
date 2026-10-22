// Compile with bra's macros and link with non-MPI objects excluding bra.o.
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <bra/gate/let_op.hpp>
#include <bra/interpreter.hpp>
#include <bra/nompi_state.hpp>
#include <bra/state.hpp>

namespace
{
  void require(bool const condition, std::string const& message)
  { if (not condition) throw std::runtime_error{message}; }
}

int main()
{
  using operation = bra::assign_operation_type;
  using comparison = bra::compare_operation_type;
  auto state = bra::nompi_state{0u, 0u, 1u, 16u, 1u, false, 0.0, 0.0, 0.0, false, 0u, 0};
  state.generate_new_variable("A", bra::variable_type::integer, 2);
  state.generate_new_variable("U", bra::variable_type::unsigned_integer, 2);
  state.generate_new_variable("I", bra::variable_type::integer, 1);
  state.generate_new_variable("R", bra::variable_type::real, 1);
  state.invoke_assign_operation("I", operation::assign, "1");
  state.invoke_assign_operation("A:0", operation::assign, "3");
  state.invoke_assign_operation("U:0", operation::assign, "3");
  state.invoke_jump_operation("PENDING");

  for (auto const dividend: {-7, -1, 0, 1, 7, std::numeric_limits<bra::int_type>::min(),
                            std::numeric_limits<bra::int_type>::max()})
    for (auto const divisor: {-3, -1, 1, 3})
    {
      state.invoke_assign_operation("A:1", operation::assign, std::to_string(dividend));
      state.invoke_assign_operation("A:I", operation::modulus_assign, std::to_string(divisor));
      auto const expected = divisor == -1 ? 0 : dividend % divisor;
      state.invoke_assert_operation("A:1", comparison::equal_to, std::to_string(expected));
      state.invoke_assert_operation("A:0", comparison::equal_to, "3");
    }
  auto const unsigned_maximum = std::to_string(std::numeric_limits<bra::uint_type>::max());
  state.invoke_assign_operation("U:1", operation::assign, unsigned_maximum);
  state.invoke_assign_operation("U:I", operation::modulus_assign, "U:0");
  state.invoke_assert_operation("U:1", comparison::equal_to,
                               std::to_string(std::numeric_limits<bra::uint_type>::max() % 3u));
  state.invoke_assign_operation("U:1", operation::assign, unsigned_maximum);
  state.invoke_assign_operation("U:I", operation::modulus_assign, unsigned_maximum);
  state.invoke_assert_operation("U:1", comparison::equal_to, "0");

  for (auto const& target: {"A:1", "U:1"})
    for (auto const& divisor: {"0", ":INT:0.5", ":UINT:0.5"})
    {
      if ((std::string{target} == "A:1" and std::string{divisor} == ":UINT:0.5")
          or (std::string{target} == "U:1" and std::string{divisor} == ":INT:0.5")) continue;
      state.invoke_assign_operation(target, operation::assign, "7");
      auto caught = false;
      try { state.invoke_assign_operation(target, operation::modulus_assign, divisor); }
      catch (bra::integer_zero_divisor_error const& error)
      {
        caught = true;
        require(std::string{error.what()} == std::string{"integer remainder by zero in LET "} + target + " %= " + divisor,
                "wrong remainder zero-divisor diagnostic");
      }
      require(caught, "zero remainder divisor accepted");
      state.invoke_assert_operation(target, comparison::equal_to, "7");
    }
  for (auto const& divisor: {"R", "A:99", ":INT:R:99", ":INT:1e309"})
  {
    auto caught = false;
    try { state.invoke_assign_operation("A:1", operation::modulus_assign, divisor); }
    catch (...) { caught = true; }
    require(caught, "invalid remainder operand accepted");
    state.invoke_assert_operation("A:1", comparison::equal_to, "7");
  }
  auto caught = false;
  try { state.invoke_assign_operation("R", operation::modulus_assign, "3"); }
  catch (bra::wrong_assignment_argument_error const& error)
  {
    caught = true;
    require(std::string{error.what()}.find("R %= 3") != std::string::npos, "wrong remainder type-error diagnostic");
  }
  require(caught, "REAL remainder destination accepted");
  require(state.maybe_label() and *state.maybe_label() == "PENDING", "remainder changed pending jump state");

  auto input = std::istringstream{"QUBITS 0\nlet a:i %= u:i\n"};
  auto const interpreter = bra::interpreter{input};
  auto rendered = std::istringstream{interpreter.back_circuit().back()->representation()};
  auto mnemonic = std::string{}, lhs = std::string{}, op = std::string{}, rhs = std::string{};
  rendered >> mnemonic >> lhs >> op >> rhs;
  require(mnemonic == "LET" and lhs == "A:I" and op == "%=" and rhs == "U:I", "wrong remainder LET representation");
  require(static_cast<int>(operation::right_shift_assign) == 9, "existing assignment operation identifier changed");
}
