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
  state.generate_new_variable("Z", bra::variable_type::complex_, 1);
  state.generate_new_variable("P", bra::variable_type::pauli_string_space, 1);
  state.generate_new_variable("POINTERS", bra::variable_type::integer, 2);
  state.generate_new_variable("BAD", bra::variable_type::real, 1);
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

  // Explicit conversions are checked, while bare cross-type operands are not
  // silently converted. Both integer storages and jump state survive failures.
  state.invoke_assign_operation("R", operation::assign, "3.9");
  state.invoke_assign_operation("Z", operation::assign, ":COMPLEX:3.9");
  state.invoke_assign_operation("POINTERS:1", operation::assign, "0");
  for (auto const& operand: {"A:0", "A:POINTERS:I", ":INT:3.9", ":INT:R", ":INT:Z", ":INT:U:0"})
  {
    state.invoke_assign_operation("A:1", operation::assign, "7");
    state.invoke_assign_operation("A:I", operation::modulus_assign, operand);
    state.invoke_assert_operation("A:1", comparison::equal_to, "1");
  }
  for (auto const& operand: {"U:0", "U:POINTERS:I", ":UINT:3.9", ":UINT:R", ":UINT:Z", ":UINT:A:0"})
  {
    state.invoke_assign_operation("U:1", operation::assign, "7");
    state.invoke_assign_operation("U:I", operation::modulus_assign, operand);
    state.invoke_assert_operation("U:1", comparison::equal_to, "1");
  }
  auto expect_unchanged = [&](std::string const& target, std::string const& operand)
  {
    state.invoke_assign_operation("A:1", operation::assign, "7");
    state.invoke_assign_operation("U:1", operation::assign, "7");
    auto failed = false;
    try { state.invoke_assign_operation(target, operation::modulus_assign, operand); }
    catch (...) { failed = true; }
    require(failed, "invalid remainder instruction accepted: " + target + " %= " + operand);
    state.invoke_assert_operation("A:0", comparison::equal_to, "3");
    state.invoke_assert_operation("A:1", comparison::equal_to, "7");
    state.invoke_assert_operation("U:0", comparison::equal_to, "3");
    state.invoke_assert_operation("U:1", comparison::equal_to, "7");
    state.invoke_assert_operation("I", comparison::equal_to, "1");
    state.invoke_assert_operation("R", comparison::equal_to, "3.9");
    require(state.maybe_label() and *state.maybe_label() == "PENDING", "failed remainder changed jump state");
  };
  for (auto const& operand: {"", "R", "Z", "P", "U", ":UINT:3", ":REAL:3", ":COMPLEX:3", "3.5",
                            "MISSING", "A:-1", "A:2", "A:POINTERS:99", ":INT:R:99",
                            ":INT:1e309", ":UNKNOWN:3"})
    expect_unchanged("A:1", operand);
  for (auto const& operand: {"", "A", "R", "Z", "P", "-1", "-0", ":INT:3", ":REAL:3", "3.5",
                            "MISSING", "U:-1", "U:2", "U:POINTERS:99", ":UINT:R:99", ":UINT:1e309"})
    expect_unchanged("U:1", operand);
  expect_unchanged("A:1", std::to_string(std::numeric_limits<bra::int_type>::max()) + "0");
  expect_unchanged("A:1", ":INT:" + std::to_string(std::numeric_limits<bra::int_type>::max()) + "0.0");
  expect_unchanged("U:1", unsigned_maximum + "0");
  expect_unchanged("A:1", ":INT::UINT:" + unsigned_maximum);
  for (auto const& target: {"R", "Z", "P", "MISSING", ":INT:A", "A:-1", "A:2", "A:POINTERS:99", "U:-1", "U:2"})
    expect_unchanged(target, "3");
  for (auto const& numerator: {"0.0", "1.0", "-1.0"})
  {
    state.invoke_assign_operation("BAD", operation::assign, numerator);
    state.invoke_assign_operation("BAD", operation::divides_assign, "0.0");
    expect_unchanged("A:1", ":INT:BAD");
    expect_unchanged("U:1", ":UINT:BAD");
  }

  // Resolve aliased RHS/index operands before any write to their destination.
  state.invoke_assign_operation("A:1", operation::assign, "1");
  state.invoke_assign_operation("A:I", operation::modulus_assign, "A:A:I");
  state.invoke_assert_operation("A:1", comparison::equal_to, "0");
  state.invoke_assign_operation("A:1", operation::assign, "1");
  state.invoke_assign_operation("I", operation::modulus_assign, "A:I");
  state.invoke_assert_operation("I", comparison::equal_to, "0");
  for (auto const& target: {"A:1", "U:1"})
  {
    state.invoke_assign_operation(target, operation::assign, "7");
    state.invoke_assign_operation(target, operation::modulus_assign, target);
    state.invoke_assert_operation(target, comparison::equal_to, "0");
    auto failed = false;
    try { state.invoke_assign_operation(target, operation::modulus_assign, target); }
    catch (bra::integer_zero_divisor_error const&) { failed = true; }
    require(failed, "zero self-remainder accepted");
    state.invoke_assert_operation(target, comparison::equal_to, "0");
  }
  require(std::string{bra::integer_zero_divisor_error{"A", "0"}.what()}
            == "integer division by zero in LET A /= 0", "legacy division diagnostic changed");
  require(state.maybe_label() and *state.maybe_label() == "PENDING", "remainder changed pending jump state");

  auto input = std::istringstream{"QUBITS 0\nlet a:i %= u:i\n"};
  auto const interpreter = bra::interpreter{input};
  auto rendered = std::istringstream{interpreter.back_circuit().back()->representation()};
  auto mnemonic = std::string{}, lhs = std::string{}, op = std::string{}, rhs = std::string{};
  rendered >> mnemonic >> lhs >> op >> rhs;
  require(mnemonic == "LET" and lhs == "A:I" and op == "%=" and rhs == "U:I", "wrong remainder LET representation");
  require(static_cast<int>(operation::right_shift_assign) == 9, "existing assignment operation identifier changed");
}
