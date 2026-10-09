// Compile with bra's macros and link with non-MPI objects excluding bra.o.
#include <limits>
#include <stdexcept>
#include <string>

#include <bra/nompi_state.hpp>
#include <bra/state.hpp>

namespace
{
  template <typename Function>
  void expect_failure(Function const& function)
  {
    auto caught = false;
    try { function(); }
    catch (std::exception const&) { caught = true; }
    if (not caught)
      throw std::runtime_error{"invalid UINT communication accepted"};
  }
}

int main()
{
  auto source = bra::nompi_state{0u, 0u, 1u, 16u, 1u, false, 0.0, 0.0, 0.0, false, 0u, 0};
  auto destination = bra::nompi_state{0u, 0u, 1u, 16u, 1u, false, 0.0, 0.0, 0.0, false, 0u, 1};
  using type = bra::variable_type;
  using operation = bra::assign_operation_type;
  using comparison = bra::compare_operation_type;
  auto const maximum = std::to_string(std::numeric_limits<bra::uint_type>::max());
  source.generate_new_variable("A", type::unsigned_integer, 2);
  destination.generate_new_variable("B", type::unsigned_integer, 1);
  source.invoke_assign_operation("A:0", operation::assign, maximum);
  destination.invoke_assign_operation("B", operation::assign, "7");
  for (auto const count: {-1, 0, 2, std::numeric_limits<int>::max()})
  {
    expect_failure([&] { bra::send_uint_variable(source, "A", destination, "B", count); });
    destination.invoke_assert_operation("B", comparison::equal_to, "7");
  }
  expect_failure([&] { bra::send_uint_variable(source, "A:-1", destination, "B", 1); });
  expect_failure([&] { bra::send_uint_variable(source, "A:2", destination, "B", 1); });
  destination.invoke_assert_operation("B", comparison::equal_to, "7");
  bra::send_uint_variable(source, "A", destination, "B", 1);
  destination.invoke_assert_operation("B", comparison::equal_to, maximum);
  source.send_variable(1, "A", type::unsigned_integer, 1);
  if (not source.is_waiting() or not source.wait_reason().is_send_uint_variable()
      or source.wait_reason().is_send_int_variable() or source.wait_reason().other_circuit_index() != 1)
    throw std::runtime_error{"incorrect UINT send wait state"};
  source.cancel_waiting();
  source.receive_variable(1, "A", type::unsigned_integer, 1);
  if (not source.wait_reason().is_receive_uint_variable() or source.wait_reason().is_receive_int_variable())
    throw std::runtime_error{"incorrect UINT receive wait state"};
  source.cancel_waiting();
  source.broadcast_variable(1, "A", type::unsigned_integer, 2);
  if (not source.wait_reason().is_broadcast_uint_variable() or source.wait_reason().root_circuit_index() != 1
      or source.wait_reason().num_elements() != 2 or source.wait_reason().variable_name() != "A")
    throw std::runtime_error{"incorrect UINT broadcast wait state"};
  source.cancel_waiting();
  source.gather_variable(1, "A", type::unsigned_integer, 1, "DEST");
  if (not source.wait_reason().is_gather_uint_variable() or source.wait_reason().other_variable_name() != "DEST")
    throw std::runtime_error{"incorrect UINT gather wait state"};
  source.cancel_waiting();
  source.scatter_variable(1, "A", type::unsigned_integer, 1, "FROM");
  if (not source.wait_reason().is_scatter_uint_variable() or source.wait_reason().other_variable_name() != "FROM")
    throw std::runtime_error{"incorrect UINT scatter wait state"};
  source.cancel_waiting();
  source.send_variable(0, "A", type::unsigned_integer, 1);
  source.receive_variable(0, "A", type::unsigned_integer, 1);
  if (source.is_waiting())
    throw std::runtime_error{"UINT self-transfer should not wait"};
}
