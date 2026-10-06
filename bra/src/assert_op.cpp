#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <bra/gate/assert_op.hpp>
#include <bra/state.hpp>

namespace bra
{
  namespace gate
  {
    std::string const assert_op::name_ = "ASSERT";

    assert_op::assert_op(std::string const& lhs_variable_name, ::bra::compare_operation_type const op,
                         std::string const& rhs_literal_or_variable_name)
      : ::bra::gate::gate{}, lhs_variable_name_{lhs_variable_name}, op_{op},
        rhs_literal_or_variable_name_{rhs_literal_or_variable_name}
    { }

    ::bra::state& assert_op::do_apply(::bra::state& state) const
    {
      state.invoke_assert_operation(lhs_variable_name_, op_, rhs_literal_or_variable_name_);
      return state;
    }

    std::string const& assert_op::do_name() const { return name_; }

    std::string assert_op::do_representation(std::ostringstream& repr_stream, int const parameter_width) const
    {
      auto const op_string
        = op_ == ::bra::compare_operation_type::equal_to ? "=="
          : op_ == ::bra::compare_operation_type::not_equal_to ? "\\="
          : op_ == ::bra::compare_operation_type::greater ? ">"
          : op_ == ::bra::compare_operation_type::less ? "<"
          : op_ == ::bra::compare_operation_type::greater_equal ? ">=" : "<=";
      repr_stream << std::right
                  << std::setw(parameter_width) << lhs_variable_name_
                  << std::setw(parameter_width) << op_string
                  << std::setw(parameter_width) << rhs_literal_or_variable_name_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
