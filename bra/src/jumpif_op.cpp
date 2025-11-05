#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/jumpif_op.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const jumpif_op::name_ = "JUMPIF";

    jumpif_op::jumpif_op(
      std::string const& label, std::string const& lhs_variable_name,
      ::bra::compare_operation_type const op,
      std::string const& rhs_literal_or_variable_name)
      : ::bra::gate::gate{}, label_{label}, lhs_variable_name_{lhs_variable_name}, op_{op}, rhs_literal_or_variable_name_{rhs_literal_or_variable_name}
    { }

    ::bra::state& jumpif_op::do_apply(::bra::state& state) const
    {
      state.invoke_jump_operation(label_, lhs_variable_name_, op_, rhs_literal_or_variable_name_);
      return state;
    }

    std::string const& jumpif_op::do_name() const { return name_; }
    std::string jumpif_op::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << label_
        << std::setw(parameter_width) << lhs_variable_name_;
      if (op_ == ::bra::compare_operation_type::equal_to)
        repr_stream
          << std::right
          << std::setw(parameter_width) << "==";
      else if (op_ == ::bra::compare_operation_type::not_equal_to)
        repr_stream
          << std::right
          << std::setw(parameter_width) << "!=";
      else if (op_ == ::bra::compare_operation_type::greater)
        repr_stream
          << std::right
          << std::setw(parameter_width) << ">";
      else if (op_ == ::bra::compare_operation_type::less)
        repr_stream
          << std::right
          << std::setw(parameter_width) << "<";
      else if (op_ == ::bra::compare_operation_type::greater_equal)
        repr_stream
          << std::right
          << std::setw(parameter_width) << ">=";
      else if (op_ == ::bra::compare_operation_type::less_equal)
        repr_stream
          << std::right
          << std::setw(parameter_width) << "<=";
      repr_stream
        << std::right
        << std::setw(parameter_width) << rhs_literal_or_variable_name_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
