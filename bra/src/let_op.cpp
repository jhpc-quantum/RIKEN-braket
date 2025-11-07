#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/let_op.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const let_op::name_ = "LET";

    let_op::let_op(std::string const& lhs_variable_name, ::bra::assign_operation_type const op, std::string const& rhs_literal_or_variable_name)
      : ::bra::gate::gate{}, lhs_variable_name_{lhs_variable_name}, op_{op}, rhs_literal_or_variable_name_{rhs_literal_or_variable_name}
    { }

    ::bra::state& let_op::do_apply(::bra::state& state) const
    {
      state.invoke_assign_operation(lhs_variable_name_, op_, rhs_literal_or_variable_name_);
      return state;
    }

    std::string const& let_op::do_name() const { return name_; }
    std::string let_op::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << lhs_variable_name_;
      if (op_ == ::bra::assign_operation_type::assign)
        repr_stream
          << std::right
          << std::setw(parameter_width) << ":=";
      else if (op_ == ::bra::assign_operation_type::plus_assign)
        repr_stream
          << std::right
          << std::setw(parameter_width) << "+=";
      else if (op_ == ::bra::assign_operation_type::minus_assign)
        repr_stream
          << std::right
          << std::setw(parameter_width) << "-=";
      else if (op_ == ::bra::assign_operation_type::multiplies_assign)
        repr_stream
          << std::right
          << std::setw(parameter_width) << "*=";
      else if (op_ == ::bra::assign_operation_type::divides_assign)
        repr_stream
          << std::right
          << std::setw(parameter_width) << "/=";
      repr_stream
        << std::right
        << std::setw(parameter_width) << rhs_literal_or_variable_name_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
