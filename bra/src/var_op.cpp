#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/var_op.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const var_op::name_ = "VAR";

    var_op::var_op(std::string const& variable_name, ::bra::variable_type const type, int const num_elements)
      : ::bra::gate::gate{}, variable_name_{variable_name}, type_{type}, num_elements_{num_elements}
    { }

    ::bra::state& var_op::do_apply(::bra::state& state) const
    {
      if (type_ == ::bra::variable_type::real)
        state.generate_new_real_variable(variable_name_, num_elements_);
      else if (type_ == ::bra::variable_type::integer)
        state.generate_new_int_variable(variable_name_, num_elements_);

      return state;
    }

    std::string const& var_op::do_name() const { return name_; }
    std::string var_op::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << variable_name_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
