#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/gather_op.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const gather_op::name_ = "BROADCAST";

    gather_op::gather_op(int const root_circuit_index, std::string const& variable_name, ::bra::variable_type const type, int const num_elements, std::string const& destination_variable_name)
      : ::bra::gate::gate{}, root_circuit_index_{root_circuit_index}, variable_name_{variable_name}, type_{type}, num_elements_{num_elements}, destination_variable_name_{destination_variable_name}
    { }

    ::bra::state& gather_op::do_apply(::bra::state& state) const
    {
      state.gather_variable(root_circuit_index_, variable_name_, type_, num_elements_, destination_variable_name_);
      return state;
    }

    std::string const& gather_op::do_name() const { return name_; }
    std::string gather_op::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << variable_name_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
