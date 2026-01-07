#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/send_op.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const send_op::name_ = "SEND";

    send_op::send_op(int const destination_circuit_index, std::string const& variable_name, ::bra::variable_type const type, int const num_elements)
      : ::bra::gate::gate{}, destination_circuit_index_{destination_circuit_index}, variable_name_{variable_name}, type_{type}, num_elements_{num_elements}
    { }

    ::bra::state& send_op::do_apply(::bra::state& state) const
    {
      state.send_variable(destination_circuit_index_, variable_name_, type_, num_elements_);
      return state;
    }

    std::string const& send_op::do_name() const { return name_; }
    std::string send_op::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << variable_name_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
