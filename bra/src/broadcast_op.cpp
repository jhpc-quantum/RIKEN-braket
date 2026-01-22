#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/broadcast_op.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const broadcast_op::name_ = "BROADCAST";

    broadcast_op::broadcast_op(int const root_circuit_index, std::string const& variable_name, ::bra::variable_type const type, int const num_elements)
      : ::bra::gate::gate{}, root_circuit_index_{root_circuit_index}, variable_name_{variable_name}, type_{type}, num_elements_{num_elements}
    { }

    ::bra::state& broadcast_op::do_apply(::bra::state& state) const
    {
      state.broadcast_variable(root_circuit_index_, variable_name_, type_, num_elements_);
      return state;
    }

    std::string const& broadcast_op::do_name() const { return name_; }
    std::string broadcast_op::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << variable_name_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
