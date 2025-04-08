#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/ic_gate.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const ic_gate::name_ = "IC";

    ic_gate::ic_gate(control_qubit_type const control_qubit)
      : ::bra::gate::gate{}, control_qubit_{control_qubit}
    { }

    ::bra::state& ic_gate::do_apply(::bra::state& state) const
    { return state.ic_gate(control_qubit_); }

    std::string const& ic_gate::do_name() const { return name_; }
    std::string ic_gate::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
