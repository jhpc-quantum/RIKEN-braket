#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/controlled_pauli_x.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const controlled_pauli_x::name_ = "CX";

    controlled_pauli_x::controlled_pauli_x(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit)
      : ::bra::gate::gate{}, target_qubit_{target_qubit}, control_qubit_{control_qubit}
    { }

    ::bra::state& controlled_pauli_x::do_apply(::bra::state& state) const
    { return state.controlled_pauli_x(target_qubit_, control_qubit_); }

    std::string const& controlled_pauli_x::do_name() const { return name_; }
    std::string controlled_pauli_x::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_
        << std::setw(parameter_width) << target_qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
