#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/pauli_z.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const pauli_z::name_ = "Z";

    pauli_z::pauli_z(control_qubit_type const control_qubit)
      : ::bra::gate::gate{}, control_qubit_{control_qubit}
    { }

    ::bra::state& pauli_z::do_apply(::bra::state& state) const
    { return state.pauli_z(control_qubit_); }

    std::string const& pauli_z::do_name() const { return name_; }
    std::string pauli_z::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
