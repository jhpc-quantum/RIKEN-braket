#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_sqrt_pauli_z.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_sqrt_pauli_z::name_ = "sZ+";

    adj_sqrt_pauli_z::adj_sqrt_pauli_z(control_qubit_type const control_qubit)
      : ::bra::gate::gate{}, control_qubit_{control_qubit}
    { }

    ::bra::state& adj_sqrt_pauli_z::do_apply(::bra::state& state) const
    { return state.adj_sqrt_pauli_z(control_qubit_); }

    std::string const& adj_sqrt_pauli_z::do_name() const { return name_; }
    std::string adj_sqrt_pauli_z::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
