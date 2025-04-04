#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/s_gate.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const s_gate::name_ = "S";

    s_gate::s_gate(qubit_type const qubit)
      : ::bra::gate::gate{}, qubit_{qubit}
    { }

    ::bra::state& s_gate::do_apply(::bra::state& state) const
    { return state.sqrt_pauli_z(qubit_); }

    std::string const& s_gate::do_name() const { return name_; }
    std::string s_gate::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
