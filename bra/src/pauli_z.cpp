#include <boost/config.hpp>

#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/pauli_z.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const pauli_z::name_ = "Z";

    pauli_z::pauli_z(qubit_type const qubit)
      : ::bra::gate::gate(), qubit_(qubit)
    { }

    ::bra::state& pauli_z::do_apply(::bra::state& state) const
    { return state.pauli_z(qubit_); }

    std::string const& pauli_z::do_name() const { return name_; }
    std::string pauli_z::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_;
      return repr_stream.str();
    }
  }
}

