#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_pauli_xx.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_pauli_xx::name_ = "XX+";

    adj_pauli_xx::adj_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
      : ::bra::gate::gate{}, qubit1_{qubit1}, qubit2_{qubit2}
    { }

    ::bra::state& adj_pauli_xx::do_apply(::bra::state& state) const
    { return state.adj_pauli_xx(qubit1_, qubit2_); }

    std::string const& adj_pauli_xx::do_name() const { return name_; }
    std::string adj_pauli_xx::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit1_
        << std::setw(parameter_width) << qubit2_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
