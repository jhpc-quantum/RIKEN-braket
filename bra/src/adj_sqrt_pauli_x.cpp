#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_sqrt_pauli_x.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_sqrt_pauli_x::name_ = "sX+";

    adj_sqrt_pauli_x::adj_sqrt_pauli_x(qubit_type const qubit)
      : ::bra::gate::gate{}, qubit_{qubit}
    { }

    ::bra::state& adj_sqrt_pauli_x::do_apply(::bra::state& state) const
    { return state.adj_sqrt_pauli_x(qubit_); }

    std::string const& adj_sqrt_pauli_x::do_name() const { return name_; }
    std::string adj_sqrt_pauli_x::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
