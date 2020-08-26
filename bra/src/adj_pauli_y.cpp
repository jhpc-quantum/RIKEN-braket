#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_pauli_y.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_pauli_y::name_ = "Y+";

    adj_pauli_y::adj_pauli_y(qubit_type const qubit)
      : ::bra::gate::gate{}, qubit_{qubit}
    { }

    ::bra::state& adj_pauli_y::do_apply(::bra::state& state) const
    { return state.adj_pauli_y(qubit_); }

    std::string const& adj_pauli_y::do_name() const { return name_; }
    std::string adj_pauli_y::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
