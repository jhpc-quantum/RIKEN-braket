#include <vector>
#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_pauli_yn.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    adj_pauli_yn::adj_pauli_yn(std::vector<qubit_type>&& qubits)
      : ::bra::gate::gate{}, qubits_{std::move(qubits)}, name_{std::string(qubits_.size(), 'Y').append("+")}
    { }

    ::bra::state& adj_pauli_yn::do_apply(::bra::state& state) const
    { return state.adj_pauli_yn(qubits_); }

    std::string const& adj_pauli_yn::do_name() const { return name_; }
    std::string adj_pauli_yn::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& qubit: qubits_)
        repr_stream << std::right << std::setw(parameter_width) << qubit;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
