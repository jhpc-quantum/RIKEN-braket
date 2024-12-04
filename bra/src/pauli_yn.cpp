#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/pauli_yn.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    pauli_yn::pauli_yn(std::vector<qubit_type> const& qubits)
      : ::bra::gate::gate{}, qubits_{qubits}, name_{std::string(qubits_.size(), 'Y')}
    { }

    pauli_yn::pauli_yn(std::vector<qubit_type>&& qubits)
      : ::bra::gate::gate{}, qubits_{std::move(qubits)}, name_{std::string(qubits_.size(), 'Y')}
    { }

    ::bra::state& pauli_yn::do_apply(::bra::state& state) const
    { return state.pauli_yn(qubits_); }

    std::string const& pauli_yn::do_name() const { return name_; }
    std::string pauli_yn::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& qubit: qubits_)
        repr_stream << std::right << std::setw(parameter_width) << qubit;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
