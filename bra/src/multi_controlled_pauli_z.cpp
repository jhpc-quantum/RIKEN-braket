#include <cstddef>
#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/multi_controlled_pauli_z.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    multi_controlled_pauli_z::multi_controlled_pauli_z(std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{}, control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size() - std::size_t{1u}, 'C').append("Z")}
    { }

    multi_controlled_pauli_z::multi_controlled_pauli_z(std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size() - std::size_t{1u}, 'C').append("Z")}
    { }

    ::bra::state& multi_controlled_pauli_z::do_apply(::bra::state& state) const
    { return state.multi_controlled_pauli_z(control_qubits_); }

    std::string const& multi_controlled_pauli_z::do_name() const { return name_; }
    std::string multi_controlled_pauli_z::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
