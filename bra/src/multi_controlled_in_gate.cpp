#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/multi_controlled_in_gate.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    multi_controlled_in_gate::multi_controlled_in_gate(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{}, target_qubits_{target_qubits}, control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size(), 'C').append(target_qubits_.size(), 'I')}
    { }

    multi_controlled_in_gate::multi_controlled_in_gate(
      std::vector<qubit_type>&& target_qubits,
      std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{}, target_qubits_{std::move(target_qubits)}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size(), 'C').append(target_qubits_.size(), 'I')}
    { }

    ::bra::state& multi_controlled_in_gate::do_apply(::bra::state& state) const
    { return state.multi_controlled_in_gate(target_qubits_, control_qubits_); }

    std::string const& multi_controlled_in_gate::do_name() const { return name_; }
    std::string multi_controlled_in_gate::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      for (auto&& target_qubit: target_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << target_qubit;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
