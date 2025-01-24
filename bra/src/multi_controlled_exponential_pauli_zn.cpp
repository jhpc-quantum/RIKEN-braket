#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/multi_controlled_exponential_pauli_zn.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    multi_controlled_exponential_pauli_zn::multi_controlled_exponential_pauli_zn(
      real_type const& phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, target_qubits_{target_qubits}, control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size(), 'C').append("e").append(target_qubits_.size(), 'Z')}
    { }

    multi_controlled_exponential_pauli_zn::multi_controlled_exponential_pauli_zn(
      real_type const& phase, std::vector<qubit_type>&& target_qubits, std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, target_qubits_{std::move(target_qubits)}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size(), 'C').append("e").append(target_qubits_.size(), 'Z')}
    { }

    ::bra::state& multi_controlled_exponential_pauli_zn::do_apply(::bra::state& state) const
    { return state.multi_controlled_exponential_pauli_zn(phase_, target_qubits_, control_qubits_); }

    std::string const& multi_controlled_exponential_pauli_zn::do_name() const { return name_; }
    std::string multi_controlled_exponential_pauli_zn::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      for (auto&& target_qubit: target_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << target_qubit;
      repr_stream << std::right << std::setw(parameter_width) << phase_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
