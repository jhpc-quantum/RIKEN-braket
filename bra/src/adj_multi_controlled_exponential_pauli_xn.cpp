#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_multi_controlled_exponential_pauli_xn.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    adj_multi_controlled_exponential_pauli_xn::adj_multi_controlled_exponential_pauli_xn(
      real_type const& phase, std::vector<qubit_type>&& target_qubits, std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, target_qubits_{std::move(target_qubits)}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size(), 'C').append("e").append(target_qubits_.size(), 'X').append("+")}
    { }

    ::bra::state& adj_multi_controlled_exponential_pauli_xn::do_apply(::bra::state& state) const
    { return state.adj_multi_controlled_exponential_pauli_xn(phase_, target_qubits_, control_qubits_); }

    std::string const& adj_multi_controlled_exponential_pauli_xn::do_name() const { return name_; }
    std::string adj_multi_controlled_exponential_pauli_xn::do_representation(
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
