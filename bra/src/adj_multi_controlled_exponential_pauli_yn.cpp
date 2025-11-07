#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <boost/variant/variant.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_multi_controlled_exponential_pauli_yn.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    adj_multi_controlled_exponential_pauli_yn::adj_multi_controlled_exponential_pauli_yn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, target_qubits_{target_qubits}, control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size(), 'C').append("e").append(target_qubits_.size(), 'Y').append("+")}
    { }

    adj_multi_controlled_exponential_pauli_yn::adj_multi_controlled_exponential_pauli_yn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type>&& target_qubits, std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, target_qubits_{std::move(target_qubits)}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size(), 'C').append("e").append(target_qubits_.size(), 'Y').append("+")}
    { }

    ::bra::state& adj_multi_controlled_exponential_pauli_yn::do_apply(::bra::state& state) const
    { return state.adj_multi_controlled_exponential_pauli_yn(phase_, target_qubits_, control_qubits_); }

    std::string const& adj_multi_controlled_exponential_pauli_yn::do_name() const { return name_; }
    std::string adj_multi_controlled_exponential_pauli_yn::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      for (auto&& target_qubit: target_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << target_qubit;
      repr_stream << std::right << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
