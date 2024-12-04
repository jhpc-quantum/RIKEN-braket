#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_multi_controlled_exponential_swap.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    adj_multi_controlled_exponential_swap::adj_multi_controlled_exponential_swap(
      real_type const& phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, target_qubit1_{target_qubit1}, target_qubit2_{target_qubit2}, control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size(), 'C').append("eSWAP")}
    { }

    adj_multi_controlled_exponential_swap::adj_multi_controlled_exponential_swap(
      real_type const& phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, target_qubit1_{target_qubit1}, target_qubit2_{target_qubit2}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size(), 'C').append("eSWAP")}
    { }

    ::bra::state& adj_multi_controlled_exponential_swap::do_apply(::bra::state& state) const
    { return state.adj_multi_controlled_exponential_swap(phase_, target_qubit1_, target_qubit2_, control_qubits_); }

    std::string const& adj_multi_controlled_exponential_swap::do_name() const { return name_; }
    std::string adj_multi_controlled_exponential_swap::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      repr_stream
        << std::right
        << std::setw(parameter_width) << target_qubit1_
        << std::setw(parameter_width) << target_qubit2_
        << std::setw(parameter_width) << phase_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
