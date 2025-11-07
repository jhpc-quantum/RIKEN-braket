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
#include <bra/gate/multi_controlled_exponential_swap.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    multi_controlled_exponential_swap::multi_controlled_exponential_swap(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, target_qubit1_{target_qubit1}, target_qubit2_{target_qubit2}, control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size(), 'C').append("eSWAP")}
    { }

    multi_controlled_exponential_swap::multi_controlled_exponential_swap(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, target_qubit1_{target_qubit1}, target_qubit2_{target_qubit2}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size(), 'C').append("eSWAP")}
    { }

    ::bra::state& multi_controlled_exponential_swap::do_apply(::bra::state& state) const
    { return state.multi_controlled_exponential_swap(phase_, target_qubit1_, target_qubit2_, control_qubits_); }

    std::string const& multi_controlled_exponential_swap::do_name() const { return name_; }
    std::string multi_controlled_exponential_swap::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      repr_stream
        << std::right
        << std::setw(parameter_width) << target_qubit1_
        << std::setw(parameter_width) << target_qubit2_
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
