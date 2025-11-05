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
#include <bra/gate/adj_multi_controlled_u2.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    adj_multi_controlled_u2::adj_multi_controlled_u2(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{},
        phase1_{phase1}, phase2_{phase2},
        target_qubit_{target_qubit}, control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size(), 'C').append("U2+")}
    { }

    adj_multi_controlled_u2::adj_multi_controlled_u2(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      qubit_type const target_qubit, std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase1_{phase1}, phase2_{phase2},
        target_qubit_{target_qubit}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size(), 'C').append("U2+")}
    { }

    ::bra::state& adj_multi_controlled_u2::do_apply(::bra::state& state) const
    { return state.adj_multi_controlled_u2(phase1_, phase2_, target_qubit_, control_qubits_); }

    std::string const& adj_multi_controlled_u2::do_name() const { return name_; }
    std::string adj_multi_controlled_u2::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      repr_stream
        << std::right
        << std::setw(parameter_width) << target_qubit_
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase1_)
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase2_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
