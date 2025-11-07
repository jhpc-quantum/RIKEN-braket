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
#include <bra/gate/multi_controlled_u3.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    multi_controlled_u3::multi_controlled_u3(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      boost::variant<real_type, std::string> const& phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{},
        phase1_{phase1}, phase2_{phase2}, phase3_{phase3},
        target_qubit_{target_qubit}, control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size(), 'C').append("U3")}
    { }

    multi_controlled_u3::multi_controlled_u3(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      boost::variant<real_type, std::string> const& phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase1_{phase1}, phase2_{phase2}, phase3_{phase3},
        target_qubit_{target_qubit}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size(), 'C').append("U3")}
    { }

    ::bra::state& multi_controlled_u3::do_apply(::bra::state& state) const
    { return state.multi_controlled_u3(phase1_, phase2_, phase3_, target_qubit_, control_qubits_); }

    std::string const& multi_controlled_u3::do_name() const { return name_; }
    std::string multi_controlled_u3::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      repr_stream
        << std::right
        << std::setw(parameter_width) << target_qubit_
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase1_)
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase2_)
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase3_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
