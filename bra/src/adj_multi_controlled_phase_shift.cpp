#include <cstddef>
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
#include <bra/gate/adj_multi_controlled_phase_shift.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    adj_multi_controlled_phase_shift::adj_multi_controlled_phase_shift(
      boost::variant<int_type, std::string> const& phase_exponent,
      std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{},
        phase_exponent_{phase_exponent},
        control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size() - std::size_t{1u}, 'C').append("R+")}
    { }

    adj_multi_controlled_phase_shift::adj_multi_controlled_phase_shift(
      boost::variant<int_type, std::string> const& phase_exponent,
      std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase_exponent_{phase_exponent},
        control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size() - std::size_t{1u}, 'C').append("R+")}
    { }

    ::bra::state& adj_multi_controlled_phase_shift::do_apply(::bra::state& state) const
    { return state.adj_multi_controlled_phase_shift(phase_exponent_, control_qubits_); }

    std::string const& adj_multi_controlled_phase_shift::do_name() const { return name_; }
    std::string adj_multi_controlled_phase_shift::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      repr_stream
        << std::right
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<int_type>{}, phase_exponent_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
