#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/multi_controlled_t_gate.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    multi_controlled_t_gate::multi_controlled_t_gate(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase_coefficient_{phase_coefficient},
        target_qubit_{target_qubit},
        control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size(), 'C').append("T")}
    { }

    ::bra::state& multi_controlled_t_gate::do_apply(::bra::state& state) const
    { return state.multi_controlled_phase_shift(phase_coefficient_, target_qubit_, control_qubits_); }

    std::string const& multi_controlled_t_gate::do_name() const { return name_; }
    std::string multi_controlled_t_gate::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      repr_stream << std::right << std::setw(parameter_width) << target_qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
