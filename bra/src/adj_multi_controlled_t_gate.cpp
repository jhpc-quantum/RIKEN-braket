#include <cstddef>
#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_multi_controlled_t_gate.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    adj_multi_controlled_t_gate::adj_multi_controlled_t_gate(std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{},
        control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size() - std::size_t{1u}, 'C').append("T+")}
    { }

    adj_multi_controlled_t_gate::adj_multi_controlled_t_gate(std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size() - std::size_t{1u}, 'C').append("T+")}
    { }

    ::bra::state& adj_multi_controlled_t_gate::do_apply(::bra::state& state) const
    { return state.adj_multi_controlled_phase_shift(3, control_qubits_); }

    std::string const& adj_multi_controlled_t_gate::do_name() const { return name_; }
    std::string adj_multi_controlled_t_gate::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
