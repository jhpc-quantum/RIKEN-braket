#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/multi_controlled_u1.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    multi_controlled_u1::multi_controlled_u1(
      real_type const& phase, qubit_type const target_qubit, std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, target_qubit_{target_qubit}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size(), 'C').append("U1")}
    { }

    ::bra::state& multi_controlled_u1::do_apply(::bra::state& state) const
    { return state.multi_controlled_u1(phase_, target_qubit_, control_qubits_); }

    std::string const& multi_controlled_u1::do_name() const { return name_; }
    std::string multi_controlled_u1::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      repr_stream
        << std::right
        << std::setw(parameter_width) << target_qubit_
        << std::setw(parameter_width) << phase_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
