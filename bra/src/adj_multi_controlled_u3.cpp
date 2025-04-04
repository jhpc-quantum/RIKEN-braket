#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_multi_controlled_u3.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    adj_multi_controlled_u3::adj_multi_controlled_u3(
      real_type const& phase1, real_type const& phase2, real_type const& phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{},
        phase1_{phase1}, phase2_{phase2}, phase3_{phase3},
        target_qubit_{target_qubit}, control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size(), 'C').append("U3+")}
    { }

    adj_multi_controlled_u3::adj_multi_controlled_u3(
      real_type const& phase1, real_type const& phase2, real_type const& phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase1_{phase1}, phase2_{phase2}, phase3_{phase3},
        target_qubit_{target_qubit}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size(), 'C').append("U3+")}
    { }

    ::bra::state& adj_multi_controlled_u3::do_apply(::bra::state& state) const
    { return state.adj_multi_controlled_u3(phase1_, phase2_, phase3_, target_qubit_, control_qubits_); }

    std::string const& adj_multi_controlled_u3::do_name() const { return name_; }
    std::string adj_multi_controlled_u3::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      repr_stream
        << std::right
        << std::setw(parameter_width) << target_qubit_
        << std::setw(parameter_width) << phase1_
        << std::setw(parameter_width) << phase2_
        << std::setw(parameter_width) << phase3_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
