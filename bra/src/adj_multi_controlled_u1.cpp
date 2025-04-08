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
#include <bra/gate/adj_multi_controlled_u1.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    adj_multi_controlled_u1::adj_multi_controlled_u1(
      real_type const& phase, std::vector<control_qubit_type> const& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, control_qubits_{control_qubits},
        name_{std::string(control_qubits_.size() - std::size_t{1u}, 'C').append("U1+")}
    { }

    adj_multi_controlled_u1::adj_multi_controlled_u1(
      real_type const& phase, std::vector<control_qubit_type>&& control_qubits)
      : ::bra::gate::gate{},
        phase_{phase}, control_qubits_{std::move(control_qubits)},
        name_{std::string(control_qubits_.size() - std::size_t{1u}, 'C').append("U1+")}
    { }

    ::bra::state& adj_multi_controlled_u1::do_apply(::bra::state& state) const
    { return state.adj_multi_controlled_u1(phase_, control_qubits_); }

    std::string const& adj_multi_controlled_u1::do_name() const { return name_; }
    std::string adj_multi_controlled_u1::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& control_qubit: control_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << control_qubit;
      repr_stream
        << std::right
        << std::setw(parameter_width) << phase_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
