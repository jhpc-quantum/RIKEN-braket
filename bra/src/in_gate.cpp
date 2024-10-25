#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/in_gate.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    in_gate::in_gate(std::vector<qubit_type>&& qubits)
      : ::bra::gate::gate{}, qubits_{std::move(qubits)}, name_{std::string(qubits_.size(), 'I')}
    { }

    ::bra::state& in_gate::do_apply(::bra::state& state) const
    { return state.in_gate(qubits_); }

    std::string const& in_gate::do_name() const { return name_; }
    std::string in_gate::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& qubit: qubits_)
        repr_stream << std::right << std::setw(parameter_width) << qubit;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
