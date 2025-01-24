#include <vector>
#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/begin_fusion.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const begin_fusion::name_ = "BEGIN FUSION";

    begin_fusion::begin_fusion(std::vector<qubit_type> const& fused_qubits)
      : ::bra::gate::gate{}, fused_qubits_{fused_qubits}
    { }

    begin_fusion::begin_fusion(std::vector<qubit_type>&& fused_qubits)
      : ::bra::gate::gate{}, fused_qubits_{std::move(fused_qubits)}
    { }

    ::bra::state& begin_fusion::do_apply(::bra::state& state) const
    { return state.begin_fusion(fused_qubits_); }

    std::string const& begin_fusion::do_name() const { return name_; }
    std::string begin_fusion::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto&& qubit: fused_qubits_)
        repr_stream << std::right << std::setw(parameter_width) << qubit;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
