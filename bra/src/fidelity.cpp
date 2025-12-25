#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/fidelity.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const fidelity::name_ = "FIDELITY";

    fidelity::fidelity(std::string const& remote_circuit_index_or_all)
      : ::bra::gate::gate{}, remote_circuit_index_or_all_{remote_circuit_index_or_all}
    { }

    ::bra::state& fidelity::do_apply(::bra::state& state) const
    {
      state.fidelity(remote_circuit_index_or_all_);
      return state;
    }

    std::string const& fidelity::do_name() const { return name_; }
    std::string fidelity::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    { return ""; }
  } // namespace gate
} // namespace bra
