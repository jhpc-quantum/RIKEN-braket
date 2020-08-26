#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/toffoli.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const toffoli::name_ = "Toffoli";

    toffoli::toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
      : ::bra::gate::gate{}, target_qubit_{target_qubit}, control_qubit1_{control_qubit1}, control_qubit2_{control_qubit2}
    { }

    ::bra::state& toffoli::do_apply(::bra::state& state) const
    { return state.toffoli(target_qubit_, control_qubit1_, control_qubit2_); }

    std::string const& toffoli::do_name() const { return name_; }
    std::string toffoli::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit1_
        << std::setw(parameter_width) << control_qubit2_
        << std::setw(parameter_width) << target_qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
