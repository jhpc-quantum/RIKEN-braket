#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/controlled_t_gate.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const controlled_t_gate::name_ = "CT";

    controlled_t_gate::controlled_t_gate(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
      : ::bra::gate::gate{},
        control_qubit1_{control_qubit1},
        control_qubit2_{control_qubit2}
    { }

    ::bra::state& controlled_t_gate::do_apply(::bra::state& state) const
    { return state.controlled_phase_shift(3, control_qubit1_, control_qubit2_); }

    std::string const& controlled_t_gate::do_name() const { return name_; }
    std::string controlled_t_gate::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit1_
        << std::setw(parameter_width) << control_qubit2_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
