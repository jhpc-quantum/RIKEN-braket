#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/controlled_phase_shift_.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const controlled_phase_shift_::name_ = "CR";

    controlled_phase_shift_::controlled_phase_shift_(
      int const phase_exponent,
      complex_type const& phase_coefficient,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2)
      : ::bra::gate::gate{},
        phase_exponent_{phase_exponent}, phase_coefficient_{phase_coefficient},
        control_qubit1_{control_qubit1}, control_qubit2_{control_qubit2}
    { }

    ::bra::state& controlled_phase_shift_::do_apply(::bra::state& state) const
    { return state.controlled_phase_shift(phase_coefficient_, control_qubit1_, control_qubit2_); }

    std::string const& controlled_phase_shift_::do_name() const { return name_; }
    std::string controlled_phase_shift_::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit1_
        << std::setw(parameter_width) << control_qubit2_
        << std::setw(parameter_width) << phase_exponent_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
