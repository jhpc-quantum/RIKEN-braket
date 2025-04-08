#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/phase_shift.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const phase_shift::name_ = "R";

    phase_shift::phase_shift(
      int const phase_exponent, complex_type const& phase_coefficient, control_qubit_type const control_qubit)
      : ::bra::gate::gate{},
        phase_exponent_{phase_exponent}, phase_coefficient_{phase_coefficient}, control_qubit_{control_qubit}
    { }

    ::bra::state& phase_shift::do_apply(::bra::state& state) const
    { return state.phase_shift(phase_coefficient_, control_qubit_); }

    std::string const& phase_shift::do_name() const { return name_; }
    std::string phase_shift::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_
        << std::setw(parameter_width) << phase_exponent_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
