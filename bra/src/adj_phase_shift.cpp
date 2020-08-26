#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_phase_shift.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_phase_shift::name_ = "R+";

    adj_phase_shift::adj_phase_shift(
      int const phase_exponent, complex_type const phase_coefficient, qubit_type const qubit)
      : ::bra::gate::gate{},
        phase_exponent_{phase_exponent}, phase_coefficient_{phase_coefficient}, qubit_{qubit}
    { }

    ::bra::state& adj_phase_shift::do_apply(::bra::state& state) const
    { return state.adj_phase_shift(phase_coefficient_, qubit_); }

    std::string const& adj_phase_shift::do_name() const { return name_; }
    std::string adj_phase_shift::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_
        << std::setw(parameter_width) << phase_exponent_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
