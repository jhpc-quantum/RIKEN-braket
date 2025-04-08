#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/controlled_u1.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const controlled_u1::name_ = "CU1";

    controlled_u1::controlled_u1(
      real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
      : ::bra::gate::gate{}, phase_{phase}, control_qubit1_{control_qubit1}, control_qubit2_{control_qubit2}
    { }

    ::bra::state& controlled_u1::do_apply(::bra::state& state) const
    { return state.controlled_u1(phase_, control_qubit1_, control_qubit2_); }

    std::string const& controlled_u1::do_name() const { return name_; }
    std::string controlled_u1::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit1_
        << std::setw(parameter_width) << control_qubit2_
        << std::setw(parameter_width) << phase_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
