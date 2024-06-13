#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/controlled_u3.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const controlled_u3::name_ = "CU3";

    controlled_u3::controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
      : ::bra::gate::gate{},
        phase1_{phase1}, phase2_{phase2}, phase3_{phase3},
        target_qubit_{target_qubit}, control_qubit_{control_qubit}
    { }

    ::bra::state& controlled_u3::do_apply(::bra::state& state) const
    { return state.controlled_u3(phase1_, phase2_, phase3_, target_qubit_, control_qubit_); }

    std::string const& controlled_u3::do_name() const { return name_; }
    std::string controlled_u3::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_
        << std::setw(parameter_width) << target_qubit_
        << std::setw(parameter_width) << phase1_
        << std::setw(parameter_width) << phase2_
        << std::setw(parameter_width) << phase3_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
