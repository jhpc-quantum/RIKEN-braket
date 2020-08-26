#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_controlled_v.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_controlled_v::name_ = "V+";

    adj_controlled_v::adj_controlled_v(
      int const phase_exponent,
      complex_type const phase_coefficient,
      qubit_type const target_qubit,
      control_qubit_type const control_qubit)
      : ::bra::gate::gate{},
        phase_exponent_{phase_exponent}, phase_coefficient_{phase_coefficient},
        target_qubit_{target_qubit}, control_qubit_{control_qubit}
    { }

    ::bra::state& adj_controlled_v::do_apply(::bra::state& state) const
    { return state.adj_controlled_v(phase_coefficient_, target_qubit_, control_qubit_); }

    std::string const& adj_controlled_v::do_name() const { return name_; }
    std::string adj_controlled_v::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_
        << std::setw(parameter_width) << target_qubit_
        << std::setw(parameter_width) << phase_exponent_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
