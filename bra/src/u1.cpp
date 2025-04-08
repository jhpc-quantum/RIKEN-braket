#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/u1.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const u1::name_ = "U1";

    u1::u1(real_type const phase, control_qubit_type const control_qubit)
      : ::bra::gate::gate{},
        phase_{phase}, control_qubit_{control_qubit}
    { }

    ::bra::state& u1::do_apply(::bra::state& state) const
    { return state.u1(phase_, control_qubit_); }

    std::string const& u1::do_name() const { return name_; }
    std::string u1::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_
        << std::setw(parameter_width) << phase_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
