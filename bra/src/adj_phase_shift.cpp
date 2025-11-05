#include <string>
#include <ios>
#include <iomanip>
#include <sstream>
#include <utility>

#include <boost/variant/variant.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <ket/qubit_io.hpp>
#include <ket/control_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_phase_shift.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_phase_shift::name_ = "R+";

    adj_phase_shift::adj_phase_shift(
      boost::variant<int_type, std::string> const& phase_exponent,
      control_qubit_type const control_qubit)
      : ::bra::gate::gate{},
        phase_exponent_{phase_exponent}, control_qubit_{control_qubit}
    { }

    ::bra::state& adj_phase_shift::do_apply(::bra::state& state) const
    { return state.adj_phase_shift(phase_exponent_, control_qubit_); }

    std::string const& adj_phase_shift::do_name() const { return name_; }
    std::string adj_phase_shift::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<int_type>{}, phase_exponent_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
