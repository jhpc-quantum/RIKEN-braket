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
#include <bra/gate/adj_controlled_phase_shift.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_controlled_phase_shift::name_ = "U+";

    adj_controlled_phase_shift::adj_controlled_phase_shift(
      boost::variant<int_type, std::string> const& phase_exponent,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2)
      : ::bra::gate::gate{},
        phase_exponent_{phase_exponent},
        control_qubit1_{control_qubit1}, control_qubit2_{control_qubit2}
    { }

    ::bra::state& adj_controlled_phase_shift::do_apply(::bra::state& state) const
    { return state.adj_controlled_phase_shift(phase_exponent_, control_qubit1_, control_qubit2_); }

    std::string const& adj_controlled_phase_shift::do_name() const { return name_; }
    std::string adj_controlled_phase_shift::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit1_
        << std::setw(parameter_width) << control_qubit2_
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<int_type>{}, phase_exponent_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
