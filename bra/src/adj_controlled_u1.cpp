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
#include <bra/gate/adj_controlled_u1.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_controlled_u1::name_ = "CU1+";

    adj_controlled_u1::adj_controlled_u1(
      boost::variant<real_type, std::string> const& phase,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
      : ::bra::gate::gate{}, phase_{phase}, control_qubit1_{control_qubit1}, control_qubit2_{control_qubit2}
    { }

    ::bra::state& adj_controlled_u1::do_apply(::bra::state& state) const
    { return state.adj_controlled_u1(phase_, control_qubit1_, control_qubit2_); }

    std::string const& adj_controlled_u1::do_name() const { return name_; }
    std::string adj_controlled_u1::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit1_
        << std::setw(parameter_width) << control_qubit2_
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
