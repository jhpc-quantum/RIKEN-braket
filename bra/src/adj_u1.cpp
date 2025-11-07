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
#include <bra/gate/adj_u1.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_u1::name_ = "U1+";

    adj_u1::adj_u1(
      boost::variant<real_type, std::string> const& phase,
      control_qubit_type const control_qubit)
      : ::bra::gate::gate{},
        phase_{phase}, control_qubit_{control_qubit}
    { }

    ::bra::state& adj_u1::do_apply(::bra::state& state) const
    { return state.adj_u1(phase_, control_qubit_); }

    std::string const& adj_u1::do_name() const { return name_; }
    std::string adj_u1::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << control_qubit_
        << std::setw(parameter_width) << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
