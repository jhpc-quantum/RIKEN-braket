#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/variant.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/global_phase.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const global_phase::name_ = "PHASE";

    global_phase::global_phase(boost::variant<real_type, std::string> const& phase)
      : ::bra::gate::gate{}, phase_{phase}
    { }

    ::bra::state& global_phase::do_apply(::bra::state& state) const
    { return state.global_phase(phase_); }

    std::string const& global_phase::do_name() const { return name_; }

    std::string global_phase::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width)
        << boost::apply_visitor(::bra::gate::gate_detail::output_visitor<real_type>{}, phase_);
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
