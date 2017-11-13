#include <boost/config.hpp>

#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <yampi/rank.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/measurement.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const measurement::name_ = "MEASURE";

    measurement::measurement(yampi::rank const root)
      : ::bra::gate::gate(), root_(root)
    { }

    ::bra::state& measurement::do_apply(::bra::state& state) const
    { return state.measurement(root_); }

    std::string const& measurement::do_name() const { return name_; }
    std::string measurement::do_representation(
      std::ostringstream& repr_stream, int const) const
    { return repr_stream.str(); }
  }
}

