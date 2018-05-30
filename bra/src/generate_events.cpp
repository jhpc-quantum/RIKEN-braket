#include <boost/config.hpp>

#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <yampi/rank.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/generate_events.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const generate_events::name_ = "GENERATE EVENTS";

    generate_events::generate_events(yampi::rank const root, int num_events, int seed)
      : ::bra::gate::gate(), root_(root), num_events_(num_events), seed_(seed)
    { }

    ::bra::state& generate_events::do_apply(::bra::state& state) const
    { return state.generate_events(root_, num_events_, seed_); }

    std::string const& generate_events::do_name() const { return name_; }
    std::string generate_events::do_representation(
      std::ostringstream& repr_stream, int const) const
    { return repr_stream.str(); }
  }
}

