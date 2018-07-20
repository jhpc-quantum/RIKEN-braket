#include <boost/config.hpp>

#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#ifndef BRA_NO_MPI
# include <yampi/rank.hpp>
#endif

#include <bra/gate/gate.hpp>
#include <bra/gate/generate_events.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const generate_events::name_ = "GENERATE EVENTS";

#ifndef BRA_NO_MPI
    generate_events::generate_events(yampi::rank const root, int num_events, int seed)
      : ::bra::gate::gate(), root_(root), num_events_(num_events), seed_(seed)
    { }

    ::bra::state& generate_events::do_apply(::bra::state& state) const
    { return state.generate_events(root_, num_events_, seed_); }
#else // BRA_NO_MPI
    generate_events::generate_events(int num_events, int seed)
      : ::bra::gate::gate(), num_events_(num_events), seed_(seed)
    { }

    ::bra::state& generate_events::do_apply(::bra::state& state) const
    { return state.generate_events(num_events_, seed_); }
#endif // BRA_NO_MPI

    std::string const& generate_events::do_name() const { return name_; }
    std::string generate_events::do_representation(
      std::ostringstream& repr_stream, int const) const
    { return repr_stream.str(); }
  }
}

