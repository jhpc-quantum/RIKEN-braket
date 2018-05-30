#include <boost/config.hpp>

#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <yampi/rank.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/exit.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const exit::name_ = "EXIT";

    exit::exit(yampi::rank const root)
      : ::bra::gate::gate(), root_(root)
    { }

    ::bra::state& exit::do_apply(::bra::state& state) const
    { return state.exit(root_); }

    std::string const& exit::do_name() const { return name_; }
    std::string exit::do_representation(
      std::ostringstream& repr_stream, int const) const
    { return repr_stream.str(); }
  }
}

