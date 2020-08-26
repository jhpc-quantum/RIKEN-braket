#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#ifndef BRA_NO_MPI
# include <yampi/rank.hpp>
#endif

#include <bra/gate/gate.hpp>
#include <bra/gate/exit.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const exit::name_ = "EXIT";

#ifndef BRA_NO_MPI
    exit::exit(yampi::rank const root)
      : ::bra::gate::gate{}, root_{root}
    { }
#else // BRA_NO_MPI
    exit::exit() : ::bra::gate::gate{} { }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
    ::bra::state& exit::do_apply(::bra::state& state) const
    { return state.exit(root_); }
#else // BRA_NO_MPI
    ::bra::state& exit::do_apply(::bra::state& state) const
    { return state.exit(); }
#endif // BRA_NO_MPI

    std::string const& exit::do_name() const { return name_; }
    std::string exit::do_representation(
      std::ostringstream& repr_stream, int const) const
    { return repr_stream.str(); }
  } // namespace gate
} // namespace bra
