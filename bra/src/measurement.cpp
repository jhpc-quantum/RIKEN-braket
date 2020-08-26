#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#ifndef BRA_NO_MPI
# include <yampi/rank.hpp>
#endif

#include <bra/gate/gate.hpp>
#include <bra/gate/measurement.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const measurement::name_ = "MEASURE";

#ifndef BRA_NO_MPI
    measurement::measurement(yampi::rank const root)
      : ::bra::gate::gate{}, root_{root}
    { }

    ::bra::state& measurement::do_apply(::bra::state& state) const
    { return state.measurement(root_); }
#else // BRA_NO_MPI
    measurement::measurement()
      : ::bra::gate::gate{}
    { }

    ::bra::state& measurement::do_apply(::bra::state& state) const
    { return state.measurement(); }
#endif // BRA_NO_MPI

    std::string const& measurement::do_name() const { return name_; }
    std::string measurement::do_representation(
      std::ostringstream& repr_stream, int const) const
    { return repr_stream.str(); }
  } // namespace gate
} // namespace bra
