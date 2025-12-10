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
    measurement::measurement(yampi::rank const root, int const precision)
      : ::bra::gate::gate{}, root_{root}, precision_{precision}
    { }

    ::bra::state& measurement::do_apply(::bra::state& state) const
    { return state.measurement(root_, precision_); }
#else // BRA_NO_MPI
    measurement::measurement(int const precision)
      : ::bra::gate::gate{}, precision_{precision}
    { }

    ::bra::state& measurement::do_apply(::bra::state& state) const
    { return state.measurement(precision_); }
#endif // BRA_NO_MPI

    std::string const& measurement::do_name() const { return name_; }
    std::string measurement::do_representation(
      std::ostringstream& repr_stream, int const) const
    { return repr_stream.str(); }
  } // namespace gate
} // namespace bra
