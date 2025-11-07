#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#ifndef BRA_NO_MPI
# include <yampi/rank.hpp>
#endif

#include <bra/gate/gate.hpp>
#include <bra/gate/amplitudes.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const amplitudes::name_ = "AMPLITUDES";

#ifndef BRA_NO_MPI
    amplitudes::amplitudes(yampi::rank const root)
      : ::bra::gate::gate{}, root_{root}
    { }

    ::bra::state& amplitudes::do_apply(::bra::state& state) const
    { return state.amplitudes(root_); }
#else // BRA_NO_MPI
    amplitudes::amplitudes()
      : ::bra::gate::gate{}
    { }

    ::bra::state& amplitudes::do_apply(::bra::state& state) const
    { return state.amplitudes(); }
#endif // BRA_NO_MPI

    std::string const& amplitudes::do_name() const { return name_; }
    std::string amplitudes::do_representation(
      std::ostringstream& repr_stream, int const) const
    { return repr_stream.str(); }
  } // namespace gate
} // namespace bra
