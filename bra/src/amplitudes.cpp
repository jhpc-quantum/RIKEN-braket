#include <string>
#include <vector>
#include <ios>
#include <iomanip>
#include <sstream>

#ifndef BRA_NO_MPI
# include <yampi/rank.hpp>
#endif

#include <bra/gate/gate.hpp>
#include <bra/gate/amplitudes.hpp>
#include <bra/types.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const amplitudes::name_ = "AMPLITUDES";

#ifndef BRA_NO_MPI
    amplitudes::amplitudes(yampi::rank const root, std::vector< ::bra::state_integer_type > const& amplitude_indices)
      : ::bra::gate::gate{}, root_{root}, amplitude_indices_{amplitude_indices}
    { }

    amplitudes::amplitudes(yampi::rank const root, std::vector< ::bra::state_integer_type >&& amplitude_indices)
      : ::bra::gate::gate{}, root_{root}, amplitude_indices_{std::move(amplitude_indices)}
    { }

    ::bra::state& amplitudes::do_apply(::bra::state& state) const
    { return state.amplitudes(root_, amplitude_indices_); }
#else // BRA_NO_MPI
    amplitudes::amplitudes(std::vector< ::bra::state_integer_type > const& amplitude_indices)
      : ::bra::gate::gate{}, amplitude_indices_{amplitude_indices}
    { }

    amplitudes::amplitudes(std::vector< ::bra::state_integer_type >&& amplitude_indices)
      : ::bra::gate::gate{}, amplitude_indices_{std::move(amplitude_indices)}
    { }

    ::bra::state& amplitudes::do_apply(::bra::state& state) const
    { return state.amplitudes(amplitude_indices_); }
#endif // BRA_NO_MPI

    std::string const& amplitudes::do_name() const { return name_; }
    std::string amplitudes::do_representation(
      std::ostringstream& repr_stream, int const) const
    { return repr_stream.str(); }
  } // namespace gate
} // namespace bra
