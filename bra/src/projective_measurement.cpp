#include <boost/config.hpp>

#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#ifndef BRA_NO_MPI
# include <yampi/rank.hpp>
#endif

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/projective_measurement.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const projective_measurement::name_ = "M";

#ifndef BRA_NO_MPI
    projective_measurement::projective_measurement(qubit_type const qubit, yampi::rank const root)
      : ::bra::gate::gate(), qubit_(qubit), root_(root)
    { }

    ::bra::state& projective_measurement::do_apply(::bra::state& state) const
    { return state.projective_measurement(qubit_, root_); }
#else // BRA_NO_MPI
    projective_measurement::projective_measurement(qubit_type const qubit)
      : ::bra::gate::gate(), qubit_(qubit)
    { }

    ::bra::state& projective_measurement::do_apply(::bra::state& state) const
    { return state.projective_measurement(qubit_); }
#endif // BRA_NO_MPI

    std::string const& projective_measurement::do_name() const { return name_; }
    std::string projective_measurement::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_;
      return repr_stream.str();
    }
  }
}

