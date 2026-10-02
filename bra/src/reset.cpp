#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#ifndef BRA_NO_MPI
# include <yampi/rank.hpp>
#endif

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/reset.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const reset::name_ = "RESET";

#ifndef BRA_NO_MPI
    reset::reset(qubit_type const qubit, yampi::rank const root)
      : ::bra::gate::gate{}, qubit_{qubit}, root_{root}
    { }

    ::bra::state& reset::do_apply(::bra::state& state) const
    { return state.reset(qubit_, root_); }
#else
    reset::reset(qubit_type const qubit)
      : ::bra::gate::gate{}, qubit_{qubit}
    { }

    ::bra::state& reset::do_apply(::bra::state& state) const
    { return state.reset(qubit_); }
#endif

    std::string const& reset::do_name() const { return name_; }
    std::string reset::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
