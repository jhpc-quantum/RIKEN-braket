#include <boost/config.hpp>

#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/set.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const set::name_ = "SET";

    set::set(qubit_type const qubit)
      : ::bra::gate::gate(), qubit_(qubit)
    { }

    ::bra::state& set::do_apply(::bra::state& state) const
    { return state.set(qubit_); }

    std::string const& set::do_name() const { return name_; }
    std::string set::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_;
      return repr_stream.str();
    }
  }
}

