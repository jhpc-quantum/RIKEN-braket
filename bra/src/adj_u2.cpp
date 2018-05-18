#include <boost/config.hpp>

#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_u2.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_u2::name_ = "U2+";

    adj_u2::adj_u2(real_type const phase1, real_type const phase2, qubit_type const qubit)
      : ::bra::gate::gate(),
        phase1_(phase1), phase2_(phase2), qubit_(qubit)
    { }

    ::bra::state& adj_u2::do_apply(::bra::state& state) const
    { return state.adj_u2(phase1_, phase2_, qubit_); }

    std::string const& adj_u2::do_name() const { return name_; }
    std::string adj_u2::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_
        << std::setw(parameter_width) << phase1_
        << std::setw(parameter_width) << phase2_;
      return repr_stream.str();
    }
  }
}

