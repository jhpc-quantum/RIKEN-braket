#include <boost/config.hpp>

#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_u3.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_u3::name_ = "U3+";

    adj_u3::adj_u3(real_type const phase1, real_type const phase2, real_type const phase3, qubit_type const qubit)
      : ::bra::gate::gate(),
        phase1_(phase1), phase2_(phase2), phase3_(phase3), qubit_(qubit)
    { }

    ::bra::state& adj_u3::do_apply(::bra::state& state) const
    { return state.adj_u3(phase1_, phase2_, phase3_, qubit_); }

    std::string const& adj_u3::do_name() const { return name_; }
    std::string adj_u3::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_
        << std::setw(parameter_width) << phase1_
        << std::setw(parameter_width) << phase2_
        << std::setw(parameter_width) << phase3_;
      return repr_stream.str();
    }
  }
}

