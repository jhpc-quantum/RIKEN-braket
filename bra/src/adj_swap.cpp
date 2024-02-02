#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/adj_swap.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const adj_swap::name_ = "SWAP+";

    adj_swap::adj_swap(qubit_type const qubit1, qubit_type const qubit2)
      : ::bra::gate::gate{}, qubit1_{qubit1}, qubit2_{qubit2}
    { }

    ::bra::state& adj_swap::do_apply(::bra::state& state) const
    { return state.adj_swap(qubit1_, qubit2_); }

    std::string const& adj_swap::do_name() const { return name_; }
    std::string adj_swap::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit1_
        << std::setw(parameter_width) << qubit2_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
