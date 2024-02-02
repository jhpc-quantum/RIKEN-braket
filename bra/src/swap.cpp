#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/swap.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const swap::name_ = "SWAP";

    swap::swap(qubit_type const qubit1, qubit_type const qubit2)
      : ::bra::gate::gate{}, qubit1_{qubit1}, qubit2_{qubit2}
    { }

    ::bra::state& swap::do_apply(::bra::state& state) const
    { return state.swap(qubit1_, qubit2_); }

    std::string const& swap::do_name() const { return name_; }
    std::string swap::do_representation(
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
