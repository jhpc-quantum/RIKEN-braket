#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/hadamard.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const hadamard::name_ = "H";

    hadamard::hadamard(qubit_type const qubit)
      : ::bra::gate::gate{}, qubit_{qubit}
    { }

    ::bra::state& hadamard::do_apply(::bra::state& state) const
    { return state.hadamard(qubit_); }

    std::string const& hadamard::do_name() const { return name_; }
    std::string hadamard::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
