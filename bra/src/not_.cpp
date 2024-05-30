#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/not_.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const not_::name_ = "NOT";

    not_::not_(qubit_type const qubit)
      : ::bra::gate::gate{}, qubit_{qubit}
    { }

    ::bra::state& not_::do_apply(::bra::state& state) const
    { return state.not_(qubit_); }

    std::string const& not_::do_name() const { return name_; }
    std::string not_::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << qubit_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
