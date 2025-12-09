#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <bra/gate/gate.hpp>
#include <bra/gate/println_op.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const println_op::name_ = "PRINTLN";

    println_op::println_op(std::vector<std::string> const& variables_or_literals)
      : ::bra::gate::gate{}, variables_or_literals_{variables_or_literals}
    { }

    println_op::println_op(std::vector<std::string>&& variables_or_literals)
      : ::bra::gate::gate{}, variables_or_literals_{std::move(variables_or_literals)}
    { }

    ::bra::state& println_op::do_apply(::bra::state& state) const
    {
      state.invoke_println_operation(variables_or_literals_);
      return state;
    }

    std::string const& println_op::do_name() const { return name_; }
    std::string println_op::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      for (auto const& variable_or_literal: variables_or_literals_)
        repr_stream
          << std::right
          << std::setw(parameter_width) << variable_or_literal;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
