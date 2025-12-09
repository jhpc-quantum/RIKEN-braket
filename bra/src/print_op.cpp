#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <bra/gate/gate.hpp>
#include <bra/gate/print_op.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const print_op::name_ = "PRINT";

    print_op::print_op(std::vector<std::string> const& variables_or_literals)
      : ::bra::gate::gate{}, variables_or_literals_{variables_or_literals}
    { }

    print_op::print_op(std::vector<std::string>&& variables_or_literals)
      : ::bra::gate::gate{}, variables_or_literals_{std::move(variables_or_literals)}
    { }

    ::bra::state& print_op::do_apply(::bra::state& state) const
    {
      state.invoke_print_operation(variables_or_literals_);
      return state;
    }

    std::string const& print_op::do_name() const { return name_; }
    std::string print_op::do_representation(
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
