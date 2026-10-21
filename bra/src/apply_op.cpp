#include <iomanip>
#include <sstream>
#include <utility>

#include <bra/gate/apply_op.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const apply_op::name_ = "APPLY";

    apply_op::apply_op(std::vector<std::string> destinations, ::bra::classical_function_id const function,
                       std::vector<std::string> operands)
      : destinations_{std::move(destinations)}, function_{function}, operands_{std::move(operands)} { }

    ::bra::state& apply_op::do_apply(::bra::state& state) const
    {
      state.invoke_apply_operation(destinations_, function_, operands_);
      return state;
    }

    std::string const& apply_op::do_name() const { return name_; }

    std::string apply_op::do_representation(std::ostringstream& stream, int const parameter_width) const
    {
      // Always retain a token separator, even when a name exceeds the width.
      for (auto const& destination: destinations_)
        stream << ' ' << std::right << std::setw(parameter_width) << destination;
      stream << ' ' << std::setw(parameter_width) << "<-"
             << ' ' << std::setw(parameter_width) << ::bra::classical_function_name(function_);
      for (auto const& operand: operands_)
        stream << ' ' << std::setw(parameter_width) << operand;
      return stream.str();
    }
  } // namespace gate
} // namespace bra
