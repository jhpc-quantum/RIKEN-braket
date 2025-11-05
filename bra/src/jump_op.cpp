#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <ket/qubit_io.hpp>

#include <bra/gate/gate.hpp>
#include <bra/gate/jump_op.hpp>
#include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    std::string const jump_op::name_ = "JUMP";

    jump_op::jump_op(std::string const& label)
      : ::bra::gate::gate{}, label_{label}
    { }

    ::bra::state& jump_op::do_apply(::bra::state& state) const
    {
      state.invoke_jump_operation(label_);
      return state;
    }

    std::string const& jump_op::do_name() const { return name_; }
    std::string jump_op::do_representation(
      std::ostringstream& repr_stream, int const parameter_width) const
    {
      repr_stream
        << std::right
        << std::setw(parameter_width) << label_;
      return repr_stream.str();
    }
  } // namespace gate
} // namespace bra
