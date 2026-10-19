#ifndef BRA_CONTROL_IO_HPP
# define BRA_CONTROL_IO_HPP

# include <string>
# include <bra/types.hpp>

namespace bra
{
  // QCX keeps the legacy spelling for positive controls. Format the complete
  // operand before insertion so std::setw applies to the suffix as well.
  inline auto control_operand(::bra::control_qubit_type const control) -> std::string
  {
    auto result = std::to_string(static_cast< ::bra::bit_integer_type >(control.qubit()));
    if (ket::control_value(control) == 0u)
      result += "#0";
    return result;
  }
}

#endif // BRA_CONTROL_IO_HPP
