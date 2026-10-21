#include <stdexcept>

#include <bra/classical_function_call.hpp>


namespace bra
{
  auto evaluate_classical_function(::bra::classical_function_call const&)
    -> std::vector< ::bra::classical_function_value >
  {
    // Checked EXP/ARG evaluation is introduced in Stage 3. Do not temporarily
    // enable unchecked library calls that violate the numerical contract.
    throw std::logic_error{"numerical function evaluation is not implemented yet"};
  }
} // namespace bra
