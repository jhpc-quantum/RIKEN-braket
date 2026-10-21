#ifndef BRA_VARIABLE_TYPE_HPP
# define BRA_VARIABLE_TYPE_HPP


namespace bra
{
  enum class variable_type : int
  { real = 0, complex_ = 1, integer = 2, pauli_string_space = 3, unsigned_integer = 4 };
} // namespace bra


#endif // BRA_VARIABLE_TYPE_HPP
