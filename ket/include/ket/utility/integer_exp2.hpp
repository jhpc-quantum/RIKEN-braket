#ifndef KET_UTILITY_INTEGER_EXP2_HPP
# define KET_UTILITY_INTEGER_EXP2_HPP


namespace ket
{
  namespace utility
  {
    template <typename UnsignedInteger, typename Exponent>
    inline constexpr UnsignedInteger integer_exp2(Exponent const exponent) noexcept
    { return UnsignedInteger{1u} << exponent; }
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_INTEGER_EXP2_HPP
