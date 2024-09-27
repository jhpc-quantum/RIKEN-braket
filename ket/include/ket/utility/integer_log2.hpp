#ifndef KET_UTILITY_INTEGER_LOG2_HPP
# define KET_UTILITY_INTEGER_LOG2_HPP


namespace ket
{
  namespace utility
  {
    template <typename Result, typename UnsignedInteger>
    inline constexpr auto integer_log2(UnsignedInteger value) noexcept -> Result
    {
      value >>= UnsignedInteger{1u};
      auto result = Result{0u};

      while (value != UnsignedInteger{0u})
      {
        ++result;
        value >>= UnsignedInteger{1u};
      }

      return result;
    }
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_INTEGER_LOG2_HPP
