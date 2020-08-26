#ifndef KET_UTILITY_INTEGER_LOG2_HPP
# define KET_UTILITY_INTEGER_LOG2_HPP

# include <boost/config.hpp>


namespace ket
{
  namespace utility
  {
# ifndef BOOST_NO_CXX14_CONSTEXPR
    template <typename Result, typename UnsignedInteger>
    inline constexpr Result integer_log2(UnsignedInteger value) noexcept
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
# else // BOOST_NO_CXX14_CONSTEXPR
    namespace integer_log2_detail
    {
      template <typename UnsignedInteger, typename Result>
      inline constexpr Result integer_log2(
        UnsignedInteger const value, Result result) noexcept
      {
        return value >> UnsignedInteger{1u} == UnsignedInteger{0u}
          ? result
          : ::ket::utility::integer_log2_detail::integer_log2(
              value >> UnsignedInteger{1u}, ++result);
      }
    } // namespace integer_log2_detail

    template <typename Result, typename UnsignedInteger>
    inline constexpr Result integer_log2(UnsignedInteger const value) noexcept
    { return ::ket::utility::integer_log2_detail::integer_log2(value, Result{0u}); }
# endif // BOOST_NO_CXX14_CONSTEXPR
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_INTEGER_LOG2_HPP
