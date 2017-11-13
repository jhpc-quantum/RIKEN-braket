#ifndef KET_UTILITY_INTEGER_LOG2_HPP
# define KET_UTILITY_INTEGER_LOG2_HPP

# include <boost/config.hpp>


namespace ket
{
  namespace utility
  {
    namespace integer_log2_detail
    {
      template <typename UnsignedInteger, typename Result>
      inline BOOST_CONSTEXPR Result integer_log2(
        UnsignedInteger const value, Result result) BOOST_NOEXCEPT_OR_NOTHROW
      {
        return value/static_cast<UnsignedInteger>(2u) == static_cast<UnsignedInteger>(0u)
          ? result
          : ::ket::utility::integer_log2_detail::integer_log2(
              value/static_cast<UnsignedInteger>(2u), ++result);
      }
    }

    template <typename Result, typename UnsignedInteger>
    inline BOOST_CONSTEXPR Result integer_log2(UnsignedInteger const value) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::utility::integer_log2_detail::integer_log2(value, Result(0u)); }
  }
}


#endif

