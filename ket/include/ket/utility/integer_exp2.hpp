#ifndef KET_UTILITY_INTEGER_EXP2_HPP
# define KET_UTILITY_INTEGER_EXP2_HPP

# include <boost/config.hpp>


namespace ket
{
  namespace utility
  {
    template <typename UnsignedInteger, typename Exponent>
    inline BOOST_CONSTEXPR UnsignedInteger integer_exp2(Exponent const exponent) BOOST_NOEXCEPT_OR_NOTHROW
    { return static_cast<UnsignedInteger>(1u) << exponent; }
  }
}


#endif

