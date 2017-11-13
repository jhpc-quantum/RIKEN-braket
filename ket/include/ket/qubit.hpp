#ifndef KET_QUBIT_HPP
#define KET_QUBIT_HPP

# include <boost/config.hpp>

# include <boost/cstdint.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#  include <type_traits>
# else
#  include <boost/type_traits/is_integral.hpp>
#  include <boost/utility/enable_if.hpp>
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_enable_if std::enable_if
#   define KET_is_integral std::is_integral
# else
#   define KET_enable_if boost::enable_if_c
#   define KET_is_integral boost::is_integral
# endif


namespace ket
{
  template <typename StateInteger = boost::uint64_t, typename BitInteger = unsigned int>
  class qubit
  {
    BitInteger bit_;

   public:
    typedef BitInteger bit_integer_type;
    typedef StateInteger state_integer_type;

    BOOST_CONSTEXPR qubit() BOOST_NOEXCEPT_OR_NOTHROW : bit_(0u) { }

    template <typename BitInteger_>
    explicit BOOST_CONSTEXPR qubit(BitInteger_ const bit) BOOST_NOEXCEPT_OR_NOTHROW
      : bit_(static_cast<BitInteger>(bit))
    { }

# ifndef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
    explicit
# endif
    operator BitInteger() const { return bit_; }

    qubit& operator++() BOOST_NOEXCEPT_OR_NOTHROW { ++bit_; return *this; }
    qubit operator++(int) BOOST_NOEXCEPT_OR_NOTHROW
    {
      qubit result = *this;
      ++(*this);
      return result;
    }
    qubit& operator--() BOOST_NOEXCEPT_OR_NOTHROW { --bit_; return *this; }
    qubit operator--(int) BOOST_NOEXCEPT_OR_NOTHROW
    {
      qubit result = *this;
      --(*this);
      return result;
    }
    qubit& operator+=(qubit const other) BOOST_NOEXCEPT_OR_NOTHROW
    { bit_ += other.bit_; return *this; }
    qubit& operator-=(qubit const other) BOOST_NOEXCEPT_OR_NOTHROW
    { bit_ -= other.bit_; return *this; }
  };

  template <typename StateInteger, typename BitInteger>
  inline BOOST_CONSTEXPR bool operator==(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) BOOST_NOEXCEPT_OR_NOTHROW
  { return static_cast<BitInteger>(qubit1) == static_cast<BitInteger>(qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline BOOST_CONSTEXPR bool operator!=(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) BOOST_NOEXCEPT_OR_NOTHROW
  { return not (qubit1 == qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline BOOST_CONSTEXPR bool operator<(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) BOOST_NOEXCEPT_OR_NOTHROW
  { return static_cast<BitInteger>(qubit1) < static_cast<BitInteger>(qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline BOOST_CONSTEXPR bool operator>(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) BOOST_NOEXCEPT_OR_NOTHROW
  { return qubit2 < qubit1; }

  template <typename StateInteger, typename BitInteger>
  inline BOOST_CONSTEXPR bool operator<=(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) BOOST_NOEXCEPT_OR_NOTHROW
  { return not (qubit1 > qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline BOOST_CONSTEXPR bool operator>=(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) BOOST_NOEXCEPT_OR_NOTHROW
  { return not (qubit1 < qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline BOOST_CONSTEXPR qubit<StateInteger, BitInteger> operator+(
    ::ket::qubit<StateInteger, BitInteger> qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) BOOST_NOEXCEPT_OR_NOTHROW
  { return qubit1 += qubit2; }

  template <typename StateInteger, typename BitInteger>
  inline BOOST_CONSTEXPR qubit<StateInteger, BitInteger> operator-(
    ::ket::qubit<StateInteger, BitInteger> qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) BOOST_NOEXCEPT_OR_NOTHROW
  { return qubit1 -= qubit2; }

  template <typename Value, typename StateInteger, typename BitInteger>
  inline BOOST_CONSTEXPR
  typename KET_enable_if<KET_is_integral<Value>::value, Value>::type
  operator<<(
    Value const value,
    ::ket::qubit<StateInteger, BitInteger> const qubit) BOOST_NOEXCEPT_OR_NOTHROW
  { return value << static_cast<BitInteger>(qubit); }

  template <typename Value, typename StateInteger, typename BitInteger>
  inline BOOST_CONSTEXPR
  typename KET_enable_if<KET_is_integral<Value>::value, Value>::type
  operator>>(
    Value const value,
    ::ket::qubit<StateInteger, BitInteger> const qubit) BOOST_NOEXCEPT_OR_NOTHROW
  { return value >> static_cast<BitInteger>(qubit); }


  template <typename BitInteger>
  inline BOOST_CONSTEXPR ::ket::qubit<boost::uint64_t, BitInteger>
  make_qubit(BitInteger const bit) BOOST_NOEXCEPT_OR_NOTHROW
  { return ::ket::qubit<boost::uint64_t, BitInteger>(bit); }

  template <typename StateInteger, typename BitInteger>
  inline BOOST_CONSTEXPR ::ket::qubit<StateInteger, BitInteger>
  make_qubit(BitInteger const bit) BOOST_NOEXCEPT_OR_NOTHROW
  { return ::ket::qubit<StateInteger, BitInteger>(bit); }


# ifndef BOOST_NO_CXX11_USER_DEFINED_LITERALS
  namespace qubit_literals
  {
    inline BOOST_CONSTEXPR ::ket::qubit<boost::uint64_t, unsigned int>
    operator"" _q(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::qubit<boost::uint64_t, unsigned int>{static_cast<unsigned int>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::qubit<boost::uint64_t, unsigned short int>
    operator"" _qs(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::qubit<boost::uint64_t, unsigned short int>{static_cast<unsigned short>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::qubit<boost::uint64_t, unsigned long int>
    operator"" _ql(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::qubit<boost::uint64_t, unsigned long int>{static_cast<unsigned long>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::qubit<boost::uint64_t, unsigned long long int>
    operator"" _qll(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::qubit<boost::uint64_t, unsigned long long int>{bit}; }

    inline BOOST_CONSTEXPR ::ket::qubit<boost::uint64_t, boost::uint8_t>
    operator"" _q8(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::qubit<boost::uint64_t, boost::uint8_t>{static_cast<boost::uint8_t>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::qubit<boost::uint64_t, boost::uint16_t>
    operator"" _q16(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::qubit<boost::uint64_t, boost::uint16_t>{static_cast<boost::uint16_t>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::qubit<boost::uint64_t, boost::uint32_t>
    operator"" _q32(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::qubit<boost::uint64_t, boost::uint32_t>{static_cast<boost::uint32_t>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::qubit<boost::uint64_t, boost::uint64_t>
    operator"" _q64(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::qubit<boost::uint64_t, boost::uint64_t>{static_cast<boost::uint64_t>(bit)}; }
  }
# endif
}


# undef KET_is_integral
# undef KET_enable_if

#endif

