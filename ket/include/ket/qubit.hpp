#ifndef KET_QUBIT_HPP
# define KET_QUBIT_HPP

# include <cstdint>
# include <type_traits>


namespace ket
{
  template <typename StateInteger = std::uint64_t, typename BitInteger = unsigned int>
  class qubit
  {
    BitInteger bit_;

   public:
    using bit_integer_type = BitInteger;
    using state_integer_type = StateInteger;

    constexpr qubit() noexcept : bit_{0u} { }

    template <typename BitInteger_>
    explicit constexpr qubit(BitInteger_ const bit) noexcept
      : bit_{static_cast<BitInteger>(bit)}
    { }

    explicit operator BitInteger() const { return bit_; }

    qubit& operator++() noexcept { ++bit_; return *this; }
    qubit operator++(int) noexcept
    {
      auto result = *this;
      ++(*this);
      return result;
    }
    qubit& operator--() noexcept { --bit_; return *this; }
    qubit operator--(int) noexcept
    {
      auto result = *this;
      --(*this);
      return result;
    }
    qubit& operator+=(BitInteger const bit) noexcept
    { bit_ += bit; return *this; }
    qubit& operator-=(BitInteger const bit) noexcept
    { bit_ -= bit; return *this; }
    BitInteger operator-(qubit const& other) const noexcept
    { return bit_ - other.bit_; }
  }; // class qubit<StateInteger, BitInteger>

  template <typename StateInteger, typename BitInteger>
  inline constexpr bool operator==(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  { return static_cast<BitInteger>(qubit1) == static_cast<BitInteger>(qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline constexpr bool operator!=(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  { return not (qubit1 == qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline constexpr bool operator<(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  { return static_cast<BitInteger>(qubit1) < static_cast<BitInteger>(qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline constexpr bool operator>(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  { return qubit2 < qubit1; }

  template <typename StateInteger, typename BitInteger>
  inline constexpr bool operator<=(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  { return not (qubit1 > qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline constexpr bool operator>=(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  { return not (qubit1 < qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline constexpr qubit<StateInteger, BitInteger> operator+(
    ::ket::qubit<StateInteger, BitInteger> qubit, BitInteger const bit) noexcept
  { return qubit += bit; }

  template <typename StateInteger, typename BitInteger>
  inline constexpr qubit<StateInteger, BitInteger> operator+(
    BitInteger const bit, ::ket::qubit<StateInteger, BitInteger> const qubit) noexcept
  { return qubit + bit; }

  template <typename StateInteger, typename BitInteger>
  inline constexpr qubit<StateInteger, BitInteger> operator-(
    ::ket::qubit<StateInteger, BitInteger> qubit, BitInteger const bit) noexcept
  { return qubit -= bit; }

  template <typename Value, typename StateInteger, typename BitInteger>
  inline constexpr
  typename std::enable_if<std::is_integral<Value>::value, Value>::type
  operator<<(
    Value const value,
    ::ket::qubit<StateInteger, BitInteger> const qubit) noexcept
  { return value << static_cast<BitInteger>(qubit); }

  template <typename Value, typename StateInteger, typename BitInteger>
  inline constexpr
  typename std::enable_if<std::is_integral<Value>::value, Value>::type
  operator>>(
    Value const value,
    ::ket::qubit<StateInteger, BitInteger> const qubit) noexcept
  { return value >> static_cast<BitInteger>(qubit); }


  template <typename StateInteger, typename BitInteger>
  inline constexpr ::ket::qubit<StateInteger, BitInteger>
  make_qubit(BitInteger const bit) noexcept
  { return ::ket::qubit<StateInteger, BitInteger>(bit); }


  namespace qubit_literals
  {
    inline constexpr ::ket::qubit<std::uint64_t, unsigned int>
    operator"" _q(unsigned long long int const bit) noexcept
    { return ::ket::qubit<std::uint64_t, unsigned int>{static_cast<unsigned int>(bit)}; }

    inline constexpr ::ket::qubit<std::uint64_t, unsigned short int>
    operator"" _qs(unsigned long long int const bit) noexcept
    { return ::ket::qubit<std::uint64_t, unsigned short int>{static_cast<unsigned short>(bit)}; }

    inline constexpr ::ket::qubit<std::uint64_t, unsigned long int>
    operator"" _ql(unsigned long long int const bit) noexcept
    { return ::ket::qubit<std::uint64_t, unsigned long int>{static_cast<unsigned long>(bit)}; }

    inline constexpr ::ket::qubit<std::uint64_t, unsigned long long int>
    operator"" _qll(unsigned long long int const bit) noexcept
    { return ::ket::qubit<std::uint64_t, unsigned long long int>{bit}; }

    inline constexpr ::ket::qubit<std::uint64_t, std::uint8_t>
    operator"" _q8(unsigned long long int const bit) noexcept
    { return ::ket::qubit<std::uint64_t, std::uint8_t>{static_cast<std::uint8_t>(bit)}; }

    inline constexpr ::ket::qubit<std::uint64_t, std::uint16_t>
    operator"" _q16(unsigned long long int const bit) noexcept
    { return ::ket::qubit<std::uint64_t, std::uint16_t>{static_cast<std::uint16_t>(bit)}; }

    inline constexpr ::ket::qubit<std::uint64_t, std::uint32_t>
    operator"" _q32(unsigned long long int const bit) noexcept
    { return ::ket::qubit<std::uint64_t, std::uint32_t>{static_cast<std::uint32_t>(bit)}; }

    inline constexpr ::ket::qubit<std::uint64_t, std::uint64_t>
    operator"" _q64(unsigned long long int const bit) noexcept
    { return ::ket::qubit<std::uint64_t, std::uint64_t>{static_cast<std::uint64_t>(bit)}; }
  } // namespace qubit_literals
} // namespace ket


#endif // KET_QUBIT_HPP
