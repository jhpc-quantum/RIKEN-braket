#ifndef KET_QUBIT_HPP
# define KET_QUBIT_HPP

# include <cstdint>
# include <type_traits>


namespace ket
{
  template <typename StateInteger = std::uint64_t, typename BitInteger = unsigned int>
  class qubit
  {
    static_assert(std::is_integral<StateInteger>::value and std::is_unsigned<StateInteger>::value, "StateInteger should be an unsigned integral type");
    static_assert(std::is_integral<BitInteger>::value and std::is_unsigned<BitInteger>::value, "BitInteger should be an unsigned integral type");

    BitInteger bit_;

   public:
    using bit_integer_type = BitInteger;
    using state_integer_type = StateInteger;

    constexpr qubit() noexcept : bit_{0u} { }

    template <typename BitInteger_>
    explicit constexpr qubit(BitInteger_ const bit) noexcept
      : bit_{static_cast<BitInteger>(bit)}
    { static_assert(std::is_integral<BitInteger_>::value and std::is_unsigned<BitInteger_>::value, "BitInteger_ should be an unsigned integral type"); }

    explicit constexpr operator BitInteger() const { return bit_; }

    constexpr auto operator++() noexcept -> qubit& { ++bit_; return *this; }
    constexpr auto operator++(int) noexcept -> qubit { auto result = *this; ++(*this); return result; }
    constexpr auto operator--() noexcept -> qubit& { --bit_; return *this; }
    constexpr auto operator--(int) noexcept -> qubit { auto result = *this; --(*this); return result; }

    template <typename Integer>
    constexpr auto operator+=(Integer const bit) noexcept -> qubit&
    {
      static_assert(std::is_integral<Integer>::value, "Integer should be an integral type");
      bit_ += static_cast<BitInteger>(bit);
      return *this;
    }

    template <typename Integer>
    constexpr auto operator-=(Integer const bit) noexcept -> qubit&
    {
      static_assert(std::is_integral<Integer>::value, "Integer should be an integral type");
      bit_ -= static_cast<BitInteger>(bit);
      return *this;
    }

    constexpr auto operator-(qubit const& other) const noexcept -> BitInteger { return bit_ - other.bit_; }
  }; // class qubit<StateInteger, BitInteger>

  template <typename StateInteger, typename BitInteger>
  inline constexpr auto operator==(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  -> bool
  { return static_cast<BitInteger>(qubit1) == static_cast<BitInteger>(qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline constexpr auto operator!=(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  -> bool
  { return not (qubit1 == qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline constexpr auto operator<(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  -> bool
  { return static_cast<BitInteger>(qubit1) < static_cast<BitInteger>(qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline constexpr auto operator>(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  -> bool
  { return qubit2 < qubit1; }

  template <typename StateInteger, typename BitInteger>
  inline constexpr auto operator<=(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  -> bool
  { return not (qubit1 > qubit2); }

  template <typename StateInteger, typename BitInteger>
  inline constexpr auto operator>=(
    ::ket::qubit<StateInteger, BitInteger> const qubit1,
    ::ket::qubit<StateInteger, BitInteger> const qubit2) noexcept
  -> bool
  { return not (qubit1 < qubit2); }

  template <typename StateInteger, typename BitInteger, typename Integer>
  inline constexpr auto operator+(
    ::ket::qubit<StateInteger, BitInteger> qubit, Integer const bit) noexcept
  -> ::ket::qubit<StateInteger, BitInteger>
  { return qubit += bit; }

  template <typename Integer, typename StateInteger, typename BitInteger>
  inline constexpr auto operator+(
    Integer const bit, ::ket::qubit<StateInteger, BitInteger> const qubit) noexcept
  -> ::ket::qubit<StateInteger, BitInteger>
  { return qubit + bit; }

  template <typename StateInteger, typename BitInteger, typename Integer>
  inline constexpr auto operator-(
    ::ket::qubit<StateInteger, BitInteger> qubit, Integer const bit) noexcept
  -> ::ket::qubit<StateInteger, BitInteger>
  { return qubit -= bit; }

  template <typename UnsignedInteger, typename StateInteger, typename BitInteger>
  inline constexpr std::enable_if_t<std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value, UnsignedInteger>
  operator<<(UnsignedInteger const unsigned_integer, ::ket::qubit<StateInteger, BitInteger> const qubit) noexcept
  { return unsigned_integer << static_cast<BitInteger>(qubit); }

  template <typename UnsignedInteger, typename StateInteger, typename BitInteger>
  inline constexpr std::enable_if_t<std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value, UnsignedInteger>
  operator>>(UnsignedInteger const unsigned_integer, ::ket::qubit<StateInteger, BitInteger> const qubit) noexcept
  { return unsigned_integer >> static_cast<BitInteger>(qubit); }

  template <typename StateInteger, typename BitInteger>
  inline constexpr auto make_qubit(BitInteger const bit) noexcept -> ::ket::qubit<StateInteger, BitInteger>
  { return ::ket::qubit<StateInteger, BitInteger>(bit); }

  template <typename StateInteger, typename BitInteger>
  inline constexpr auto remove_control(::ket::qubit<StateInteger, BitInteger> const qubit) -> ::ket::qubit<StateInteger, BitInteger>
  { return qubit; }

  namespace literals
  {
    inline namespace qubit_literals
    {
      inline constexpr auto operator"" _q(unsigned long long int const bit) noexcept
      -> ::ket::qubit<std::uint64_t, unsigned int>
      { return ::ket::qubit<std::uint64_t, unsigned int>{static_cast<unsigned int>(bit)}; }

      inline constexpr auto operator"" _qs(unsigned long long int const bit) noexcept
      -> ::ket::qubit<std::uint64_t, unsigned short int>
      { return ::ket::qubit<std::uint64_t, unsigned short int>{static_cast<unsigned short>(bit)}; }

      inline constexpr auto operator"" _ql(unsigned long long int const bit) noexcept
      -> ::ket::qubit<std::uint64_t, unsigned long int>
      { return ::ket::qubit<std::uint64_t, unsigned long int>{static_cast<unsigned long>(bit)}; }

      inline constexpr auto operator"" _qll(unsigned long long int const bit) noexcept
      -> ::ket::qubit<std::uint64_t, unsigned long long int>
      { return ::ket::qubit<std::uint64_t, unsigned long long int>{bit}; }

      inline constexpr auto operator"" _q8(unsigned long long int const bit) noexcept
      -> ::ket::qubit<std::uint64_t, std::uint8_t>
      { return ::ket::qubit<std::uint64_t, std::uint8_t>{static_cast<std::uint8_t>(bit)}; }

      inline constexpr auto operator"" _q16(unsigned long long int const bit) noexcept
      -> ::ket::qubit<std::uint64_t, std::uint16_t>
      { return ::ket::qubit<std::uint64_t, std::uint16_t>{static_cast<std::uint16_t>(bit)}; }

      inline constexpr auto operator"" _q32(unsigned long long int const bit) noexcept
      -> ::ket::qubit<std::uint64_t, std::uint32_t>
      { return ::ket::qubit<std::uint64_t, std::uint32_t>{static_cast<std::uint32_t>(bit)}; }

      inline constexpr auto operator"" _q64(unsigned long long int const bit) noexcept
      -> ::ket::qubit<std::uint64_t, std::uint64_t>
      { return ::ket::qubit<std::uint64_t, std::uint64_t>{static_cast<std::uint64_t>(bit)}; }
    } // namespace qubit_literals
  } // namespace literals

  namespace [[deprecated]] qubit_literals
  { using namespace ::ket::literals::qubit_literals; }
} // namespace ket


#endif // KET_QUBIT_HPP
