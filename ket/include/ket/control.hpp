#ifndef KET_CONTROL_HPP
# define KET_CONTROL_HPP

# include <cstdint>
# include <type_traits>

# include <ket/qubit.hpp>


namespace ket
{
  template <typename Qubit>
  class control
  {
    using qubit_type = Qubit;
    qubit_type qubit_;

   public:
    using bit_integer_type = typename qubit_type::bit_integer_type;
    using state_integer_type = typename qubit_type::state_integer_type;

    constexpr control() noexcept : qubit_{} { }

    explicit constexpr control(Qubit const qubit) noexcept
      : qubit_{qubit}
    { }

    template <typename BitInteger_>
    explicit constexpr control(BitInteger_ const bit) noexcept
      : qubit_{bit}
    { static_assert(std::is_integral<BitInteger_>::value and std::is_unsigned<BitInteger_>::value, "BitInteger_ should be an unsigned integral type"); }

    constexpr auto qubit() -> qubit_type& { return qubit_; }
    constexpr auto qubit() const -> qubit_type const& { return qubit_; }

    constexpr auto operator++() noexcept -> control& { ++qubit_; return *this; }
    constexpr auto operator++(int) noexcept -> control { auto result = *this; ++(*this); return result; }
    constexpr auto operator--() noexcept -> control& { --qubit_; return *this; }
    constexpr auto operator--(int) noexcept -> control { auto result = *this; --(*this); return result; }

    template <typename Integer>
    constexpr auto operator+=(Integer const bit) noexcept -> control&
    {
      static_assert(std::is_integral<Integer>::value, "Integer should be an integral type");
      qubit_ += bit;
      return *this;
    }

    template <typename Integer>
    constexpr auto operator-=(Integer const bit) noexcept -> control&
    {
      static_assert(std::is_integral<Integer>::value, "Integer should be an integral type");
      qubit_ -= bit;
      return *this;
    }
  }; // class control<Qubit>

  template <typename Qubit>
  inline constexpr auto operator==(
    ::ket::control<Qubit> const control_qubit1, ::ket::control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() == control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator==(::ket::control<Qubit> const control_qubit, Qubit const qubit) noexcept
  -> bool
  { return control_qubit.qubit() == qubit; }

  template <typename Qubit>
  inline constexpr auto operator==(Qubit const qubit, ::ket::control<Qubit> const control_qubit) noexcept
  -> bool
  { return qubit == control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator!=(
    ::ket::control<Qubit> const control_qubit1, ::ket::control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() != control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator!=(::ket::control<Qubit> const control_qubit, Qubit const qubit) noexcept
  -> bool
  { return control_qubit.qubit() != qubit; }

  template <typename Qubit>
  inline constexpr auto operator!=(Qubit const qubit, ::ket::control<Qubit> const control_qubit) noexcept
  -> bool
  { return qubit != control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator<(
    ::ket::control<Qubit> const control_qubit1, ::ket::control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() < control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator<(::ket::control<Qubit> const control_qubit, Qubit const qubit) noexcept
  -> bool
  { return control_qubit.qubit() < qubit; }

  template <typename Qubit>
  inline constexpr auto operator<(Qubit const qubit, ::ket::control<Qubit> const control_qubit) noexcept
  -> bool
  { return qubit < control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator>(
    ::ket::control<Qubit> const control_qubit1, ::ket::control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() > control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator>(::ket::control<Qubit> const control_qubit, Qubit const qubit) noexcept -> bool
  { return control_qubit.qubit() > qubit; }

  template <typename Qubit>
  inline constexpr auto operator>(Qubit const qubit, ::ket::control<Qubit> const control_qubit) noexcept -> bool
  { return qubit > control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator<=(
    ::ket::control<Qubit> const control_qubit1, ::ket::control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() <= control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator<=(::ket::control<Qubit> const control_qubit, Qubit const qubit) noexcept -> bool
  { return control_qubit.qubit() <= qubit; }

  template <typename Qubit>
  inline constexpr auto operator<=(Qubit const qubit, ::ket::control<Qubit> const control_qubit) noexcept -> bool
  { return qubit <= control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator>=(
    ::ket::control<Qubit> const control_qubit1, ::ket::control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() >= control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator>=(::ket::control<Qubit> const control_qubit, Qubit const qubit) noexcept -> bool
  { return control_qubit.qubit() >= qubit; }

  template <typename Qubit>
  inline constexpr auto operator>=(Qubit const qubit, ::ket::control<Qubit> const control_qubit) noexcept -> bool
  { return qubit >= control_qubit.qubit(); }

  template <typename Qubit, typename Integer>
  inline constexpr auto operator+(::ket::control<Qubit> control_qubit, Integer const bit) noexcept -> ::ket::control<Qubit>
  { return control_qubit += bit; }

  template <typename Integer, typename Qubit>
  inline constexpr auto operator+(Integer const bit, ::ket::control<Qubit> const control_qubit) noexcept -> ::ket::control<Qubit>
  { return control_qubit + bit; }

  template <typename Qubit1, typename Qubit2>
  inline auto operator+(::ket::control<Qubit1> const control_qubit1, ::ket::control<Qubit2> const control_qubit2) -> ::ket::control<Qubit1>
    = delete;

  template <typename Qubit, typename Integer>
  inline constexpr auto operator-(::ket::control<Qubit> control_qubit, Integer const bit) noexcept -> ::ket::control<Qubit>
  { return control_qubit -= bit; }

  template <typename Qubit>
  inline constexpr auto operator-(
    ::ket::control<Qubit> const control_qubit1, ::ket::control<Qubit> const control_qubit2) noexcept
  -> decltype(control_qubit1.qubit() - control_qubit2.qubit())
  { return control_qubit1.qubit() - control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator-(::ket::control<Qubit> const control_qubit, Qubit const qubit) noexcept
  -> decltype(control_qubit.qubit() - qubit)
  { return control_qubit.qubit() - qubit; }

  template <typename Qubit>
  inline constexpr auto operator-(Qubit const qubit, ::ket::control<Qubit> const control_qubit) noexcept
  -> decltype(qubit - control_qubit.qubit())
  { return qubit - control_qubit.qubit(); }

  template <typename UnsignedInteger, typename Qubit>
  inline constexpr std::enable_if_t<std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value, UnsignedInteger>
  operator<<(UnsignedInteger const unsigned_integer, ::ket::control<Qubit> const control_qubit) noexcept
  { return unsigned_integer << control_qubit.qubit(); }

  template <typename UnsignedInteger, typename Qubit>
  inline constexpr std::enable_if_t<std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value, UnsignedInteger>
  operator>>(UnsignedInteger const unsigned_integer, ::ket::control<Qubit> const control_qubit) noexcept
  { return unsigned_integer >> control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto make_control(Qubit const qubit) -> ::ket::control<Qubit>
  { return ::ket::control<Qubit>{qubit}; }

  template <typename Qubit>
  inline constexpr auto remove_control(::ket::control<Qubit> const control_qubit) -> Qubit
  { return control_qubit.qubit(); }

  namespace literals
  {
    inline namespace control_literals
    {
      inline constexpr auto operator"" _cq(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned int> >
      { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned int> >{static_cast<unsigned int>(bit)}; }

      inline constexpr auto operator"" _cqs(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned short int> >
      { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned short int> >{static_cast<unsigned short>(bit)}; }

      inline constexpr auto operator"" _cql(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned long int> >
      { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned long int> >{static_cast<unsigned long>(bit)}; }

      inline constexpr auto operator"" _cqll(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned long long int> >
      { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned long long int> >{bit}; }

      inline constexpr auto operator"" _cq8(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint8_t> >
      { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint8_t> >{static_cast<std::uint8_t>(bit)}; }

      inline constexpr auto operator"" _cq16(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint16_t> >
      { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint16_t> >{static_cast<std::uint16_t>(bit)}; }

      inline constexpr auto operator"" _cq32(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint32_t> >
      { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint32_t> >{static_cast<std::uint32_t>(bit)}; }

      inline constexpr auto operator"" _cq64(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint64_t> >
      { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint64_t> >{static_cast<std::uint64_t>(bit)}; }
    } // namespace control_literals
  } // namespace literals

  namespace [[deprecated]] control_literals
  { using namespace ::ket::literals::control_literals; }
} // namespace ket


#endif // KET_CONTROL_HPP
