#ifndef KET_CONTROL_HPP
# define KET_CONTROL_HPP

# include <cstdint>

# include <ket/qubit.hpp>


namespace ket
{
  template <typename Qubit>
  class control
  {
    using qubit_type = Qubit;
    qubit_type qubit_;

   public:
    constexpr control() noexcept : qubit_() { }

    explicit constexpr control(Qubit const qubit) noexcept
      : qubit_(qubit)
    { }

    template <typename BitInteger_>
    explicit constexpr control(BitInteger_ const bit) noexcept
      : qubit_(bit)
    { }

    qubit_type& qubit() { return qubit_; }
    qubit_type const& qubit() const { return qubit_; }
  }; // class control<Qubit>

  template <typename Qubit>
  inline constexpr bool operator==(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    noexcept
  { return control_qubit1.qubit() == control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr bool operator!=(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    noexcept
  { return control_qubit1.qubit() != control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr bool operator<(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    noexcept
  { return control_qubit1.qubit() < control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr bool operator>(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    noexcept
  { return control_qubit1.qubit() > control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr bool operator<=(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    noexcept
  { return control_qubit1.qubit() <= control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr bool operator>=(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    noexcept
  { return control_qubit1.qubit() >= control_qubit2.qubit(); }


  template <typename Qubit>
  inline constexpr ::ket::control<Qubit> make_control(Qubit const qubit)
  { return ::ket::control<Qubit>{qubit}; }


  namespace control_literals
  {
    inline constexpr ::ket::control< ::ket::qubit<std::uint64_t, unsigned int> >
    operator"" _cq(unsigned long long int const bit) noexcept
    { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned int> >{static_cast<unsigned int>(bit)}; }

    inline constexpr ::ket::control< ::ket::qubit<std::uint64_t, unsigned short int> >
    operator"" _cqs(unsigned long long int const bit) noexcept
    { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned short int> >{static_cast<unsigned short>(bit)}; }

    inline constexpr ::ket::control< ::ket::qubit<std::uint64_t, unsigned long int> >
    operator"" _cql(unsigned long long int const bit) noexcept
    { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned long int> >{static_cast<unsigned long>(bit)}; }

    inline constexpr ::ket::control< ::ket::qubit<std::uint64_t, unsigned long long int> >
    operator"" _cqll(unsigned long long int const bit) noexcept
    { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned long long int> >{bit}; }

    inline constexpr ::ket::control< ::ket::qubit<std::uint64_t, std::uint8_t> >
    operator"" _cq8(unsigned long long int const bit) noexcept
    { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint8_t> >{static_cast<std::uint8_t>(bit)}; }

    inline constexpr ::ket::control< ::ket::qubit<std::uint64_t, std::uint16_t> >
    operator"" _cq16(unsigned long long int const bit) noexcept
    { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint16_t> >{static_cast<std::uint16_t>(bit)}; }

    inline constexpr ::ket::control< ::ket::qubit<std::uint64_t, std::uint32_t> >
    operator"" _cq32(unsigned long long int const bit) noexcept
    { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint32_t> >{static_cast<std::uint32_t>(bit)}; }

    inline constexpr ::ket::control< ::ket::qubit<std::uint64_t, std::uint64_t> >
    operator"" _cq64(unsigned long long int const bit) noexcept
    { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint64_t> >{static_cast<std::uint64_t>(bit)}; }
  } // namespace control_literals
} // namespace ket


#endif // KET_CONTROL_HPP
