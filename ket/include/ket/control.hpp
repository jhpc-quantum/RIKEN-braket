#ifndef KET_CONTROL_HPP
# define KET_CONTROL_HPP

# include <cstdint>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/meta/base.hpp>


namespace ket
{
  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value = ::ket::meta::base_v<Qubit> - 1u>
  class control
  {
    static_assert(Value < ::ket::meta::base_v<Qubit>, "Control value must be smaller than the quantum digit base");
    Qubit qubit_;

   public:
    using qubit_type = Qubit;
    using bit_integer_type = typename qubit_type::bit_integer_type;
    using state_integer_type = typename qubit_type::state_integer_type;

    constexpr control() noexcept : qubit_{} { }

    explicit constexpr control(Qubit const qubit) noexcept
      : qubit_{qubit}
    { }

    template <typename BitInteger_,
              std::enable_if_t<std::is_integral<BitInteger_>::value and std::is_unsigned<BitInteger_>::value, int> = 0>
    explicit constexpr control(BitInteger_ const bit) noexcept
      : qubit_{bit}
    { }

    static constexpr auto value() noexcept -> state_integer_type { return Value; }

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
  }; // class control<Qubit, Value>

  namespace meta
  {
    template <typename>
    struct is_control
      : std::false_type
    { };

    template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
    struct is_control< ::ket::control<Qubit, Value> >
      : std::true_type
    { };

    template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
    struct base< ::ket::control<Qubit, Value> > : ::ket::meta::base<Qubit>
    { };

    template <typename T>
    struct is_control_cvref
      : ::ket::meta::is_control<typename std::remove_cv<typename std::remove_reference<T>::type>::type>
    { };
  } // namespace meta

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator==(
    ::ket::control<Qubit, Value> const control_qubit1, ::ket::control<Qubit, Value> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() == control_qubit2.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator==(::ket::control<Qubit, Value> const control_qubit, Qubit const qubit) noexcept
  -> bool
  { return control_qubit.qubit() == qubit; }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator==(Qubit const qubit, ::ket::control<Qubit, Value> const control_qubit) noexcept
  -> bool
  { return qubit == control_qubit.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator!=(
    ::ket::control<Qubit, Value> const control_qubit1, ::ket::control<Qubit, Value> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() != control_qubit2.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator!=(::ket::control<Qubit, Value> const control_qubit, Qubit const qubit) noexcept
  -> bool
  { return control_qubit.qubit() != qubit; }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator!=(Qubit const qubit, ::ket::control<Qubit, Value> const control_qubit) noexcept
  -> bool
  { return qubit != control_qubit.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator<(
    ::ket::control<Qubit, Value> const control_qubit1, ::ket::control<Qubit, Value> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() < control_qubit2.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator<(::ket::control<Qubit, Value> const control_qubit, Qubit const qubit) noexcept
  -> bool
  { return control_qubit.qubit() < qubit; }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator<(Qubit const qubit, ::ket::control<Qubit, Value> const control_qubit) noexcept
  -> bool
  { return qubit < control_qubit.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator>(
    ::ket::control<Qubit, Value> const control_qubit1, ::ket::control<Qubit, Value> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() > control_qubit2.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator>(::ket::control<Qubit, Value> const control_qubit, Qubit const qubit) noexcept -> bool
  { return control_qubit.qubit() > qubit; }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator>(Qubit const qubit, ::ket::control<Qubit, Value> const control_qubit) noexcept -> bool
  { return qubit > control_qubit.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator<=(
    ::ket::control<Qubit, Value> const control_qubit1, ::ket::control<Qubit, Value> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() <= control_qubit2.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator<=(::ket::control<Qubit, Value> const control_qubit, Qubit const qubit) noexcept -> bool
  { return control_qubit.qubit() <= qubit; }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator<=(Qubit const qubit, ::ket::control<Qubit, Value> const control_qubit) noexcept -> bool
  { return qubit <= control_qubit.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator>=(
    ::ket::control<Qubit, Value> const control_qubit1, ::ket::control<Qubit, Value> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() >= control_qubit2.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator>=(::ket::control<Qubit, Value> const control_qubit, Qubit const qubit) noexcept -> bool
  { return control_qubit.qubit() >= qubit; }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator>=(Qubit const qubit, ::ket::control<Qubit, Value> const control_qubit) noexcept -> bool
  { return qubit >= control_qubit.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value, typename Integer>
  inline constexpr auto operator+(::ket::control<Qubit, Value> control_qubit, Integer const bit) noexcept -> ::ket::control<Qubit, Value>
  { return control_qubit += bit; }

  template <typename Integer, typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator+(Integer const bit, ::ket::control<Qubit, Value> const control_qubit) noexcept -> ::ket::control<Qubit, Value>
  { return control_qubit + bit; }

  template <typename Qubit1, ::ket::meta::state_integer_t<Qubit1> Value1,
            typename Qubit2, ::ket::meta::state_integer_t<Qubit2> Value2>
  inline auto operator+(::ket::control<Qubit1, Value1> const control_qubit1, ::ket::control<Qubit2, Value2> const control_qubit2) -> ::ket::control<Qubit1, Value1>
    = delete;

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value, typename Integer>
  inline constexpr auto operator-(::ket::control<Qubit, Value> control_qubit, Integer const bit) noexcept -> ::ket::control<Qubit, Value>
  { return control_qubit -= bit; }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator-(
    ::ket::control<Qubit, Value> const control_qubit1, ::ket::control<Qubit, Value> const control_qubit2) noexcept
  -> decltype(control_qubit1.qubit() - control_qubit2.qubit())
  { return control_qubit1.qubit() - control_qubit2.qubit(); }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator-(::ket::control<Qubit, Value> const control_qubit, Qubit const qubit) noexcept
  -> decltype(control_qubit.qubit() - qubit)
  { return control_qubit.qubit() - qubit; }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto operator-(Qubit const qubit, ::ket::control<Qubit, Value> const control_qubit) noexcept
  -> decltype(qubit - control_qubit.qubit())
  { return qubit - control_qubit.qubit(); }

  template <typename UnsignedInteger, typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr std::enable_if_t<std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value, UnsignedInteger>
  operator<<(UnsignedInteger const unsigned_integer, ::ket::control<Qubit, Value> const control_qubit) noexcept
  { return unsigned_integer << control_qubit.qubit(); }

  template <typename UnsignedInteger, typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr std::enable_if_t<std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value, UnsignedInteger>
  operator>>(UnsignedInteger const unsigned_integer, ::ket::control<Qubit, Value> const control_qubit) noexcept
  { return unsigned_integer >> control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto make_control(Qubit const qubit) -> ::ket::control<Qubit>
  { return ::ket::control<Qubit>{qubit}; }

  template <std::uintmax_t Value, typename Qubit>
  inline constexpr auto make_control(Qubit const qubit)
  -> ::ket::control<Qubit, static_cast< ::ket::meta::state_integer_t<Qubit> >(Value)>
  {
    static_assert(Value < ::ket::meta::base_v<Qubit>, "Control value must be smaller than the quantum digit base");
    return ::ket::control<Qubit, static_cast< ::ket::meta::state_integer_t<Qubit> >(Value)>{qubit};
  }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto control_value(::ket::control<Qubit, Value> const&) noexcept
  -> ::ket::meta::state_integer_t<Qubit>
  { return Value; }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto remove_control(::ket::control<Qubit, Value> const control_qubit) -> Qubit
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

      inline constexpr auto operator"" _pcq(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned int>, 1u>
      { return operator"" _cq(bit); }

      inline constexpr auto operator"" _ncq(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned int>, 0u>
      { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned int>, 0u>{static_cast<unsigned int>(bit)}; }

      inline constexpr auto operator"" _pcqs(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned short int>, 1u>
      { return operator"" _cqs(bit); }

      inline constexpr auto operator"" _ncqs(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned short int>, 0u>
      { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned short int>, 0u>{static_cast<unsigned short int>(bit)}; }

      inline constexpr auto operator"" _pcql(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned long int>, 1u>
      { return operator"" _cql(bit); }

      inline constexpr auto operator"" _ncql(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned long int>, 0u>
      { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned long int>, 0u>{static_cast<unsigned long int>(bit)}; }

      inline constexpr auto operator"" _pcqll(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned long long int>, 1u>
      { return operator"" _cqll(bit); }

      inline constexpr auto operator"" _ncqll(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, unsigned long long int>, 0u>
      { return ::ket::control< ::ket::qubit<std::uint64_t, unsigned long long int>, 0u>{bit}; }

      inline constexpr auto operator"" _pcq8(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint8_t>, 1u>
      { return operator"" _cq8(bit); }

      inline constexpr auto operator"" _ncq8(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint8_t>, 0u>
      { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint8_t>, 0u>{static_cast<std::uint8_t>(bit)}; }

      inline constexpr auto operator"" _pcq16(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint16_t>, 1u>
      { return operator"" _cq16(bit); }

      inline constexpr auto operator"" _ncq16(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint16_t>, 0u>
      { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint16_t>, 0u>{static_cast<std::uint16_t>(bit)}; }

      inline constexpr auto operator"" _pcq32(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint32_t>, 1u>
      { return operator"" _cq32(bit); }

      inline constexpr auto operator"" _ncq32(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint32_t>, 0u>
      { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint32_t>, 0u>{static_cast<std::uint32_t>(bit)}; }

      inline constexpr auto operator"" _pcq64(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint64_t>, 1u>
      { return operator"" _cq64(bit); }

      inline constexpr auto operator"" _ncq64(unsigned long long int const bit) noexcept
      -> ::ket::control< ::ket::qubit<std::uint64_t, std::uint64_t>, 0u>
      { return ::ket::control< ::ket::qubit<std::uint64_t, std::uint64_t>, 0u>{static_cast<std::uint64_t>(bit)}; }
    } // namespace control_literals
  } // namespace literals

  namespace [[deprecated]] control_literals
  { using namespace ::ket::literals::control_literals; }
} // namespace ket


#endif // KET_CONTROL_HPP
