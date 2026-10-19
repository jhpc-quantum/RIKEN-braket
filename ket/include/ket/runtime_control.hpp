#ifndef KET_RUNTIME_CONTROL_HPP
# define KET_RUNTIME_CONTROL_HPP

# include <cstdint>
# include <stdexcept>
# include <type_traits>

# include <ket/control.hpp>

namespace ket
{
  template <typename Qubit>
  class runtime_control
  {
    Qubit qubit_;
    ::ket::meta::state_integer_t<Qubit> value_;

   public:
    using qubit_type = Qubit;
    using bit_integer_type = typename qubit_type::bit_integer_type;
    using state_integer_type = typename qubit_type::state_integer_type;

    constexpr runtime_control() noexcept
      : qubit_{}, value_{::ket::meta::base_v<Qubit> - 1u} { }

    explicit constexpr runtime_control(Qubit const qubit) noexcept
      : qubit_{qubit}, value_{::ket::meta::base_v<Qubit> - 1u} { }

    template <typename Integer, std::enable_if_t<std::is_integral<Integer>::value, int> = 0>
    constexpr runtime_control(Qubit const qubit, Integer const value)
      : qubit_{qubit}, value_{static_cast<state_integer_type>(value)}
    {
      // Validate the original integer, before narrowing can hide an invalid value.
      if ((std::is_signed<Integer>::value and value < Integer{0})
          or static_cast<std::uintmax_t>(value) >= ::ket::meta::base_v<Qubit>)
        throw std::out_of_range{"Control value must be within the quantum digit base"};
    }

    template <state_integer_type Value>
    explicit constexpr runtime_control(::ket::control<Qubit, Value> const control_qubit) noexcept
      : qubit_{control_qubit.qubit()}, value_{Value} { }

    constexpr auto qubit() noexcept -> qubit_type& { return qubit_; }
    constexpr auto qubit() const noexcept -> qubit_type const& { return qubit_; }
    constexpr auto value() const noexcept -> state_integer_type { return value_; }

    constexpr auto operator++() noexcept -> runtime_control& { ++qubit_; return *this; }
    constexpr auto operator++(int) noexcept -> runtime_control { auto result = *this; ++(*this); return result; }
    constexpr auto operator--() noexcept -> runtime_control& { --qubit_; return *this; }
    constexpr auto operator--(int) noexcept -> runtime_control { auto result = *this; --(*this); return result; }

    template <typename Integer>
    constexpr auto operator+=(Integer const bit) noexcept -> runtime_control&
    { qubit_ += bit; return *this; }

    template <typename Integer>
    constexpr auto operator-=(Integer const bit) noexcept -> runtime_control&
    { qubit_ -= bit; return *this; }
  };

  namespace meta
  {
    template <typename Qubit>
    struct is_control< ::ket::runtime_control<Qubit> > : std::true_type
    { };

    template <typename Qubit>
    struct base< ::ket::runtime_control<Qubit> > : ::ket::meta::base<Qubit>
    { };
  }

  // Like existing control operators, comparisons and distances refer to wire
  // positions. Compare control_value(...) separately to compare conditions.
  template <typename Qubit>
  inline constexpr auto operator==(
    ::ket::runtime_control<Qubit> const control_qubit1, ::ket::runtime_control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() == control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator==(::ket::runtime_control<Qubit> const control_qubit, Qubit const qubit) noexcept
  -> bool
  { return control_qubit.qubit() == qubit; }

  template <typename Qubit>
  inline constexpr auto operator==(Qubit const qubit, ::ket::runtime_control<Qubit> const control_qubit) noexcept
  -> bool
  { return qubit == control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator!=(
    ::ket::runtime_control<Qubit> const control_qubit1, ::ket::runtime_control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() != control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator!=(::ket::runtime_control<Qubit> const control_qubit, Qubit const qubit) noexcept
  -> bool
  { return control_qubit.qubit() != qubit; }

  template <typename Qubit>
  inline constexpr auto operator!=(Qubit const qubit, ::ket::runtime_control<Qubit> const control_qubit) noexcept
  -> bool
  { return qubit != control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator<(
    ::ket::runtime_control<Qubit> const control_qubit1, ::ket::runtime_control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() < control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator<(::ket::runtime_control<Qubit> const control_qubit, Qubit const qubit) noexcept
  -> bool
  { return control_qubit.qubit() < qubit; }

  template <typename Qubit>
  inline constexpr auto operator<(Qubit const qubit, ::ket::runtime_control<Qubit> const control_qubit) noexcept
  -> bool
  { return qubit < control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator>(
    ::ket::runtime_control<Qubit> const control_qubit1, ::ket::runtime_control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() > control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator>(::ket::runtime_control<Qubit> const control_qubit, Qubit const qubit) noexcept -> bool
  { return control_qubit.qubit() > qubit; }

  template <typename Qubit>
  inline constexpr auto operator>(Qubit const qubit, ::ket::runtime_control<Qubit> const control_qubit) noexcept -> bool
  { return qubit > control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator<=(
    ::ket::runtime_control<Qubit> const control_qubit1, ::ket::runtime_control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() <= control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator<=(::ket::runtime_control<Qubit> const control_qubit, Qubit const qubit) noexcept -> bool
  { return control_qubit.qubit() <= qubit; }

  template <typename Qubit>
  inline constexpr auto operator<=(Qubit const qubit, ::ket::runtime_control<Qubit> const control_qubit) noexcept -> bool
  { return qubit <= control_qubit.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator>=(
    ::ket::runtime_control<Qubit> const control_qubit1, ::ket::runtime_control<Qubit> const control_qubit2) noexcept
  -> bool
  { return control_qubit1.qubit() >= control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator>=(::ket::runtime_control<Qubit> const control_qubit, Qubit const qubit) noexcept -> bool
  { return control_qubit.qubit() >= qubit; }

  template <typename Qubit>
  inline constexpr auto operator>=(Qubit const qubit, ::ket::runtime_control<Qubit> const control_qubit) noexcept -> bool
  { return qubit >= control_qubit.qubit(); }

  template <typename Qubit, typename Integer>
  inline constexpr auto operator+(::ket::runtime_control<Qubit> control_qubit, Integer const bit) noexcept -> ::ket::runtime_control<Qubit>
  { return control_qubit += bit; }

  template <typename Integer, typename Qubit>
  inline constexpr auto operator+(Integer const bit, ::ket::runtime_control<Qubit> const control_qubit) noexcept -> ::ket::runtime_control<Qubit>
  { return control_qubit + bit; }

  template <typename Qubit1, typename Qubit2>
  inline auto operator+(::ket::runtime_control<Qubit1> const control_qubit1, ::ket::runtime_control<Qubit2> const control_qubit2) -> ::ket::runtime_control<Qubit1>
    = delete;

  template <typename Qubit, typename Integer>
  inline constexpr auto operator-(::ket::runtime_control<Qubit> control_qubit, Integer const bit) noexcept -> ::ket::runtime_control<Qubit>
  { return control_qubit -= bit; }

  template <typename Qubit>
  inline constexpr auto operator-(
    ::ket::runtime_control<Qubit> const control_qubit1, ::ket::runtime_control<Qubit> const control_qubit2) noexcept
  -> decltype(control_qubit1.qubit() - control_qubit2.qubit())
  { return control_qubit1.qubit() - control_qubit2.qubit(); }

  template <typename Qubit>
  inline constexpr auto operator-(::ket::runtime_control<Qubit> const control_qubit, Qubit const qubit) noexcept
  -> decltype(control_qubit.qubit() - qubit)
  { return control_qubit.qubit() - qubit; }

  template <typename StateInteger, typename BitInteger>
  inline constexpr auto operator-(
    ::ket::qubit<StateInteger, BitInteger> const qubit,
    ::ket::runtime_control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit) noexcept
  -> decltype(qubit - control_qubit.qubit())
  { return qubit - control_qubit.qubit(); }

  template <typename UnsignedInteger, typename Qubit>
  inline constexpr std::enable_if_t<std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value, UnsignedInteger>
  operator<<(UnsignedInteger const unsigned_integer, ::ket::runtime_control<Qubit> const control_qubit) noexcept
  { return unsigned_integer << control_qubit.qubit(); }

  template <typename UnsignedInteger, typename Qubit>
  inline constexpr std::enable_if_t<std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value, UnsignedInteger>
  operator>>(UnsignedInteger const unsigned_integer, ::ket::runtime_control<Qubit> const control_qubit) noexcept
  { return unsigned_integer >> control_qubit.qubit(); }


  template <typename Qubit>
  inline constexpr auto control_value(::ket::runtime_control<Qubit> const& control_qubit) noexcept
  -> ::ket::meta::state_integer_t<Qubit>
  { return control_qubit.value(); }

  template <typename Qubit>
  inline constexpr auto remove_control(::ket::runtime_control<Qubit> const control_qubit) noexcept -> Qubit
  { return control_qubit.qubit(); }

  template <typename Qubit, typename Integer>
  inline constexpr auto make_runtime_control(Qubit const qubit, Integer const value)
  -> ::ket::runtime_control<Qubit>
  { return ::ket::runtime_control<Qubit>{qubit, value}; }

  template <typename Qubit, ::ket::meta::state_integer_t<Qubit> Value>
  inline constexpr auto make_runtime_control(::ket::control<Qubit, Value> const control_qubit) noexcept
  -> ::ket::runtime_control<Qubit>
  { return ::ket::runtime_control<Qubit>{control_qubit}; }
}

#endif // KET_RUNTIME_CONTROL_HPP
