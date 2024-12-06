#ifndef KET_MPI_PERMUTATED_HPP
# define KET_MPI_PERMUTATED_HPP

# include <cstdint>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/meta/state_integer_of.hpp>


namespace ket
{
  namespace mpi
  {
    template <typename Qubit>
    class permutated
    {
      using qubit_type = Qubit;
      qubit_type qubit_;

     public:
      using bit_integer_type = typename qubit_type::bit_integer_type;
      using state_integer_type = typename qubit_type::state_integer_type;

      constexpr permutated() noexcept : qubit_{} { }

      explicit constexpr permutated(Qubit const qubit) noexcept
        : qubit_{qubit}
      { }

      template <typename BitInteger_>
      explicit constexpr permutated(BitInteger_ const bit) noexcept
        : qubit_{bit}
      { static_assert(std::is_integral<BitInteger_>::value and std::is_unsigned<BitInteger_>::value, "BitInteger_ should be an unsigned integral type"); }

      constexpr auto qubit() -> qubit_type& { return qubit_; }
      constexpr auto qubit() const -> qubit_type const& { return qubit_; }

      constexpr auto operator++() noexcept -> permutated& { ++qubit_; return *this; }
      constexpr auto operator++(int) noexcept -> permutated { auto result = *this; ++(*this); return result; }
      constexpr auto operator--() noexcept -> permutated& { --qubit_; return *this; }
      constexpr auto operator--(int) noexcept -> permutated { auto result = *this; --(*this); return result; }

      template <typename Integer>
      constexpr auto operator+=(Integer const bit) noexcept -> permutated&
      {
        static_assert(std::is_integral<Integer>::value, "Integer should be an integral type");
        qubit_ += bit;
        return *this;
      }

      template <typename Integer>
      constexpr auto operator-=(Integer const bit) noexcept -> permutated&
      {
        static_assert(std::is_integral<Integer>::value, "Integer should be an integral type");
        qubit_ -= bit;
        return *this;
      }
    }; // class permutated<Qubit>

    template <typename Qubit1, typename Qubit2>
    inline constexpr auto operator==(
      ::ket::mpi::permutated<Qubit1> const permutated_qubit1, ::ket::mpi::permutated<Qubit2> const permutated_qubit2) noexcept
    -> bool
    {
      static_assert(
        std::is_same<::ket::meta::bit_integer_t<Qubit1>, ::ket::meta::bit_integer_t<Qubit2>>::value,
        "BitInteger's of Qubit1 and Qubit2 should be the same");
      static_assert(
        std::is_same<::ket::meta::state_integer_t<Qubit1>, ::ket::meta::state_integer_t<Qubit2>>::value,
        "StateInteger's of Qubit1 and Qubit2 should be the same");
      return permutated_qubit1.qubit() == permutated_qubit2.qubit();
    }

    template <typename Qubit1, typename Qubit2>
    inline constexpr auto operator!=(
      ::ket::mpi::permutated<Qubit1> const permutated_qubit1, ::ket::mpi::permutated<Qubit2> const permutated_qubit2) noexcept
    -> bool
    {
      static_assert(
        std::is_same<::ket::meta::bit_integer_t<Qubit1>, ::ket::meta::bit_integer_t<Qubit2>>::value,
        "BitInteger's of Qubit1 and Qubit2 should be the same");
      static_assert(
        std::is_same<::ket::meta::state_integer_t<Qubit1>, ::ket::meta::state_integer_t<Qubit2>>::value,
        "StateInteger's of Qubit1 and Qubit2 should be the same");
      return permutated_qubit1.qubit() != permutated_qubit2.qubit();
    }

    template <typename Qubit1, typename Qubit2>
    inline constexpr auto operator<(
      ::ket::mpi::permutated<Qubit1> const permutated_qubit1, ::ket::mpi::permutated<Qubit2> const permutated_qubit2) noexcept
    -> bool
    {
      static_assert(
        std::is_same<::ket::meta::bit_integer_t<Qubit1>, ::ket::meta::bit_integer_t<Qubit2>>::value,
        "BitInteger's of Qubit1 and Qubit2 should be the same");
      static_assert(
        std::is_same<::ket::meta::state_integer_t<Qubit1>, ::ket::meta::state_integer_t<Qubit2>>::value,
        "StateInteger's of Qubit1 and Qubit2 should be the same");
      return permutated_qubit1.qubit() < permutated_qubit2.qubit();
    }

    template <typename Qubit1, typename Qubit2>
    inline constexpr auto operator>(
      ::ket::mpi::permutated<Qubit1> const permutated_qubit1, ::ket::mpi::permutated<Qubit2> const permutated_qubit2) noexcept
    -> bool
    {
      static_assert(
        std::is_same<::ket::meta::bit_integer_t<Qubit1>, ::ket::meta::bit_integer_t<Qubit2>>::value,
        "BitInteger's of Qubit1 and Qubit2 should be the same");
      static_assert(
        std::is_same<::ket::meta::state_integer_t<Qubit1>, ::ket::meta::state_integer_t<Qubit2>>::value,
        "StateInteger's of Qubit1 and Qubit2 should be the same");
      return permutated_qubit1.qubit() > permutated_qubit2.qubit();
    }

    template <typename Qubit1, typename Qubit2>
    inline constexpr auto operator<=(
      ::ket::mpi::permutated<Qubit1> const permutated_qubit1, ::ket::mpi::permutated<Qubit2> const permutated_qubit2) noexcept
    -> bool
    {
      static_assert(
        std::is_same<::ket::meta::bit_integer_t<Qubit1>, ::ket::meta::bit_integer_t<Qubit2>>::value,
        "BitInteger's of Qubit1 and Qubit2 should be the same");
      static_assert(
        std::is_same<::ket::meta::state_integer_t<Qubit1>, ::ket::meta::state_integer_t<Qubit2>>::value,
        "StateInteger's of Qubit1 and Qubit2 should be the same");
      return permutated_qubit1.qubit() <= permutated_qubit2.qubit();
    }

    template <typename Qubit1, typename Qubit2>
    inline constexpr auto operator>=(
      ::ket::mpi::permutated<Qubit1> const permutated_qubit1, ::ket::mpi::permutated<Qubit2> const permutated_qubit2) noexcept
    -> bool
    {
      static_assert(
        std::is_same<::ket::meta::bit_integer_t<Qubit1>, ::ket::meta::bit_integer_t<Qubit2>>::value,
        "BitInteger's of Qubit1 and Qubit2 should be the same");
      static_assert(
        std::is_same<::ket::meta::state_integer_t<Qubit1>, ::ket::meta::state_integer_t<Qubit2>>::value,
        "StateInteger's of Qubit1 and Qubit2 should be the same");
      return permutated_qubit1.qubit() >= permutated_qubit2.qubit();
    }

    template <typename Qubit, typename Integer>
    inline constexpr auto operator+(::ket::mpi::permutated<Qubit> permutated_qubit, Integer const bit) noexcept -> ::ket::mpi::permutated<Qubit>
    { return permutated_qubit += bit; }

    template <typename Integer, typename Qubit>
    inline constexpr auto operator+(Integer const bit, ::ket::mpi::permutated<Qubit> const permutated_qubit) noexcept -> ::ket::mpi::permutated<Qubit>
    { return permutated_qubit + bit; }

    template <typename Qubit1, typename Qubit2>
    inline auto operator+(::ket::mpi::permutated<Qubit1> const permutated_qubit1, ::ket::mpi::permutated<Qubit2> const permutated_qubit2)
    -> ::ket::mpi::permutated<Qubit1>
    = delete;

    template <typename Qubit, typename Integer>
    inline constexpr auto operator-(::ket::mpi::permutated<Qubit> permutated_qubit, Integer const bit) noexcept -> ::ket::mpi::permutated<Qubit>
    { return permutated_qubit -= bit; }

    template <typename Qubit1, typename Qubit2>
    inline constexpr auto operator-(
      ::ket::mpi::permutated<Qubit1> const permutated_qubit1,
      ::ket::mpi::permutated<Qubit2> const permutated_qubit2) noexcept
    -> decltype(permutated_qubit1.qubit() - permutated_qubit2.qubit())
    {
      static_assert(
        std::is_same<::ket::meta::bit_integer_t<Qubit1>, ::ket::meta::bit_integer_t<Qubit2>>::value,
        "BitInteger's of Qubit1 and Qubit2 should be the same");
      static_assert(
        std::is_same<::ket::meta::state_integer_t<Qubit1>, ::ket::meta::state_integer_t<Qubit2>>::value,
        "StateInteger's of Qubit1 and Qubit2 should be the same");
      return permutated_qubit1.qubit() - permutated_qubit2.qubit();
    }

    template <typename UnsignedInteger, typename Qubit>
    inline constexpr std::enable_if_t<std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value, UnsignedInteger>
    operator<<(UnsignedInteger const unsigned_integer, ::ket::mpi::permutated<Qubit> const permutated_qubit) noexcept
    { return unsigned_integer << permutated_qubit.qubit(); }

    template <typename UnsignedInteger, typename Qubit>
    inline constexpr std::enable_if_t<std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value, UnsignedInteger>
    operator>>(UnsignedInteger const unsigned_integer, ::ket::mpi::permutated<Qubit> const permutated_qubit) noexcept
    { return unsigned_integer >> permutated_qubit.qubit(); }

    template <typename Qubit>
    inline constexpr auto make_permutated(Qubit const qubit) -> ::ket::mpi::permutated<Qubit>
    { return ::ket::mpi::permutated<Qubit>{qubit}; }

    namespace permutated_detail
    {
      template <typename Qubit>
      struct remove_control
      {
        static constexpr auto call(::ket::mpi::permutated<Qubit> const permutated_qubit) -> ::ket::mpi::permutated<Qubit>
        { return permutated_qubit; }
      }; // struct remove_control<Qubit>

      template <typename Qubit>
      struct remove_control< ::ket::control<Qubit> >
      {
        static constexpr auto call(
          ::ket::mpi::permutated< ::ket::control<Qubit> > const permutated_control_qubit)
        -> ::ket::mpi::permutated<Qubit>
        { return ::ket::mpi::make_permutated(permutated_control_qubit.qubit().qubit()); }
      }; // struct remove_control< ::ket::control<Qubit> >
    } // namespace permutated_detail

    template <typename Qubit>
    inline constexpr auto remove_control(::ket::mpi::permutated<Qubit> const permutated_qubit)
      -> decltype(::ket::mpi::permutated_detail::remove_control<Qubit>::call(permutated_qubit))
    { return ::ket::mpi::permutated_detail::remove_control<Qubit>::call(permutated_qubit); }

    namespace literals
    {
      inline namespace permutated_literals
      {
        inline constexpr auto operator"" _pq(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, unsigned int> >
        { return ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, unsigned int> >{static_cast<unsigned int>(bit)}; }

        inline constexpr auto operator"" _pqs(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, unsigned short int> >
        { return ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, unsigned short int> >{static_cast<unsigned short>(bit)}; }

        inline constexpr auto operator"" _pql(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, unsigned long int> >
        { return ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, unsigned long int> >{static_cast<unsigned long>(bit)}; }

        inline constexpr auto operator"" _pqll(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, unsigned long long int> >
        { return ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, unsigned long long int> >{bit}; }

        inline constexpr auto operator"" _pq8(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, std::uint8_t> >
        { return ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, std::uint8_t> >{static_cast<std::uint8_t>(bit)}; }

        inline constexpr auto operator"" _pq16(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, std::uint16_t> >
        { return ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, std::uint16_t> >{static_cast<std::uint16_t>(bit)}; }

        inline constexpr auto operator"" _pq32(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, std::uint32_t> >
        { return ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, std::uint32_t> >{static_cast<std::uint32_t>(bit)}; }

        inline constexpr auto operator"" _pq64(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, std::uint64_t> >
        { return ::ket::mpi::permutated< ::ket::qubit<std::uint64_t, std::uint64_t> >{static_cast<std::uint64_t>(bit)}; }

        inline constexpr auto operator"" _pcq(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, unsigned int> > >
        { return ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, unsigned int> > >{static_cast<unsigned int>(bit)}; }

        inline constexpr auto operator"" _pcqs(unsigned long long int const bit) noexcept
          -> ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, unsigned short int> > >
        { return ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, unsigned short int> > >{static_cast<unsigned short>(bit)}; }

        inline constexpr auto operator"" _pcql(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, unsigned long int> > >
        { return ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, unsigned long int> > >{static_cast<unsigned long>(bit)}; }

        inline constexpr auto operator"" _pcqll(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, unsigned long long int> > >
        { return ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, unsigned long long int> > >{bit}; }

        inline constexpr auto operator"" _pcq8(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, std::uint8_t> > >
        { return ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, std::uint8_t> > >{static_cast<std::uint8_t>(bit)}; }

        inline constexpr auto operator"" _pcq16(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, std::uint16_t> > >
        { return ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, std::uint16_t> > >{static_cast<std::uint16_t>(bit)}; }

        inline constexpr auto operator"" _pcq32(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, std::uint32_t> > >
        { return ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, std::uint32_t> > >{static_cast<std::uint32_t>(bit)}; }

        inline constexpr auto operator"" _pcq64(unsigned long long int const bit) noexcept
        -> ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, std::uint64_t> > >
        { return ::ket::mpi::permutated< ::ket::control< ::ket::qubit<std::uint64_t, std::uint64_t> > >{static_cast<std::uint64_t>(bit)}; }
      } // namespace permutated_literals
    } // namespace literals

    namespace [[deprecated]] permutated_literals
    { using namespace ::ket::mpi::literals::permutated_literals; }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_PERMUTATED_HPP
