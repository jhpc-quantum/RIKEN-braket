#ifndef KET_CONTROL_HPP
#define KET_CONTROL_HPP

# include <boost/config.hpp>

# include <boost/cstdint.hpp>

# include <ket/qubit.hpp>


namespace ket
{
  template <typename Qubit>
  class control
  {
    typedef Qubit qubit_type;
    qubit_type qubit_;

   public:
    BOOST_CONSTEXPR control() BOOST_NOEXCEPT_OR_NOTHROW : qubit_() { }

    explicit BOOST_CONSTEXPR control(Qubit const qubit) BOOST_NOEXCEPT_OR_NOTHROW
      : qubit_(qubit)
    { }

    template <typename BitInteger_>
    explicit BOOST_CONSTEXPR control(BitInteger_ const bit) BOOST_NOEXCEPT_OR_NOTHROW
      : qubit_(bit)
    { }

    qubit_type& qubit() { return qubit_; }
    qubit_type const& qubit() const { return qubit_; }
  };

  template <typename Qubit>
  inline BOOST_CONSTEXPR bool operator==(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return control_qubit1.qubit() == control_qubit2.qubit(); }

  template <typename Qubit>
  inline BOOST_CONSTEXPR bool operator!=(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return control_qubit1.qubit() != control_qubit2.qubit(); }

  template <typename Qubit>
  inline BOOST_CONSTEXPR bool operator<(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return control_qubit1.qubit() < control_qubit2.qubit(); }

  template <typename Qubit>
  inline BOOST_CONSTEXPR bool operator>(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return control_qubit1.qubit() > control_qubit2.qubit(); }

  template <typename Qubit>
  inline BOOST_CONSTEXPR bool operator<=(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return control_qubit1.qubit() <= control_qubit2.qubit(); }

  template <typename Qubit>
  inline BOOST_CONSTEXPR bool operator>=(
    ::ket::control<Qubit> const control_qubit1,
    ::ket::control<Qubit> const control_qubit2)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return control_qubit1.qubit() >= control_qubit2.qubit(); }


  template <typename Qubit>
  inline BOOST_CONSTEXPR ::ket::control<Qubit> make_control(Qubit const qubit)
  { return ::ket::control<Qubit>(qubit); }


# ifndef BOOST_NO_CXX11_USER_DEFINED_LITERALS
  namespace control_literals
  {
    inline BOOST_CONSTEXPR ::ket::control<ket::qubit<boost::uint64_t, unsigned int> >
    operator"" _cq(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::control<ket::qubit<boost::uint64_t, unsigned int> >{static_cast<unsigned int>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::control<ket::qubit<boost::uint64_t, unsigned short int> >
    operator"" _cqs(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::control<ket::qubit<boost::uint64_t, unsigned short int> >{static_cast<unsigned short>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::control<ket::qubit<boost::uint64_t, unsigned long int> >
    operator"" _cql(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::control<ket::qubit<boost::uint64_t, unsigned long int> >{static_cast<unsigned long>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::control<ket::qubit<boost::uint64_t, unsigned long long int> >
    operator"" _cqll(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::control<ket::qubit<boost::uint64_t, unsigned long long int> >{bit}; }

    inline BOOST_CONSTEXPR ::ket::control<ket::qubit<boost::uint64_t, boost::uint8_t> >
    operator"" _cq8(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::control<ket::qubit<boost::uint64_t, boost::uint8_t> >{static_cast<boost::uint8_t>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::control<ket::qubit<boost::uint64_t, boost::uint16_t> >
    operator"" _cq16(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::control<ket::qubit<boost::uint64_t, boost::uint16_t> >{static_cast<boost::uint16_t>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::control<ket::qubit<boost::uint64_t, boost::uint32_t> >
    operator"" _cq32(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::control<ket::qubit<boost::uint64_t, boost::uint32_t> >{static_cast<boost::uint32_t>(bit)}; }

    inline BOOST_CONSTEXPR ::ket::control<ket::qubit<boost::uint64_t, boost::uint64_t> >
    operator"" _cq64(unsigned long long int const bit) BOOST_NOEXCEPT_OR_NOTHROW
    { return ::ket::control<ket::qubit<boost::uint64_t, boost::uint64_t> >{static_cast<boost::uint64_t>(bit)}; }
  }
# endif
}


#endif

