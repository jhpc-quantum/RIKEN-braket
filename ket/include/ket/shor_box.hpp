#ifndef KET_SHOR_BOX_HPP
# define KET_SHOR_BOX_HPP

# include <boost/config.hpp>

# include <cmath>
//# include <vector>
# include <iterator>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_unsigned.hpp>
#   include <boost/type_traits/is_same.hpp>
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <ket/qubit.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/const_iterator_of.hpp>
# include <ket/utility/meta/real_of.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_is_unsigned std::is_unsigned
#   define KET_is_same std::is_same
# else
#   define KET_is_unsigned boost::is_unsigned
#   define KET_is_same boost::is_same
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif


namespace ket
{
  namespace shor_box_detail
  {
    template <typename StateInteger, typename BitInteger>
    inline StateInteger reverse_bits(StateInteger const integer, BitInteger const num_bits)
    {
      StateInteger result = static_cast<StateInteger>(0u);

      for (BitInteger next_from_bit = num_bits, to_bit = static_cast<BitInteger>(0u);
           next_from_bit > static_cast<BitInteger>(0u); --next_from_bit, ++to_bit)
        result |= ((integer >> (next_from_bit - static_cast<BitInteger>(1u))) bitand static_cast<StateInteger>(1u)) << to_bit;

      return result;
    }

    template <typename UnsignedInteger, typename Qubits>
    inline UnsignedInteger make_filtered_integer(UnsignedInteger const unsigned_integer, Qubits const& qubits)
    {
      typedef typename ::ket::utility::meta::const_iterator_of<Qubits const>::type qubits_iterator;
      typedef typename boost::range_size<Qubits const>::type qubits_size_type;
      qubits_size_type const num_qubits = boost::size(qubits);
      qubits_iterator const qubits_first = ::ket::utility::begin(qubits);

      UnsignedInteger result = static_cast<UnsignedInteger>(0u);
      for (qubits_size_type index = static_cast<qubits_size_type>(0u); index < num_qubits; ++index)
        result |= ((unsigned_integer >> index) bitand static_cast<UnsignedInteger>(1u)) << *(qubits_first + index);

      return result;
    }

    template <typename StateInteger, typename Qubits>
    inline StateInteger calculate_index(
      StateInteger const exponent, Qubits const& exponent_qubits,
      StateInteger const modular_exponentiation_value, Qubits const& modular_exponentiation_qubits)
    {
      return
        ::ket::shor_box_detail::make_filtered_integer(exponent, exponent_qubits)
        bitor ::ket::shor_box_detail::make_filtered_integer(modular_exponentiation_value, modular_exponentiation_qubits);
    }
  } // namespace shor_box_detail


  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename StateInteger, typename Qubits>
  inline void shor_box(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    StateInteger const base, StateInteger const divisor,
    Qubits const& exponent_qubits, Qubits const& modular_exponentiation_qubits)
  {
    typedef typename boost::range_value<Qubits>::type qubit_type;
    typedef typename ::ket::meta::bit_integer_of<qubit_type>::type bit_integer_type;
    static_assert(
      (KET_is_same<
         StateInteger, typename ::ket::meta::state_integer_of<qubit_type>::type>::value),
      "StateInteger should be state_integer_type of qubit_type");
    static_assert(KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
    static_assert(KET_is_unsigned<bit_integer_type>::value, "BitInteger should be unsigned");

    typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
    typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
    ::ket::utility::fill(
      parallel_policy, first, last, static_cast<complex_type>(static_cast<real_type>(0)));

    bit_integer_type const num_exponent_qubits = static_cast<bit_integer_type>(boost::size(exponent_qubits));
    StateInteger const num_exponents = ::ket::utility::integer_exp2<StateInteger>(num_exponent_qubits);
    StateInteger modular_exponentiation_value = static_cast<StateInteger>(1u);

    using std::pow;
    complex_type const constant_coefficient
      = static_cast<complex_type>(static_cast<real_type>(pow(num_exponents, -0.5)));

    for (StateInteger exponent = static_cast<StateInteger>(0u); exponent < num_exponents; ++exponent)
    {
      StateInteger const index
        = ::ket::shor_box_detail::calculate_index(
            ::ket::shor_box_detail::reverse_bits(exponent, num_exponent_qubits), exponent_qubits,
            modular_exponentiation_value, modular_exponentiation_qubits);

      *(first + index) = constant_coefficient;

      modular_exponentiation_value *= base;
      modular_exponentiation_value %= divisor;
    }


    /*
    // preliminary implement of parallel shor box
    typedef typename boost::range_size<Qubits const>::type qubits_size_type;
    qubits_size_type const num_exponent_qubits = boost::size(exponent_qubits);
    qubits_size_type const num_modular_exponentiation_qubits = boost::size(modular_exponentiation_qubits);

    typedef typename boost::range_value<Qubits>::type qubit_type;

    static_assert(KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
    static_assert(
      KET_is_unsigned<typename ::ket::meta::bit_integer_of<qubit_type>::type>::value,
      "BitInteger should be unsigned");

    std::vector<StateInteger> modular_squares;
    modular_squares.reserve(num_exponent_qubits);
    modular_squares.push_back(base);
    using ::ket::utility::loop_n;
    loop_n(
      ::ket::utility::policy::make_sequential(), num_exponent_qubits-static_cast<qubits_size_type>(1u),
      [&modular_squares, divisor](qubits_size_type const, int const)
      { modular_squares.push_back((modular_squares.back() * modular_squares.back()) % divisor); });

    std::vector<StateInteger> modular_exponentiation_values;
    modular_exponentiation_values.reserve(::ket::utility::num_threads());

    typedef typename boost::range_size<RandomAccessRange>::type range_size_type;
    using ::ket::utility::loop_n;
    loop_n(
      parallel_policy, boost::size(state),
      [first](range_size_type const index, int const)
      { *(first+index) = static_cast<complex_type>(static_cast<real_type>(0)); });

    using std::pow;
    complex_type const constant_coefficient = static_cast<complex_type>(pow(num_exponents, -0.5));
    StateInteger const num_exponents = ::ket::utility::integer_exp2<StateInteger>(num_exponent_qubits);
    StateInteger modular_exponentiation_value = static_cast<StateInteger>(1u);
    int const num_threads = ::ket::utility::num_threads();
    loop_n(
      parallel_policy, num_exponents,
      [](StateInteger const exponent, int const thread_index)
      {
        a
      });


    using std::pow;
    complex_type const constant_coefficient = static_cast<complex_type>(pow(num_exponents, -0.5));
    StateInteger const num_exponents = ::ket::utility::integer_exp2<StateInteger>(num_exponent_qubits);
    StateInteger modular_exponentiation_value = static_cast<StateInteger>(1u);
    for (StateInteger exponent = static_cast<StateInteger>(0u); exponent < num_exponents; ++exponent)
    {
      StateInteger const index
        = ::ket::shor_box_detail::reverse_bits(exponent, num_exponent_qubits)
          bitor (modular_exponentiation_value << num_exponent_qubits);

      *(first + convert(index, exponent_qubits, modular_exponentiation_qubits))
        = constant_coefficient;

      modular_exponentiation_value *= base;
      modular_exponentiation_value %= divisor;
    }
    */
  }

  template <typename RandomAccessIterator, typename StateInteger, typename Qubits>
  inline void shor_box(
    RandomAccessIterator const first, RandomAccessIterator const last,
    StateInteger const base, StateInteger const divisor,
    Qubits const& exponent_qubits, Qubits const& modular_exponentiation_qubits)
  {
    ::ket::shor_box(
      ::ket::utility::policy::make_sequential(),
      first, last,
      base, divisor, exponent_qubits, modular_exponentiation_qubits);
  }

  namespace ranges
  {
    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename StateInteger, typename Qubits>
    inline RandomAccessRange& shor_box(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      StateInteger const base, StateInteger const divisor,
      Qubits const& exponent_qubits, Qubits const& modular_exponentiation_qubits)
    {
      ::ket::shor_box(
        parallel_policy, ::ket::utility::begin(state), ::ket::utility::end(state),
        base, divisor, exponent_qubits, modular_exponentiation_qubits);
      return state;
    }

    template <typename RandomAccessRange, typename StateInteger, typename Qubits>
    inline RandomAccessRange& shor_box(
      RandomAccessRange& state,
      StateInteger const base, StateInteger const divisor,
      Qubits const& exponent_qubits, Qubits const& modular_exponentiation_qubits)
    {
      ::ket::shor_box(
        ::ket::utility::policy::make_sequential(),
        ::ket::utility::begin(state), ::ket::utility::end(state),
        base, divisor, exponent_qubits, modular_exponentiation_qubits);
      return state;
    }
  }
}


# undef KET_is_unsigned
# undef KET_is_same
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

