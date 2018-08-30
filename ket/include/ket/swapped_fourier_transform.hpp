#ifndef KET_SWAPPED_FOURIER_TRANSFORM_HPP
# define KET_SWAPPED_FOURIER_TRANSFORM_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cstddef>
# include <cmath>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_unsigned.hpp>
#   include <boost/utility/enable_if.hpp>
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <boost/math/constants/constants.hpp>

# include <boost/range/value_type.hpp>
# include <boost/range/size.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/hadamard.hpp>
# include <ket/gate/controlled_phase_shift.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/is_unique.hpp>
# endif
# include <ket/utility/generate_phase_coefficients.hpp>
# include <ket/utility/meta/real_of.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_is_unsigned std::is_unsigned
#   define KET_enable_if std::enable_if
# else
#   define KET_is_unsigned boost::is_unsigned
#   define KET_enable_if boost::enable_if_c
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif


namespace ket
{
  template <
    typename ParallelPolicy, typename RandomAccessRange, typename Qubits,
    typename PhaseCoefficientsAllocator>>
  inline RandomAccessRange& swapped_fourier_transform(
    ParallelPolicy const parallel_policy, RandomAccessRange& state,
    Qubits const& qubits,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    typename boost::range_size<Qubits>::type const num_qubits
      = boost::size(qubits);
    ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

    typedef typename boost::range_value<Qubits>::type qubit_type;

    static_assert(
      KET_is_unsigned<typename ::ket::meta::state_integer_of<qubit_type>::type>::value,
      "StateInteger should be unsigned");
    static_assert(
      KET_is_unsigned<typename ::ket::meta::bit_integer_of<qubit_type>::type>::value,
      "BitInteger should be unsigned");
# ifndef NDEBUG
    using ::ket::utility::range::is_unique;
# endif
    assert(
      ::ket::utility::integer_exp2<typename ::ket::meta::state_integer_of<qubit_type>::typeer>(num_qubits)
        <= static_cast<typename ::ket::meta::state_integer_of<qubit_type>::typeer>(boost::size(state))
      and is_unique(qubits));

    typedef typename boost::range_iterator<Qubits const>::type qubits_iterator;
    qubits_iterator const qubits_first = boost::begin(qubits);

    for (std::size_t index = 0u; index < num_qubits; ++index)
    {
      std::size_t target_bit = num_qubits-index-1u;

      using ::ket::gate::hadamard;
      hadamard(parallel_policy, state, qubits_first[target_bit]);

      for (std::size_t phase_exponent = 2u;
           phase_exponent <= num_qubits-index; ++phase_exponent)
      {
        std::size_t const control_bit = target_bit-(phase_exponent-1u);

        using ::ket::gate::controlled_phase_shift;
        controlled_phase_shift(
          parallel_policy,
          state, phase_coefficients[phase_exponent],
          qubits_first[target_bit], ::ket::make_control(qubits_first[control_bit]));
      }
    }

    return state;
  }

  template <typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator>
  inline typename KET_enable_if<
    not ::ket::utility::policy::is_loop_n_policy<RandomAccessRange>::value,
    RandomAccessRange&>::type
  swapped_fourier_transform(
    RandomAccessRange& state, Qubits const& qubits,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    return ::ket::swapped_fourier_transform(
      ::ket::utility::policy::make_sequential(), state, qubits, phase_coefficients);
  }


  template <
    typename ParallelPolicy, typename RandomAccessRange, typename Qubits>
  inline typename KET_enable_if<
    ::ket::utility::policy::is_loop_n_policy<ParallelPolicy>::value,
    RandomAccessRange&>::type
  swapped_fourier_transform(
    ParallelPolicy const parallel_policy, RandomAccessRange& state,
    Qubits const& qubits)
  {
    typedef typename boost::range_value<RandomAccessRange>::type complex_type;
    std::vector<complex_type> phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

    return ::ket::swapped_fourier_transform(
      parallel_policy, state, qubits, phase_coefficients);
  }

  template <typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator>
  inline RandomAccessRange& swapped_fourier_transform(
    RandomAccessRange& state, Qubits const& qubits)
  {
    typedef typename boost::range_value<RandomAccessRange>::type complex_type;
    std::vector<complex_type> phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

    return ::ket::swapped_fourier_transform(state, qubits, phase_coefficients);
  }


  template <
    typename ParallelPolicy, typename RandomAccessRange, typename Qubits,
    typename PhaseCoefficientsAllocator>
  inline RandomAccessRange& adj_swapped_fourier_transform(
    ParallelPolicy const parallel_policy, RandomAccessRange& state,
    Qubits const& qubits,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    typename boost::range_size<Qubits>::type const num_qubits
      = boost::size(qubits);
    ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

    typedef typename boost::range_value<Qubits>::type qubit_type;

    static_assert(
      KET_is_unsigned<typename ::ket::meta::state_integer_of<qubit_type>::type>::value,
      "StateInteger should be unsigned");
    static_assert(
      KET_is_unsigned<typename ::ket::meta::bit_integer_of<qubit_type>::type>::value,
      "BitInteger should be unsigned");
# ifndef NDEBUG
    using ::ket::utility::range::is_unique;
# endif
    assert(
      ::ket::utility::integer_exp2<typename ::ket::meta::state_integer_of<qubit_type>::typeer>(num_qubits)
        <= static_cast<typename ::ket::meta::state_integer_of<qubit_type>::typeer>(boost::size(state))
      and is_unique(qubits));

    typedef typename boost::range_iterator<Qubits const>::type qubits_iterator;
    qubits_iterator const qubits_first = boost::begin(qubits);

    for (std::size_t target_bit = 0u; target_bit < num_qubits; ++target_bit)
    {
      for (std::size_t index = 0u; index < target_bit; ++index)
      {
        std::size_t const phase_exponent = 1u+target_bit-index;
        std::size_t const control_bit = target_bit-(phase_exponent-1u);

        using ::ket::gate::adj_controlled_phase_shift;
        adj_controlled_phase_shift(
          parallel_policy,
          state, phase_coefficients[phase_exponent], qubits_first[target_bit],
          ::ket::make_control(qubits_first[control_bit]));
      }

      using ::ket::gate::adj_hadamard;
      adj_hadamard(parallel_policy, state, qubits_first[target_bit]);
    }

    return state;
  }

  template <typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator>
  inline typename KET_enable_if<
    not ::ket::utility::policy::is_loop_n_policy<RandomAccessRange>::value,
    RandomAccessRange&>::type
  adj_swapped_fourier_transform(
    RandomAccessRange& state, Qubits const& qubits,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    return adj_swapped_fourier_transform(
      ::ket::utility::policy::make_sequential(), state, qubits, phase_coefficients);
  }


  template <
    typename ParallelPolicy, typename RandomAccessRange, typename Qubits>
  inline typename KET_enable_if<
    ::ket::utility::policy::is_loop_n_policy<ParallelPolicy>::value,
    RandomAccessRange&>::type
  adj_swapped_fourier_transform(
    ParallelPolicy const parallel_policy, RandomAccessRange& state,
    Qubits const& qubits)
  {
    typedef typename boost::range_value<RandomAccessRange>::type complex_type;
    std::vector<complex_type> phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

    return adj_swapped_fourier_transform(
      parallel_policy, state, qubits, phase_coefficients);
  }

  template <typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator>
  inline RandomAccessRange& adj_swapped_fourier_transform(
    RandomAccessRange& state, Qubits const& qubits)
  {
    typedef typename boost::range_value<RandomAccessRange>::type complex_type;
    std::vector<complex_type> phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

    return adj_swapped_fourier_transform(state, qubits, phase_coefficients);
  }
} // namespace ket


# undef KET_enable_if
# undef KET_is_unsigned
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

