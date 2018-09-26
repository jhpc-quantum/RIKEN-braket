#ifndef KET_SWAPPED_FOURIER_TRANSFORM_HPP
# define KET_SWAPPED_FOURIER_TRANSFORM_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cstddef>
# include <cmath>
# include <iterator>
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
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/const_iterator_of.hpp>
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
    typename ParallelPolicy, typename RandomAccessIterator, typename Qubits,
    typename PhaseCoefficientsAllocator>>
  inline void swapped_fourier_transform(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
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
    assert(
      ::ket::utility::integer_exp2<typename ::ket::meta::state_integer_of<qubit_type>::type>(num_qubits)
        <= static_cast<typename ::ket::meta::state_integer_of<qubit_type>::type>(boost::size(state))
      and ::ket::utility::ranges::is_unique(qubits));

    typedef typename ::ket::iterator::meta::const_iterator_of<Qubits const>::type qubits_iterator;
    qubits_iterator const qubits_first = ::ket::utility::begin(qubits);

    for (std::size_t index = 0u; index < num_qubits; ++index)
    {
      std::size_t target_bit = num_qubits-index-1u;

      ::ket::gate::hadamard(parallel_policy, first, last, qubits_first[target_bit]);

      for (std::size_t phase_exponent = 2u;
           phase_exponent <= num_qubits-index; ++phase_exponent)
      {
        std::size_t const control_bit = target_bit-(phase_exponent-1u);

        ::ket::gate::controlled_phase_shift(
          parallel_policy,
          first, last, phase_coefficients[phase_exponent],
          qubits_first[target_bit], ::ket::make_control(qubits_first[control_bit]));
      }
    }
  }

  template <typename RandomAccessIterator, typename Qubits, typename PhaseCoefficientsAllocator>
  inline typename KET_enable_if<
    not ::ket::utility::policy::is_loop_n_policy<RandomAccessIterator>::value,
    void>::type
  swapped_fourier_transform(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    ::ket::swapped_fourier_transform(
      ::ket::utility::policy::make_sequential(), first, last, qubits, phase_coefficients);
  }

  template <
    typename ParallelPolicy, typename RandomAccessIterator, typename Qubits>
  inline typename KET_enable_if<
    ::ket::utility::policy::is_loop_n_policy<ParallelPolicy>::value,
    void>::type
  swapped_fourier_transform(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits)
  {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
    std::vector<complex_type> phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

    ::ket::swapped_fourier_transform(
      parallel_policy, first, last, qubits, phase_coefficients);
  }

  template <
    typename ParallelPolicy, typename RandomAccessIterator, typename Qubits>
  inline void swapped_fourier_transform(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits)
  {
    ::ket::swapped_fourier_transform(
      ::ket::utility::policy::make_sequential(), first, last, qubits);
  }


  namespace ranges
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
      ::ket::swapped_fourier_transform(
        parallel_policy,
        ::ket::utility::begin(state), ::ket::utility::end(state), qubits, phase_coefficients);
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
      ::ket::swapped_fourier_transform(
        ::ket::utility::begin(state), ::ket::utility::end(state), qubits, phase_coefficients);
      return state;
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
      ::ket::swapped_fourier_transform(
        parallel_policy, ::ket::utility::begin(state), ::ket::utility::end(state), qubits);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& swapped_fourier_transform(
      RandomAccessRange& state, Qubits const& qubits)
    {
      ::ket::swapped_fourier_transform(::ket::utility::begin(state), ::ket::utility::end(state), qubits);
      return state;
    }
  }


  template <
    typename ParallelPolicy, typename RandomAccessIterator, typename Qubits,
    typename PhaseCoefficientsAllocator>>
  inline void adj_swapped_fourier_transform(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
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
    assert(
      ::ket::utility::integer_exp2<typename ::ket::meta::state_integer_of<qubit_type>::type>(num_qubits)
        <= static_cast<typename ::ket::meta::state_integer_of<qubit_type>::type>(boost::size(state))
      and ::ket::utility::ranges::is_unique(qubits));

    typedef typename ::ket::utility::meta::const_iterator_of<Qubits const>::type qubits_iterator;
    qubits_iterator const qubits_first = ::ket::utility::begin(qubits);

    for (std::size_t target_bit = 0u; target_bit < num_qubits; ++target_bit)
    {
      for (std::size_t index = 0u; index < target_bit; ++index)
      {
        std::size_t const phase_exponent = 1u+target_bit-index;
        std::size_t const control_bit = target_bit-(phase_exponent-1u);

        ::ket::gate::adj_controlled_phase_shift(
          parallel_policy,
          first, last, phase_coefficients[phase_exponent], qubits_first[target_bit],
          ::ket::make_control(qubits_first[control_bit]));
      }

      ::ket::gate::adj_hadamard(parallel_policy, first, last, qubits_first[target_bit]);
    }
  }

  template <typename RandomAccessIterator, typename Qubits, typename PhaseCoefficientsAllocator>
  inline typename KET_enable_if<
    not ::ket::utility::policy::is_loop_n_policy<RandomAccessIterator>::value,
    void>::type
  adj_swapped_fourier_transform(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    ::ket::adj_swapped_fourier_transform(
      ::ket::utility::policy::make_sequential(), first, last, qubits, phase_coefficients);
  }

  template <
    typename ParallelPolicy, typename RandomAccessIterator, typename Qubits>
  inline typename KET_enable_if<
    ::ket::utility::policy::is_loop_n_policy<ParallelPolicy>::value,
    void>::type
  adj_swapped_fourier_transform(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits)
  {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
    std::vector<complex_type> phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

    ::ket::adj_swapped_fourier_transform(
      parallel_policy, first, last, qubits, phase_coefficients);
  }

  template <
    typename ParallelPolicy, typename RandomAccessIterator, typename Qubits>
  inline void adj_swapped_fourier_transform(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits)
  {
    ::ket::adj_swapped_fourier_transform(
      ::ket::utility::policy::make_sequential(), first, last, qubits);
  }


  namespace ranges
  {
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
      ::ket::adj_swapped_fourier_transform(
        parallel_policy,
        ::ket::utility::begin(state), ::ket::utility::end(state),
        qubits, phase_coefficients);
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
      ::ket::adj_swapped_fourier_transform(
        ::ket::utility::begin(state), ::ket::utility::end(state),
        qubits, phase_coefficients);
      return state;
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
      ::ket::adj_swapped_fourier_transform(
        parallel_policy, ::ket::utility::begin(state), ::ket::utility::end(state), qubits);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& adj_swapped_fourier_transform(
      RandomAccessRange& state, Qubits const& qubits)
    {
      ::ket::adj_swapped_fourier_transform(::ket::utility::begin(state), ::ket::utility::end(state), qubits);
      return state;
    }
  }
} // namespace ket


# undef KET_enable_if
# undef KET_is_unsigned
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

