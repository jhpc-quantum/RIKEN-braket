#ifndef KET_ADDITION_ASSIGNMENT_HPP
# define KET_ADDITION_ASSIGNMENT_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cstddef>
# include <vector>
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

# include <boost/range/size.hpp>
# include <boost/range/value_type.hpp>
# ifndef NDEBUG
#   include <boost/range/join.hpp>
# endif

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/swapped_fourier_transform.hpp>
# include <ket/gate/controlled_phase_shift.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/integer_exp2.hpp>
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
  // lhs += rhs
  namespace addition_assignment_detail
  {
    template <
      typename ParallelPolicy,
      typename RandomAccessIterator, typename Iterator1, typename Iterator2,
      typename PhaseCoefficientsAllocator>
    inline void addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Iterator1 const& lhs_qubits_first, Iterator2 const& rhs_qubits_first,
      std::size_t const num_qubits,
      std::vector<
        typename std::iterator_traits<RandomAccessIterator>::value_type,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      for (std::size_t phase_exponent = 1u;
           phase_exponent <= num_qubits; ++phase_exponent)
      {
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
        complex_type const phase_coefficient = phase_coefficients[phase_exponent];

        for (std::size_t control_bit_index = 0u;
             control_bit_index <= num_qubits-phase_exponent; ++control_bit_index)
        {
          std::size_t const target_bit_index
            = control_bit_index+(phase_exponent-1u);

          ::ket::gate::controlled_phase_shift_coeff(
            parallel_policy,
            first, last, phase_coefficient,
            lhs_qubits_first[target_bit_index],
            ::ket::make_control(rhs_qubits_first[control_bit_index]));
        }
      }
    }
  } // namespace addition_assignment_detail


  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline void addition_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector<
      typename std::iterator_traits<RandomAccessIterator>::value_type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    typename boost::range_size<Qubits const>::type const num_qubits
      = boost::size(lhs_qubits);
    assert(
      boost::algorithm::all_of(
        rhs_qubits_range,
        [num_qubits](typename boost::range_value<QubitsRange const>::type const& rhs_qubits)
        { return num_qubits == boost::size(rhs_qubits); }));

    ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

    typedef typename boost::range_value<Qubits>::type qubit_type;

    static_assert(
      KET_is_unsigned<typename ::ket::meta::state_integer_of<qubit_type>::type>::value,
      "StateInteger should be unsigned");
    static_assert(
      KET_is_unsigned<typename ::ket::meta::bit_integer_of<qubit_type>::type>::value,
      "BitInteger should be unsigned");

    ::ket::swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);

# ifndef BOOST_NO_CXX11_RANGE_BASED_FOR
    typedef typename boost::range_value<QubitsRange const>::type qubits_type;
    for (qubits_type const& rhs_qubits: rhs_qubits_range)
      ::ket::addition_assignment_detail::addition_assignment(
        parallel_policy,
        first, last, ::ket::utility::begin(lhs_qubits), ::ket::utility::begin(rhs_qubits), num_qubits, phase_coefficients);
# else // BOOST_NO_CXX11_RANGE_BASED_FOR
    typedef typename ::ket::utility::meta::const_iterator_of<QubitsRange const>::type iterator;
    iterator rhs_qubits_iter = ::ket::utility::begin(rhs_qubits_range);
    iterator const rhs_qubits_last = ::ket::utility::end(rhs_qubits_range);
    for (; rhs_qubits_iter != rhs_qubits_last; ++rhs_qubits_iter)
    {
      Qubits const rhs_qubits = *rhs_qubits_iter;
      ::ket::addition_assignment_detail::addition_assignment(
        parallel_policy,
        first, last, ::ket::utility::begin(lhs_qubits), ::ket::utility::begin(rhs_qubits), num_qubits, phase_coefficients);
    }
# endif // BOOST_NO_CXX11_RANGE_BASED_FOR

    ::ket::adj_swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);
  }

  template <
    typename RandomAccessIterator, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline typename KET_enable_if<
    not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessIterator>::value,
    void>::type
  addition_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector<
      typename std::iterator_traits<RandomAccessIterator>::value_type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    ::ket::addition_assignment(
      ::ket::utility::policy::make_sequential(),
      first, last, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline typename KET_enable_if<
    ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
    void>::type
  addition_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
    std::vector<complex_type> phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

    ::ket::addition_assignment(
      parallel_policy,
      first, last, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <
    typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline void addition_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  {
    ::ket::addition_assignment(
      ::ket::utility::policy::make_sequential(),
      first, last, lhs_qubits, rhs_qubits_range);
  }


  namespace ranges
  {
    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      ::ket::addition_assignment(
        parallel_policy,
        ::ket::utility::begin(state), ::ket::utility::end(state),
        lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator>
    inline typename KET_enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value,
      RandomAccessRange&>::type
    addition_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      ::ket::addition_assignment(
        ::ket::utility::begin(state), ::ket::utility::end(state),
        lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      ::ket::addition_assignment(
        parallel_policy,
        ::ket::utility::begin(state), ::ket::utility::end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline RandomAccessRange& addition_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      ::ket::addition_assignment(
        ::ket::utility::begin(state), ::ket::utility::end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }
  }


  namespace addition_assignment_detail
  {
    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename Iterator1, typename Iterator2,
      typename PhaseCoefficientsAllocator>
    inline void adj_addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Iterator1 const& lhs_qubits_first, Iterator2 const& rhs_qubits_first,
      std::size_t const num_qubits,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      for (std::size_t index = 0u; index < num_qubits; ++index)
      {
        std::size_t const phase_exponent = num_qubits-index;

        typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
        complex_type const phase_coefficient = phase_coefficients[phase_exponent];

        for (std::size_t control_bit_index = 0u;
             control_bit_index <= num_qubits-phase_exponent; ++control_bit_index)
        {
          std::size_t const target_bit_index
            = control_bit_index+(phase_exponent-1u);

          ::ket::gate::adj_controlled_phase_shift_coeff(
            parallel_policy,
            first, last, phase_coefficient,
            lhs_qubits_first[target_bit_index],
            ::ket::make_control(rhs_qubits_first[control_bit_index]));
        }
      }
    }
  } // namespace addition_assignment_detail


  template <
    typename ParallelPolicy, typename RandomAccessIterator,
    typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline void adj_addition_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    typename boost::range_size<Qubits const>::type const num_qubits
      = boost::size(lhs_qubits);
    assert(
      boost::algorithm::all_of(
        rhs_qubits_range,
        [num_qubits](typename boost::range_value<QubitsRange const>::type const& rhs_qubits)
        { return num_qubits == boost::size(rhs_qubits); }));

    ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

    typedef typename boost::range_value<Qubits>::type qubit_type;

    static_assert(
      KET_is_unsigned<typename ::ket::meta::state_integer_of<qubit_type>::type>::value,
      "StateInteger should be unsigned");
    static_assert(
      KET_is_unsigned<typename ::ket::meta::bit_integer_of<qubit_type>::type>::value,
      "BitInteger should be unsigned");

    ::ket::swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);

# ifndef BOOST_NO_CXX11_RANGE_BASED_FOR
    typedef typename boost::range_value<QubitsRange const>::type qubits_type;
    for (qubits_type const& rhs_qubits: rhs_qubits_range)
      ::ket::addition_assignment_detail::adj_addition_assignment(
        parallel_policy,
        first, last, ::ket::utility::begin(lhs_qubits), ::ket::utility::begin(rhs_qubits), num_qubits, phase_coefficients);
# else // BOOST_NO_CXX11_RANGE_BASED_FOR
    typedef typename ::ket::utility::meta::const_iterator_of<QubitsRange const>::type iterator;
    iterator rhs_qubits_iter = ::ket::utility::begin(rhs_qubits_range);
    iterator const rhs_qubits_last = ::ket::utility::end(rhs_qubits_range);
    for (; rhs_qubits_iter != rhs_qubits_last; ++rhs_qubits_iter)
    {
      Qubits const rhs_qubits = *rhs_qubits_iter;
      ::ket::addition_assignment_detail::do_adj_addition_assignment(
        parallel_policy,
        first, last, ::ket::utility::begin(lhs_qubits), ::ket::utility::begin(rhs_qubits), num_qubits, phase_coefficients);
    }
# endif // BOOST_NO_CXX11_RANGE_BASED_FOR

    ::ket::adj_swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);
  }

  template <
    typename RandomAccessIterator,
    typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline typename KET_enable_if<
    not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessIterator>::value,
    void>::type
  adj_addition_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    ::ket::adj_addition_assignment(
      ::ket::utility::policy::make_sequential(),
      first, last, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline typename KET_enable_if<
    ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
    void>::type
  adj_addition_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
    std::vector<complex_type> phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

    ::ket::adj_addition_assignment(
      parallel_policy,
      first, last, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <
    typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline void adj_addition_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  {
    ::ket::adj_addition_assignment(
      ::ket::utility::policy::make_sequential(),
      first, last, lhs_qubits, rhs_qubits_range);
  }


  namespace ranges
  {
    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& adj_addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      ::ket::adj_addition_assignment(
        parallel_policy,
        ::ket::utility::begin(state), ::ket::utility::end(state),
        lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline typename KET_enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value,
      RandomAccessRange&>::type
    adj_addition_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      ::ket::adj_addition_assignment(
        ::ket::utility::begin(state), ::ket::utility::end(state),
        lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename Qubits, typename QubitsRange>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      ::ket::adj_addition_assignment(
        parallel_policy,
        ::ket::utility::begin(state), ::ket::utility::end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline RandomAccessRange& adj_addition_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      ::ket::adj_addition_assignment(
        ::ket::utility::begin(state), ::ket::utility::end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }
  }
}


# undef KET_enable_if
# undef KET_is_unsigned
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

