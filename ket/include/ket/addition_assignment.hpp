#ifndef KET_ADDITION_ASSIGNMENT_HPP
# define KET_ADDITION_ASSIGNMENT_HPP

# include <cassert>
# include <cstddef>
# include <vector>
# include <iterator>
# include <type_traits>

# include <boost/range/value_type.hpp>

# include <ket/control.hpp>
# include <ket/swapped_fourier_transform.hpp>
# include <ket/gate/controlled_phase_shift.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/generate_phase_coefficients.hpp>


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
      for (auto phase_exponent = std::size_t{1u};
           phase_exponent <= num_qubits; ++phase_exponent)
      {
        auto const phase_coefficient = phase_coefficients[phase_exponent];

        for (auto control_bit_index = std::size_t{0u};
             control_bit_index <= num_qubits - phase_exponent; ++control_bit_index)
        {
          auto const target_bit_index
            = control_bit_index + (phase_exponent - std::size_t{1u});

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
    auto const num_qubits = std::distance(std::begin(lhs_qubits), std::end(rhs_qubits));
    assert(
      std::all_of(
        std::begin(rhs_qubits_range), std::end(rhs_qubits_range),
        [num_qubits](typename boost::range_value<QubitsRange const>::type const& rhs_qubits)
        { return num_qubits == std::distance(std::begin(rhs_qubits), std::end(rhs_qubits)); }));

    ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

    using qubit_type = typename boost::range_value<Qubits>::type;
    static_assert(
      std::is_unsigned<typename ::ket::meta::state_integer_of<qubit_type>::type>::value,
      "StateInteger should be unsigned");
    static_assert(
      std::is_unsigned<typename ::ket::meta::bit_integer_of<qubit_type>::type>::value,
      "BitInteger should be unsigned");

    ::ket::swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);

    for (auto const& rhs_qubits: rhs_qubits_range)
      ::ket::addition_assignment_detail::addition_assignment(
        parallel_policy,
        first, last, std::begin(lhs_qubits), std::begin(rhs_qubits), num_qubits, phase_coefficients);

    ::ket::adj_swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);
  }

  template <
    typename RandomAccessIterator, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline typename std::enable_if<
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
  inline typename std::enable_if<
    ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
    void>::type
  addition_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  {
    using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    auto phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(
          std::distance(std::begin(lhs_qubits), std::end(lhs_qubits)));

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
        std::begin(state), std::end(state),
        lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator>
    inline typename std::enable_if<
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
        std::begin(state), std::end(state),
        lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      ::ket::addition_assignment(
        parallel_policy,
        std::begin(state), std::end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline RandomAccessRange& addition_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      ::ket::addition_assignment(
        std::begin(state), std::end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }
  } // namespace ranges


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
      for (auto index = std::size_t{0u}; index < num_qubits; ++index)
      {
        auto const phase_exponent = num_qubits - index;
        auto const phase_coefficient = phase_coefficients[phase_exponent];

        for (auto control_bit_index = std::size_t{0u};
             control_bit_index <= num_qubits - phase_exponent; ++control_bit_index)
        {
          auto const target_bit_index
            = control_bit_index + (phase_exponent - std::size_t{1u});

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
    auto const num_qubits = std::distance(std::begin(lhs_qubits), std::end(rhs_qubits));
    assert(
      std::all_of(
        std::begin(rhs_qubits_range), std::end(rhs_qubits_range),
        [num_qubits](typename boost::range_value<QubitsRange const>::type const& rhs_qubits)
        { return num_qubits == std::distance(std::begin(rhs_qubits), std::end(rhs_qubits)); }));

    ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

    using qubit_type = typename boost::range_value<Qubits>::type;
    static_assert(
      std::is_unsigned<typename ::ket::meta::state_integer_of<qubit_type>::type>::value,
      "StateInteger should be unsigned");
    static_assert(
      std::is_unsigned<typename ::ket::meta::bit_integer_of<qubit_type>::type>::value,
      "BitInteger should be unsigned");

    ::ket::swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);

    for (auto const& rhs_qubits: rhs_qubits_range)
      ::ket::addition_assignment_detail::adj_addition_assignment(
        parallel_policy,
        first, last, std::begin(lhs_qubits), std::begin(rhs_qubits), num_qubits, phase_coefficients);

    ::ket::adj_swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);
  }

  template <
    typename RandomAccessIterator,
    typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline typename std::enable_if<
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
  inline typename std::enable_if<
    ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
    void>::type
  adj_addition_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  {
    using complex_type = std::iterator_traits<RandomAccessIterator>::value_type;
    auto phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(
          std::distance(std::begin(lhs_qubits), std::end(lhs_qubits)));

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
        std::begin(state), std::end(state),
        lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline typename std::enable_if<
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
        std::begin(state), std::end(state),
        lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename Qubits, typename QubitsRange>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      ::ket::adj_addition_assignment(
        parallel_policy,
        std::begin(state), std::end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline RandomAccessRange& adj_addition_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      ::ket::adj_addition_assignment(
        std::begin(state), std::end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }
  } // namespace ranges
} // namespace ket


#endif // KET_ADDITION_ASSIGNMENT_HPP
