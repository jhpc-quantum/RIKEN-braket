#ifndef KET_ADDITION_ASSIGNMENT_HPP
# define KET_ADDITION_ASSIGNMENT_HPP

# include <cassert>
# include <cstddef>
# include <vector>
# include <iterator>
# include <type_traits>

# include <ket/control.hpp>
# include <ket/swapped_fourier_transform.hpp>
# include <ket/gate/controlled_phase_shift.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/generate_phase_coefficients.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  // lhs += rhs
  namespace addition_assignment_detail
  {
    template <
      typename ParallelPolicy,
      typename RandomAccessIterator, typename Iterator1, typename Iterator2,
      typename BitInteger, typename PhaseCoefficientsAllocator>
    inline auto addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Iterator1 const& lhs_qubits_first, Iterator2 const& rhs_qubits_first, BitInteger const register_size,
      std::vector<typename std::iterator_traits<RandomAccessIterator>::value_type, PhaseCoefficientsAllocator> const& phase_coefficients)
    -> void
    {
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      for (auto phase_exponent = BitInteger{1u}; phase_exponent <= register_size; ++phase_exponent)
      {
        auto const phase_coefficient = phase_coefficients[phase_exponent];

        for (auto control_bit_index = BitInteger{0u};
             control_bit_index <= register_size - phase_exponent; ++control_bit_index)
        {
          auto const target_bit_index = control_bit_index + (phase_exponent - BitInteger{1u});

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
    using bit_integer_type = ::ket::meta::bit_integer_t< ::ket::utility::meta::range_value_t<Qubits> >;
    static_assert(std::is_unsigned< ::ket::meta::state_integer_t< ::ket::utility::meta::range_value_t<Qubits> > >::value, "StateInteger should be unsigned");
    static_assert(std::is_unsigned<bit_integer_type>::value, "BitInteger should be unsigned");
    static_assert(
      std::is_same< ::ket::utility::meta::range_value_t<Qubits>, ::ket::utility::meta::range_value_t< ::ket::utility::meta::range_value_t<QubitsRange> > >::value,
      "Qubits' value_type and QubitsRange's value_type's value_type should be the same");

    using std::begin;
    using std::end;
    auto const register_size = static_cast<bit_integer_type>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
    assert(
      std::all_of(
        begin(rhs_qubits_range), end(rhs_qubits_range),
        [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
        { return register_size == static_cast<bit_integer_type>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

    ::ket::utility::generate_phase_coefficients(phase_coefficients, register_size);

    ::ket::swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);

    for (auto const& rhs_qubits: rhs_qubits_range)
      ::ket::addition_assignment_detail::addition_assignment(
        parallel_policy, first, last, begin(lhs_qubits), begin(rhs_qubits), register_size, phase_coefficients);

    ::ket::adj_swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);
  }

  template <typename RandomAccessIterator, typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
  inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessIterator>::value, void>
  addition_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector<
      typename std::iterator_traits<RandomAccessIterator>::value_type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  { ::ket::addition_assignment(::ket::utility::policy::make_sequential(), first, last, lhs_qubits, rhs_qubits_range, phase_coefficients); }

  template <typename ParallelPolicy, typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, void >
  addition_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  {
    using bit_integer_type = ::ket::meta::bit_integer_t< ::ket::utility::meta::range_value_t<Qubits> >;
    static_assert(std::is_unsigned< ::ket::meta::state_integer_t< ::ket::utility::meta::range_value_t<Qubits> > >::value, "StateInteger should be unsigned");
    static_assert(std::is_unsigned<bit_integer_type>::value, "BitInteger should be unsigned");
    static_assert(
      std::is_same< ::ket::utility::meta::range_value_t<Qubits>, ::ket::utility::meta::range_value_t< ::ket::utility::meta::range_value_t<QubitsRange> > >::value,
      "Qubits' value_type and QubitsRange's value_type's value_type should be the same");

    using std::begin;
    using std::end;
    auto const register_size = static_cast<bit_integer_type>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
    assert(
      std::all_of(
        begin(rhs_qubits_range), end(rhs_qubits_range),
        [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
        { return register_size == static_cast<bit_integer_type>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

    using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    auto phase_coefficients = ::ket::utility::generate_phase_coefficients<complex_type>(register_size);

    ::ket::addition_assignment(parallel_policy, first, last, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline void addition_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  { ::ket::addition_assignment(::ket::utility::policy::make_sequential(), first, last, lhs_qubits, rhs_qubits_range); }


  namespace ranges
  {
    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      using std::begin;
      using std::end;
      ::ket::addition_assignment(parallel_policy, begin(state), end(state), lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value, RandomAccessRange&>
    addition_assignment(
      RandomAccessRange& state, Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      using std::begin;
      using std::end;
      ::ket::addition_assignment(begin(state), end(state), lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <typename ParallelPolicy, typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
    addition_assignment(ParallelPolicy const parallel_policy, RandomAccessRange& state, Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      using std::begin;
      using std::end;
      ::ket::addition_assignment(parallel_policy, begin(state), end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline RandomAccessRange& addition_assignment(RandomAccessRange& state, Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      using std::begin;
      using std::end;
      ::ket::addition_assignment(begin(state), end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }
  } // namespace ranges


  namespace addition_assignment_detail
  {
    template <
      typename ParallelPolicy,
      typename RandomAccessIterator, typename Iterator1, typename Iterator2,
      typename BitInteger, typename PhaseCoefficientsAllocator>
    inline auto adj_addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      Iterator1 const& lhs_qubits_first, Iterator2 const& rhs_qubits_first,
      BitInteger const register_size,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator > const& phase_coefficients)
    -> void
    {
      for (auto index = BitInteger{0u}; index < register_size; ++index)
      {
        auto const phase_exponent = register_size - index;
        auto const phase_coefficient = phase_coefficients[phase_exponent];

        for (auto control_bit_index = BitInteger{0u};
             control_bit_index <= register_size - phase_exponent; ++control_bit_index)
        {
          auto const target_bit_index = control_bit_index + (phase_exponent - BitInteger{1u});

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
    typename ParallelPolicy,
    typename RandomAccessIterator, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline void adj_addition_assignment(
    ParallelPolicy const parallel_policy, RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator>& phase_coefficients)
  {
    using bit_integer_type = ::ket::meta::bit_integer_t< ::ket::utility::meta::range_value_t<Qubits> >;
    static_assert(std::is_unsigned< ::ket::meta::state_integer_t< ::ket::utility::meta::range_value_t<Qubits> > >::value, "StateInteger should be unsigned");
    static_assert(std::is_unsigned<bit_integer_type>::value, "BitInteger should be unsigned");
    static_assert(
      std::is_same< ::ket::utility::meta::range_value_t<Qubits>, ::ket::utility::meta::range_value_t< ::ket::utility::meta::range_value_t<QubitsRange> > >::value,
      "Qubits' value_type and QubitsRange's value_type's value_type should be the same");

    using std::begin;
    using std::end;
    auto const register_size = static_cast<bit_integer_type>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
    assert(
      std::all_of(
        begin(rhs_qubits_range), end(rhs_qubits_range),
        [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
        { return register_size == static_cast<bit_integer_type>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

    ::ket::utility::generate_phase_coefficients(phase_coefficients, register_size);

    ::ket::swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);

    for (auto const& rhs_qubits: rhs_qubits_range)
      ::ket::addition_assignment_detail::adj_addition_assignment(
        parallel_policy, first, last, begin(lhs_qubits), begin(rhs_qubits), register_size, phase_coefficients);

    ::ket::adj_swapped_fourier_transform(parallel_policy, first, last, lhs_qubits, phase_coefficients);
  }

  template <typename RandomAccessIterator, typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
  inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessIterator>::value, void>
  adj_addition_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator>& phase_coefficients)
  {
    ::ket::adj_addition_assignment(
      ::ket::utility::policy::make_sequential(),
      first, last, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <typename ParallelPolicy, typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, void >
  adj_addition_assignment(
    ParallelPolicy const parallel_policy, RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  {
    using bit_integer_type = ::ket::meta::bit_integer_t< ::ket::utility::meta::range_value_t<Qubits> >;
    static_assert(std::is_unsigned< ::ket::meta::state_integer_t< ::ket::utility::meta::range_value_t<Qubits> > >::value, "StateInteger should be unsigned");
    static_assert(std::is_unsigned<bit_integer_type>::value, "BitInteger should be unsigned");
    static_assert(
      std::is_same< ::ket::utility::meta::range_value_t<Qubits>, ::ket::utility::meta::range_value_t< ::ket::utility::meta::range_value_t<QubitsRange> > >::value,
      "Qubits' value_type and QubitsRange's value_type's value_type should be the same");

    using std::begin;
    using std::end;
    auto const register_size = static_cast<bit_integer_type>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
    assert(
      std::all_of(
        begin(rhs_qubits_range), end(rhs_qubits_range),
        [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
        { return register_size == static_cast<bit_integer_type>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

    using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    auto phase_coefficients = ::ket::utility::generate_phase_coefficients<complex_type>(register_size);

    ::ket::adj_addition_assignment(parallel_policy, first, last, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline void adj_addition_assignment(RandomAccessIterator const first, RandomAccessIterator const last, Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  { ::ket::adj_addition_assignment(::ket::utility::policy::make_sequential(), first, last, lhs_qubits, rhs_qubits_range); }


  namespace ranges
  {
    template <typename ParallelPolicy, typename RandomAccessRange, typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& adj_addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state, Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator>& phase_coefficients)
    {
      using std::begin;
      using std::end;
      ::ket::adj_addition_assignment(parallel_policy, begin(state), end(state), lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value, RandomAccessRange&>
    adj_addition_assignment(
      RandomAccessRange& state, Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator>& phase_coefficients)
    {
      using std::begin;
      using std::end;
      ::ket::adj_addition_assignment(begin(state), end(state), lhs_qubits, rhs_qubits_range, phase_coefficients);
      return state;
    }

    template <typename ParallelPolicy, typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
    adj_addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      using std::begin;
      using std::end;
      ::ket::adj_addition_assignment(parallel_policy, begin(state), end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline RandomAccessRange& adj_addition_assignment(RandomAccessRange& state, Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      using std::begin;
      using std::end;
      ::ket::adj_addition_assignment(begin(state), end(state), lhs_qubits, rhs_qubits_range);
      return state;
    }
  } // namespace ranges
} // namespace ket


#endif // KET_ADDITION_ASSIGNMENT_HPP
