#ifndef KET_SWAPPED_FOURIER_TRANSFORM_HPP
# define KET_SWAPPED_FOURIER_TRANSFORM_HPP

# include <cassert>
# include <cstddef>
# include <cmath>
# include <iterator>
# include <type_traits>

# include <ket/control.hpp>
# include <ket/gate/hadamard.hpp>
# include <ket/gate/controlled_phase_shift.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/is_unique_if_sorted.hpp>
# endif
# include <ket/utility/generate_phase_coefficients.hpp>
# include <ket/utility/meta/ranges.hpp>


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
      ::ket::utility::meta::range_value_t<RandomAccessRange>,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    using std::begin;
    using std::end;
    auto const qubits_first = begin(qubits);

    using qubit_type = ::ket::utility::meta::range_value_t<Qubits>;
    using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;
    auto const num_qubits = static_cast<bit_integer_type>(std::distance(qubits_first, end(qubits)));
    ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

    static_assert(
      std::is_unsigned< ::ket::meta::state_integer_t<qubit_type> >::value,
      "StateInteger should be unsigned");
    static_assert(std::is_unsigned<bit_integer_type>::value, "BitInteger should be unsigned");
    assert(
      ::ket::utility::integer_exp2< ::ket::meta::state_integer_t<qubit_type> >(num_qubits)
        <= static_cast< ::ket::meta::state_integer_t<qubit_type> >(std::distance(begin(state), end(state)))
      and ::ket::utility::ranges::is_unique_if_sorted(qubits));

    for (auto index = std::size_t{0u}; index < num_qubits; ++index)
    {
      auto target_bit = num_qubits - index - std::size_t{1u};

      ::ket::gate::hadamard(parallel_policy, first, last, qubits_first[target_bit]);

      for (auto phase_exponent = std::size_t{2u};
           phase_exponent <= num_qubits - index; ++phase_exponent)
      {
        auto const control_bit = target_bit - (phase_exponent - std::size_t{1u});

        ::ket::gate::controlled_phase_shift(
          parallel_policy,
          first, last, phase_coefficients[phase_exponent],
          qubits_first[target_bit], ::ket::make_control(qubits_first[control_bit]));
      }
    }
  }

  template <typename RandomAccessIterator, typename Qubits, typename PhaseCoefficientsAllocator>
  inline std::enable_if_t<not ::ket::utility::policy::is_loop_n_policy<RandomAccessIterator>::value, void>
  swapped_fourier_transform(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits,
    std::vector<
      ::ket::utility::meta::range_value_t<RandomAccessRange>,
      PhaseCoefficientsAllocator>& phase_coefficients)
  { ::ket::swapped_fourier_transform(::ket::utility::policy::make_sequential(), first, last, qubits, phase_coefficients); }

  template <typename ParallelPolicy, typename RandomAccessIterator, typename Qubits>
  inline std::enable_if_t< ::ket::utility::policy::is_loop_n_policy<ParallelPolicy>::value, void >
  swapped_fourier_transform(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits)
  {
    using std::begin;
    using std::end;
    using complex_type = std::iterator_traits<RandomAccessIterator>::value_type;
    auto phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(std::distance(begin(lhs_qubits), end(lhs_qubits)));

    ::ket::swapped_fourier_transform(parallel_policy, first, last, qubits, phase_coefficients);
  }

  template <typename ParallelPolicy, typename RandomAccessIterator, typename Qubits>
  inline void swapped_fourier_transform(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits)
  { ::ket::swapped_fourier_transform(::ket::utility::policy::make_sequential(), first, last, qubits); }


  namespace ranges
  {
    template <
      typename ParallelPolicy, typename RandomAccessRange, typename Qubits,
      typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& qubits,
      std::vector<
        ::ket::utility::meta::range_value_t<RandomAccessRange>,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      using std::begin;
      using std::end;
      ::ket::swapped_fourier_transform(parallel_policy, begin(state), end(state), qubits, phase_coefficients);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      not ::ket::utility::policy::is_loop_n_policy<RandomAccessRange>::value, RandomAccessRange&>
    swapped_fourier_transform(
      RandomAccessRange& state, Qubits const& qubits,
      std::vector<
        ::ket::utility::meta::range_value_t<RandomAccessRange>,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      using std::begin;
      using std::end;
      ::ket::swapped_fourier_transform(begin(state), end(state), qubits, phase_coefficients);
      return state;
    }

    template <typename ParallelPolicy, typename RandomAccessRange, typename Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange&>
    swapped_fourier_transform(ParallelPolicy const parallel_policy, RandomAccessRange& state, Qubits const& qubits)
    {
      using std::begin;
      using std::end;
      ::ket::swapped_fourier_transform(parallel_policy, begin(state), end(state), qubits);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& swapped_fourier_transform(RandomAccessRange& state, Qubits const& qubits)
    {
      using std::begin;
      using std::end;
      ::ket::swapped_fourier_transform(begin(state), end(state), qubits);
      return state;
    }
  } // namespace ranges


  template <
    typename ParallelPolicy, typename RandomAccessIterator, typename Qubits,
    typename PhaseCoefficientsAllocator>>
  inline void adj_swapped_fourier_transform(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits,
    std::vector<
      ::ket::utility::meta::range_value_t<RandomAccessRange>,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    using std::begin;
    auto const qubits_first = begin(qubits);

    using qubit_type = ::ket::utility::meta::range_value_t<Qubits>;
    using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;
    using std::end;
    auto const num_qubits = static_cast<bit_integer_type>(std::distance(qubits_first, end(qubits)));
    ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

    static_assert(
      std::is_unsigned< ::ket::meta::state_integer_t<qubit_type> >::value,
      "StateInteger should be unsigned");
    static_assert(std::is_unsigned<bit_integer_type>::value, "BitInteger should be unsigned");
    assert(
      ::ket::utility::integer_exp2< ::ket::meta::state_integer_t<qubit_type> >(num_qubits)
        <= static_cast< ::ket::meta::state_integer_t<qubit_type> >(std::distance(begin(state), end(state)))
      and ::ket::utility::ranges::is_unique_if_sorted(qubits));

    for (auto target_bit = std::size_t{0u}; target_bit < num_qubits; ++target_bit)
    {
      for (auto index = std::size_t{0u}; index < target_bit; ++index)
      {
        auto const phase_exponent = std::size_t{1u} + target_bit - index;
        auto const control_bit = target_bit - (phase_exponent - std::size_t{1u});

        ::ket::gate::adj_controlled_phase_shift(
          parallel_policy,
          first, last, phase_coefficients[phase_exponent], qubits_first[target_bit],
          ::ket::make_control(qubits_first[control_bit]));
      }

      ::ket::gate::adj_hadamard(parallel_policy, first, last, qubits_first[target_bit]);
    }
  }

  template <typename RandomAccessIterator, typename Qubits, typename PhaseCoefficientsAllocator>
  inline std::enable_if_t<not ::ket::utility::policy::is_loop_n_policy<RandomAccessIterator>::value, void>
  adj_swapped_fourier_transform(
    RandomAccessIterator const first, RandomAccessIterator const last, Qubits const& qubits,
    std::vector<
      ::ket::utility::meta::range_value_t<RandomAccessRange>,
      PhaseCoefficientsAllocator>& phase_coefficients)
  { ::ket::adj_swapped_fourier_transform(::ket::utility::policy::make_sequential(), first, last, qubits, phase_coefficients); }

  template <
    typename ParallelPolicy, typename RandomAccessIterator, typename Qubits>
  inline std::enable_if_t< ::ket::utility::policy::is_loop_n_policy<ParallelPolicy>::value, void >
  adj_swapped_fourier_transform(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last, Qubits const& qubits)
  {
    using std::begin;
    using std::end;
    using bit_integer_type = ::ket::meta::bit_integer_t< ::ket::utility::meta::range_value_t<Qubits> >;
    using complex_type = std::iterator_traits<RandomAccessIterator>::value_type;
    auto phase_coefficients
      = ::ket::utility::generate_phase_coefficients<complex_type>(
          static_cast<bit_integer_type>(std::distance(begin(qubits), end(qubits))));

    ::ket::adj_swapped_fourier_transform(parallel_policy, first, last, qubits, phase_coefficients);
  }

  template <
    typename ParallelPolicy, typename RandomAccessIterator, typename Qubits>
  inline void adj_swapped_fourier_transform(
    RandomAccessIterator const first, RandomAccessIterator const last, Qubits const& qubits)
  { ::ket::adj_swapped_fourier_transform(::ket::utility::policy::make_sequential(), first, last, qubits); }


  namespace ranges
  {
    template <
      typename ParallelPolicy, typename RandomAccessRange, typename Qubits,
      typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& qubits,
      std::vector<
        ::ket::utility::meta::range_value_t<RandomAccessRange>,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      using std::begin;
      using std::end;
      ::ket::adj_swapped_fourier_transform(parallel_policy, begin(state), end(state), qubits, phase_coefficients);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<not ::ket::utility::policy::is_loop_n_policy<RandomAccessRange>::value, RandomAccessRange&>
    adj_swapped_fourier_transform(
      RandomAccessRange& state, Qubits const& qubits,
      std::vector<
        ::ket::utility::meta::range_value_t<RandomAccessRange>,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      using std::begin;
      using std::end;
      ::ket::adj_swapped_fourier_transform(begin(state), end(state), qubits, phase_coefficients);
      return state;
    }

    template <typename ParallelPolicy, typename RandomAccessRange, typename Qubits>
    inline std::enable_if_t< ::ket::utility::policy::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
    adj_swapped_fourier_transform(ParallelPolicy const parallel_policy, RandomAccessRange& state, Qubits const& qubits)
    {
      using std::begin;
      using std::end;
      ::ket::adj_swapped_fourier_transform(parallel_policy, begin(state), end(state), qubits);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& adj_swapped_fourier_transform(RandomAccessRange& state, Qubits const& qubits)
    {
      using std::begin;
      using std::end;
      ::ket::adj_swapped_fourier_transform(begin(state), end(state), qubits);
      return state;
    }
  } // namespace ranges
} // namespace ket


#endif // KET_SWAPPED_FOURIER_TRANSFORM_HPP
