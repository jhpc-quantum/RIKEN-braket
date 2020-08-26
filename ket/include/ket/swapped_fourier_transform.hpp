#ifndef KET_SWAPPED_FOURIER_TRANSFORM_HPP
# define KET_SWAPPED_FOURIER_TRANSFORM_HPP

# include <cassert>
# include <cstddef>
# include <cmath>
# include <iterator>
# include <type_traits>

# include <boost/range/size.hpp>
# include <boost/range/value_type.hpp>

# include <ket/control.hpp>
# include <ket/gate/hadamard.hpp>
# include <ket/gate/controlled_phase_shift.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/is_unique_if_sorted.hpp>
# endif
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/generate_phase_coefficients.hpp>


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
    auto const num_qubits = boost::size(qubits);
    ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

    using qubit_type = typename boost::range_value<Qubits>::type;

    static_assert(
      std::is_unsigned<typename ::ket::meta::state_integer_of<qubit_type>::type>::value,
      "StateInteger should be unsigned");
    static_assert(
      std::is_unsigned<typename ::ket::meta::bit_integer_of<qubit_type>::type>::value,
      "BitInteger should be unsigned");
    assert(
      ::ket::utility::integer_exp2<typename ::ket::meta::state_integer_of<qubit_type>::type>(num_qubits)
        <= static_cast<typename ::ket::meta::state_integer_of<qubit_type>::type>(boost::size(state))
      and ::ket::utility::ranges::is_unique_if_sorted(qubits));

    auto const qubits_first = ::ket::utility::begin(qubits);

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
  inline typename std::enable_if<
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
  inline typename std::enable_if<
    ::ket::utility::policy::is_loop_n_policy<ParallelPolicy>::value,
    void>::type
  swapped_fourier_transform(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits)
  {
    using complex_type = std::iterator_traits<RandomAccessIterator>::value_type;
    auto phase_coefficients
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
    inline typename std::enable_if<
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
    inline typename std::enable_if<
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
  } // namespace ranges


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
    auto const num_qubits = boost::size(qubits);
    ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

    using qubit_type = typename boost::range_value<Qubits>::type;

    static_assert(
      std::is_unsigned<typename ::ket::meta::state_integer_of<qubit_type>::type>::value,
      "StateInteger should be unsigned");
    static_assert(
      std::is_unsigned<typename ::ket::meta::bit_integer_of<qubit_type>::type>::value,
      "BitInteger should be unsigned");
    assert(
      ::ket::utility::integer_exp2<typename ::ket::meta::state_integer_of<qubit_type>::type>(num_qubits)
        <= static_cast<typename ::ket::meta::state_integer_of<qubit_type>::type>(boost::size(state))
      and ::ket::utility::ranges::is_unique_if_sorted(qubits));

    auto const qubits_first = ::ket::utility::begin(qubits);

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
  inline typename std::enable_if<
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
  inline typename std::enable_if<
    ::ket::utility::policy::is_loop_n_policy<ParallelPolicy>::value,
    void>::type
  adj_swapped_fourier_transform(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& qubits)
  {
    using complex_type = std::iterator_traits<RandomAccessIterator>::value_type;
    auto phase_coefficients
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
        ::ket::utility::begin(state), ::ket::utility::end(state), qubits, phase_coefficients);
      return state;
    }

    template <typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator>
    inline typename std::enable_if<
      not ::ket::utility::policy::is_loop_n_policy<RandomAccessRange>::value,
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      RandomAccessRange& state, Qubits const& qubits,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      ::ket::adj_swapped_fourier_transform(
        ::ket::utility::begin(state), ::ket::utility::end(state), qubits, phase_coefficients);
      return state;
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange, typename Qubits>
    inline typename std::enable_if<
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
  } // namespace ranges
} // namespace ket


#endif // KET_SWAPPED_FOURIER_TRANSFORM_HPP
