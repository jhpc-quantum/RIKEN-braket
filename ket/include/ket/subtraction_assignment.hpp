#ifndef KET_SUBTRACTION_ASSIGNMENT_HPP
# define KET_SUBTRACTION_ASSIGNMENT_HPP

# include <cstddef>
# include <vector>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/addition_assignment.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  // lhs -= rhs
  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline void subtraction_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, Qubits const& rhs_qubits_range,
    std::vector<
      ::ket::utility::meta::range_value_t<RandomAccessRange>,
      PhaseCoefficientsAllocator>& phase_coefficients)
  { ::ket::adj_addition_assignment(parallel_policy, first, last, lhs_qubits, rhs_qubits_range, phase_coefficients); }

  template <
    typename RandomAccessIterator, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessIterator>::value, void>
  subtraction_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector<
      ::ket::utility::meta::range_value_t<RandomAccessRange>,
      PhaseCoefficientsAllocator>& phase_coefficients)
  { ::ket::adj_addition_assignment(first, last, lhs_qubits, rhs_qubits_range, phase_coefficients); }

  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, void >
  subtraction_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, Qubits const& rhs_qubits_range)
  { ::ket::adj_addition_assignment(parallel_policy, first, last, lhs_qubits, rhs_qubits_range); }

  template <typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline void subtraction_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  { ::ket::adj_addition_assignment(first, last, lhs_qubits, rhs_qubits_range); }


  namespace ranges
  {
    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& subtraction_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, Qubits const& rhs_qubits_range,
      std::vector<
        ::ket::utility::meta::range_value_t<RandomAccessRange>,
        PhaseCoefficientsAllocator>& phase_coefficients)
    { return ::ket::adj_addition_assignment(parallel_policy, state, lhs_qubits, rhs_qubits_range, phase_coefficients); }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value,
      RandomAccessRange&>
    subtraction_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        ::ket::utility::meta::range_value_t<RandomAccessRange>,
        PhaseCoefficientsAllocator>& phase_coefficients)
    { return ::ket::adj_addition_assignment(state, lhs_qubits, rhs_qubits_range, phase_coefficients); }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    subtraction_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, Qubits const& rhs_qubits_range)
    { return ::ket::adj_addition_assignment(parallel_policy, state, lhs_qubits, rhs_qubits_range); }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline RandomAccessRange& subtraction_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    { return ::ket::adj_addition_assignment(state, lhs_qubits, rhs_qubits_range); }
  } // namespace ranges


  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline void adj_subtraction_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, Qubits const& rhs_qubits_range,
    std::vector<
      ::ket::utility::meta::range_value_t<RandomAccessRange>,
      PhaseCoefficientsAllocator>& phase_coefficients)
  { ::ket::addition_assignment(parallel_policy, first, last, lhs_qubits, rhs_qubits_range, phase_coefficients); }

  template <
    typename RandomAccessIterator, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline std::enable_if_t<not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessIterator>::value, void>
  adj_subtraction_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector<
      ::ket::utility::meta::range_value_t<RandomAccessRange>,
      PhaseCoefficientsAllocator>& phase_coefficients)
  { ::ket::addition_assignment(first, last, lhs_qubits, rhs_qubits_range, phase_coefficients); }

  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, void >
  adj_subtraction_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, Qubits const& rhs_qubits_range)
  { ::ket::addition_assignment(parallel_policy, first, last, lhs_qubits, rhs_qubits_range); }

  template <typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline void adj_subtraction_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  { ::ket::addition_assignment(first, last, lhs_qubits, rhs_qubits_range); }


  namespace ranges
  {
    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator>
    inline RandomAccessRange& adj_subtraction_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, Qubits const& rhs_qubits_range,
      std::vector<
        ::ket::utility::meta::range_value_t<RandomAccessRange>,
        PhaseCoefficientsAllocator>& phase_coefficients)
    { return ::ket::addition_assignment(parallel_policy, state, lhs_qubits, rhs_qubits_range, phase_coefficients); }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value,
      RandomAccessRange&>
    adj_subtraction_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        ::ket::utility::meta::range_value_t<RandomAccessRange>,
        PhaseCoefficientsAllocator>& phase_coefficients)
    { return ::ket::addition_assignment(state, lhs_qubits, rhs_qubits_range, phase_coefficients); }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, Qubits const& rhs_qubits_range)
    { return ::ket::addition_assignment(parallel_policy, state, lhs_qubits, rhs_qubits_range); }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline RandomAccessRange& adj_subtraction_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    { return ::ket::addition_assignment(state, lhs_qubits, rhs_qubits_range); }
  } // namespace ranges
} // namespace ket


#endif // KET_SUBTRACTION_ASSIGNMENT_HPP
