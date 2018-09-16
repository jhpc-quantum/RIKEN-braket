#ifndef KET_SUBTRACTION_ASSIGNMENT_HPP
# define KET_SUBTRACTION_ASSIGNMENT_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <vector>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif

# include <boost/range/value_type.hpp>

# include <ket/qubit.hpp>
# include <ket/addition_assignment.hpp>
# include <ket/utility/loop_n.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_enable_if std::enable_if
# else
#   define KET_enable_if boost::enable_if_c
# endif


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
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    ::ket::adj_addition_assignment(
      parallel_policy, first, last, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <
    typename RandomAccessIterator, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline typename KET_enable_if<
    not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessIterator>::value,
    void>::type
  subtraction_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    ::ket::adj_addition_assignment(
      first, last, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline typename KET_enable_if<
    ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
    void>::type
  subtraction_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, Qubits const& rhs_qubits_range)
  {
    ::ket::adj_addition_assignment(
      parallel_policy, first, last, lhs_qubits, rhs_qubits_range);
  }

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
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      return ::ket::adj_addition_assignment(
        parallel_policy, state, lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator>
    inline typename KET_enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value,
      RandomAccessRange&>::type
    subtraction_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      return ::ket::adj_addition_assignment(
        state, lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    subtraction_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, Qubits const& rhs_qubits_range)
    {
      return ::ket::adj_addition_assignment(
        parallel_policy, state, lhs_qubits, rhs_qubits_range);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline RandomAccessRange& subtraction_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    { return ::ket::adj_addition_assignment(state, lhs_qubits, rhs_qubits_range); }
  }


  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline void adj_subtraction_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, Qubits const& rhs_qubits_range,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    ::ket::addition_assignment(
      parallel_policy, first, last, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <
    typename RandomAccessIterator, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline typename KET_enable_if<
    not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessIterator>::value,
    void>::type
  adj_subtraction_assignment(
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    ::ket::addition_assignment(
      first, last, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename Qubits, typename QubitsRange>
  inline typename KET_enable_if<
    ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
    void>::type
  adj_subtraction_assignment(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    Qubits const& lhs_qubits, Qubits const& rhs_qubits_range)
  {
    ::ket::addition_assignment(
      parallel_policy, first, last, lhs_qubits, rhs_qubits_range);
  }

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
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      return ::ket::addition_assignment(
        parallel_policy, state, lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator>
    inline typename KET_enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value,
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients)
    {
      return ::ket::addition_assignment(
        state, lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& state,
      Qubits const& lhs_qubits, Qubits const& rhs_qubits_range)
    {
      return ::ket::addition_assignment(
        parallel_policy, state, lhs_qubits, rhs_qubits_range);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange>
    inline RandomAccessRange& adj_subtraction_assignment(
      RandomAccessRange& state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    { return ::ket::addition_assignment(state, lhs_qubits, rhs_qubits_range); }
  }
}


# undef KET_enable_if

#endif

