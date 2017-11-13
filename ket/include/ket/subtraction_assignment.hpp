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
    typename RandomAccessRange, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline RandomAccessRange& subtraction_assignment(
    ParallelPolicy const parallel_policy, RandomAccessRange& state,
    Qubits const& lhs_qubits, Qubits const& rhs_qubits_range,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    using ::ket::adj_addition_assignment;
    return adj_addition_assignment(
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
    return subtraction_assignment(
      ::ket::utility::policy::make_sequential(),
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
    using ::ket::adj_addition_assignment;
    return adj_addition_assignment(
      parallel_policy, state, lhs_qubits, rhs_qubits_range);
  }

  template <
    typename RandomAccessRange, typename Qubits, typename QubitsRange>
  inline RandomAccessRange& subtraction_assignment(
    RandomAccessRange& state,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  {
    return subtraction_assignment(
      ::ket::utility::policy::make_sequential(),
      state, lhs_qubits, rhs_qubits_range);
  }


  template <
    typename ParallelPolicy,
    typename RandomAccessRange, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline RandomAccessRange& adj_subtraction_assignment(
    ParallelPolicy const parallel_policy, RandomAccessRange& state,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    using ::ket::addition_assignment;
    return addition_assignment(
      parallel_policy, state, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }

  template <
    typename RandomAccessRange, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline typename KET_enable_if<
    not ::ket::utility::policy::is_loop_n_policy<RandomAccessRange>::value,
    RandomAccessRange&>::type
  adj_subtraction_assignment(
    RandomAccessRange& state,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
    std::vector<
      typename boost::range_value<RandomAccessRange>::type,
      PhaseCoefficientsAllocator>& phase_coefficients)
  {
    return adj_subtraction_assignment(
      ::ket::utility::policy::make_sequential(),
      state, lhs_qubits, rhs_qubits_range, phase_coefficients);
  }


  template <
    typename ParallelPolicy,
    typename RandomAccessRange, typename Qubits, typename QubitsRange>
  inline typename KET_enable_if<
    ::ket::utility::policy::is_loop_n_policy<ParallelPolicy>::value,
    RandomAccessRange&>::type
  adj_subtraction_assignment(
    ParallelPolicy const parallel_policy, RandomAccessRange& state,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  {
    using ::ket::addition_assignment;
    return addition_assignment(
      parallel_policy, state, lhs_qubits, rhs_qubits_range);
  }

  template <
    typename RandomAccessRange, typename Qubits, typename QubitsRange,
    typename PhaseCoefficientsAllocator>
  inline RandomAccessRange& adj_subtraction_assignment(
    RandomAccessRange& state,
    Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
  {
    return adj_subtraction_assignment(
      ::ket::utility::policy::make_sequential(),
      state, lhs_qubits, rhs_qubits_range);
  }
}


# undef KET_enable_if

#endif

