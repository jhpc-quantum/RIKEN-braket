#ifndef KET_ALL_EXPECTATION_VALUES_HPP
# define KET_ALL_EXPECTATION_VALUES_HPP

# include <cassert>
# include <vector>
# include <array>
# include <iterator>

# include <ket/spin_expectation_value.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/integer_log2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_exp2.hpp>
# endif
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  template <typename Qubit, typename ParallelPolicy, typename RandomAccessIterator>
  inline auto all_spin_expectation_values(ParallelPolicy const parallel_policy, RandomAccessIterator const first, RandomAccessIterator const last)
  -> std::vector<std::array< ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>, 3u >>
  {
    using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;
    auto const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(last - first);
    assert(
      ::ket::utility::integer_exp2<bit_integer_type>(num_qubits)
        == static_cast<bit_integer_type>(last - first));

    using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    using real_type = ::ket::utility::meta::real_t<complex_type>;
    using spin_type = std::array<real_type, 3u>;
    auto result = std::vector<spin_type>{};
    result.reserve(num_qubits);

    auto const last_qubit = Qubit{num_qubits};
    for (auto qubit = Qubit{bit_integer_type{0u}}; qubit < last_qubit; ++qubit)
      result.push_back(::ket::spin_expectation_value(parallel_policy, first, last, qubit));

    return result;
  }

  template <typename Qubit, typename RandomAccessIterator>
  inline auto all_spin_expectation_values(RandomAccessIterator const first, RandomAccessIterator const last)
  -> std::vector<std::array< ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator>::value_type>, 3u >>
  { return ::ket::all_spin_expectation_values<Qubit>(::ket::utility::policy::make_sequential(), first, last); }

  namespace ranges
  {
    template <typename Qubit, typename ParallelPolicy, typename RandomAccessRange>
    inline auto all_spin_expectation_values(ParallelPolicy const parallel_policy, RandomAccessRange const& state)
    -> std::vector<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange> >, 3u >>
    { using std::begin; using std::end; return ::ket::all_spin_expectation_values<Qubit>(parallel_policy, begin(state), end(state)); }

    template <typename Qubit, typename RandomAccessRange>
    inline auto all_spin_expectation_values(RandomAccessRange const& state)
    -> std::vector<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange> >, 3u >>
    { using std::begin; using std::end; return ::ket::all_spin_expectation_values<Qubit>(begin(state), end(state)); }
  } // namespace ranges
} // namespace ket


#endif // KET_ALL_EXPECTATION_VALUES_HPP
