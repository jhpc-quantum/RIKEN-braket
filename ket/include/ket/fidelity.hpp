#ifndef KET_FIDELITY_HPP
# define KET_FIDELITY_HPP

# include <iterator>
# include <utility>

# include <ket/qubit.hpp>
# include <ket/inner_product.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  // |<Psi_2|Psi_1>|^2
  template <typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2>
  inline auto fidelity(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2)
  -> ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator1>::value_type>
  {
    using std::norm;
    return norm(::ket::inner_product(parallel_policy, first1, last1, first2));
  }

  template <typename RandomAccessIterator1, typename RandomAccessIterator2>
  inline auto fidelity(
    RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2)
  -> ::ket::utility::meta::real_t<typename std::iterator_traits<RandomAccessIterator1>::value_type>
  { return ::ket::fidelity(::ket::utility::policy::make_sequential(), first1, last1, first2); }

  namespace ranges
  {
    template <typename ParallelPolicy, typename RandomAccessRange1, typename RandomAccessRange2>
    inline auto fidelity(
      ParallelPolicy const parallel_policy,
      RandomAccessRange1 const& state1, RandomAccessRange2 const& state2)
    -> ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange1> >
    { using std::begin; using std::end; return ::ket::fidelity(parallel_policy, begin(state1), end(state1), begin(state2)); }

    template <typename RandomAccessRange1, typename RandomAccessRange2>
    inline auto fidelity(RandomAccessRange1 const& state1, RandomAccessRange2 const& state2)
    -> ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange1> >
    { using std::begin; using std::end; return ::ket::fidelity(begin(state1), end(state1), begin(state2)); }
  } // namespace ranges

  // |<Psi_2| A_{ij} |Psi_1>|^2
  template <
    typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2,
    typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
  inline auto fidelity(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2,
    Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
  -> ::ket::utility::meta::real_t< typename std::iterator_traits<RandomAccessIterator1>::value_type >
  {
    using std::norm;
    return norm(::ket::inner_product(parallel_policy, first1, last1, first2, std::forward<Observable>(observable), qubit, qubits...));
  }

  template <
    typename RandomAccessIterator1, typename RandomAccessIterator2,
    typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
  inline auto fidelity(
    RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2,
    Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
  -> ::ket::utility::meta::real_t< typename std::iterator_traits<RandomAccessIterator1>::value_type >
  { return ::ket::fidelity(::ket::utility::policy::make_sequential(), first1, last1, first2, std::forward<Observable>(observable), qubit, qubits...); }

  namespace ranges
  {
    template <
      typename ParallelPolicy, typename RandomAccessRange1, typename RandomAccessRange2,
      typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto fidelity(
      ParallelPolicy const parallel_policy,
      RandomAccessRange1 const& state1, RandomAccessRange2 const& state2,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    -> ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange1> >
    { using std::begin; using std::end; return ::ket::fidelity(parallel_policy, begin(state1), end(state1), begin(state2), std::forward<Observable>(observable), qubit, qubits...); }

    template <
      typename RandomAccessRange1, typename RandomAccessRange2,
      typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto fidelity(
      RandomAccessRange1 const& state1, RandomAccessRange2 const& state2,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    -> ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange1> >
    { using std::begin; using std::end; return ::ket::fidelity(begin(state1), end(state1), begin(state2), std::forward<Observable>(observable), qubit, qubits...); }
  } // namespace ranges

  namespace runtime
  {
    namespace qubit_ranges
    {
      // |<Psi_2| A_{ij} |Psi_1>|^2
      template <typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Observable, typename QubitsRange>
      inline auto fidelity(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2,
        Observable&& observable, QubitsRange const& qubits)
      -> ::ket::utility::meta::real_t< typename std::iterator_traits<RandomAccessIterator1>::value_type >
      {
        using std::norm;
        return norm(::ket::runtime::qubit_ranges::inner_product(
          parallel_policy, first1, last1, first2, std::forward<Observable>(observable), qubits));
      }

      template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename Observable, typename QubitsRange>
      inline auto fidelity(
        RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2,
        Observable&& observable, QubitsRange const& qubits)
      -> ::ket::utility::meta::real_t< typename std::iterator_traits<RandomAccessIterator1>::value_type >
      {
        return ::ket::runtime::qubit_ranges::fidelity(
          ::ket::utility::policy::make_sequential(),
          first1, last1, first2, std::forward<Observable>(observable), qubits);
      }
    } // namespace qubit_ranges

    template <typename ParallelPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2, typename Observable, typename QubitIterator>
    inline auto fidelity(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2,
      Observable&& observable, QubitIterator const qubit_first, QubitIterator const qubit_last)
    -> ::ket::utility::meta::real_t< typename std::iterator_traits<RandomAccessIterator1>::value_type >
    {
      using std::norm;
      return norm(::ket::runtime::inner_product(
        parallel_policy, first1, last1, first2, std::forward<Observable>(observable), qubit_first, qubit_last));
    }

    template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename Observable, typename QubitIterator>
    inline auto fidelity(
      RandomAccessIterator1 const first1, RandomAccessIterator1 const last1, RandomAccessIterator2 const first2,
      Observable&& observable, QubitIterator const qubit_first, QubitIterator const qubit_last)
    -> ::ket::utility::meta::real_t< typename std::iterator_traits<RandomAccessIterator1>::value_type >
    {
      return ::ket::runtime::fidelity(
        ::ket::utility::policy::make_sequential(),
        first1, last1, first2, std::forward<Observable>(observable), qubit_first, qubit_last);
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange1, typename RandomAccessRange2, typename Observable, typename QubitsRange>
      inline auto fidelity(
        ParallelPolicy const parallel_policy,
        RandomAccessRange1 const& state1, RandomAccessRange2 const& state2,
        Observable&& observable, QubitsRange const& qubits)
      -> ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange1> >
      {
        using std::norm;
        return norm(::ket::runtime::ranges::inner_product(
          parallel_policy, state1, state2, std::forward<Observable>(observable), qubits));
      }

      template <typename RandomAccessRange1, typename RandomAccessRange2, typename Observable, typename QubitsRange>
      inline auto fidelity(
        RandomAccessRange1 const& state1, RandomAccessRange2 const& state2,
        Observable&& observable, QubitsRange const& qubits)
      -> ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange1> >
      {
        return ::ket::runtime::ranges::fidelity(
          ::ket::utility::policy::make_sequential(),
          state1, state2, std::forward<Observable>(observable), qubits);
      }
    } // namespace ranges
  } // namespace runtime
} // namespace ket


#endif // KET_FIDELITY_HPP
