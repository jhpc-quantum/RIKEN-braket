#ifndef KET_ALL_EXPECTATION_VALUES_HPP
# define KET_ALL_EXPECTATION_VALUES_HPP

# include <boost/config.hpp>

# include <cassert>
# include <iterator>
# include <vector>
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif

# include <boost/range/value_type.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>

# include <ket/spin_expectation_value.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/integer_log2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_exp2.hpp>
# endif
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define KET_array std::array
# else
#   define KET_array boost::array
# endif


namespace ket
{
  template <typename Qubit, typename ParallelPolicy, typename RandomAccessIterator>
  inline
  std::vector<
    KET_array<
      typename ::ket::utility::meta::real_of<
        typename std::iterator_traits<RandomAccessIterator>::value_type>::type, 3u> >
  all_spin_expectation_values(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last)
  {
    typedef typename ::ket::meta::bit_integer_of<Qubit>::type bit_integer_type;
    bit_integer_type const num_qubits = ::ket::utility::integer_log2<bit_integer_type>(last-first);
    assert(
      ::ket::utility::integer_exp2<bit_integer_type>(num_qubits)
        == static_cast<bit_integer_type>(last-first));

    typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
    typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
    typedef KET_array<real_type, 3u> spin_type;
    std::vector<spin_type> result;
    result.reserve(num_qubits);

    Qubit const last_qubit(num_qubits);
    for (Qubit qubit = static_cast<Qubit>(static_cast<bit_integer_type>(0u));
         qubit < last_qubit; ++qubit)
      result.push_back(
        ::ket::spin_expectation_value(
          parallel_policy, first, last, qubit, permutation);

    return result;
  }

  template <typename Qubit, typename RandomAccessIterator>
  inline
  std::vector<
    KET_array<
      typename ::ket::utility::meta::real_of<
        typename std::iterator_traits<RandomAccessIterator>::value_type>::type, 3u> >
  all_spin_expectation_values(
    RandomAccessIterator const first, RandomAccessIterator const last)
  {
    return ::ket::all_spin_expectation_values<Qubit>(
      ::ket::utility::policy::make_sequential(), first, last);
  }

  namespace ranges
  {
    template <typename Qubit, typename ParallelPolicy, typename RandomAccessRange>
    inline
    std::vector<
      KET_array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<RandomAccessRange>::type>::type, 3u> >
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy, RandomAccessRange& state)
    {
      return ::ket::all_spin_expectation_values<Qubit>(
        parallel_policy, boost::begin(state), boost::end(state));
    }

    template <typename Qubit, typename RandomAccessRange>
    inline
    std::vector<
      KET_array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<RandomAccessRange>::type>::type, 3u> >
    all_spin_expectation_values(RandomAccessRange& state)
    { return ::ket::all_spin_expectation_values<Qubit>(boost::begin(state), boost::end(state)); }
  } // namespace ranges
} // namespace ket


#endif
