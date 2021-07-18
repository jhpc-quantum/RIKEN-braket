#ifndef KET_GATE_PAULI_Z_HPP
# define KET_GATE_PAULI_Z_HPP

# include <boost/config.hpp>

# include <array>
# include <iterator>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::gate(
        parallel_policy, first, last,
        [](RandomAccessIterator const first, std::array<StateInteger, 2u> const& indices, int const)
        {
          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
# ifndef BOOST_NO_CXX14_BINARY_LITERALS
          *(first + indices[0b1u]) *= real_type{-1};
# else // BOOST_NO_CXX14_BINARY_LITERALS
          *(first + indices[1u]) *= real_type{-1};
# endif // BOOST_NO_CXX14_BINARY_LITERALS
        },
        qubit);
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_z(::ket::utility::policy::make_sequential(), first, last, qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& pauli_z(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::pauli_z(parallel_policy, std::begin(state), std::end(state), qubit);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& pauli_z(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_z(::ket::utility::policy::make_sequential(), state, qubit); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_z(parallel_policy, first, last, qubit); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_z(first, last, qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_pauli_z(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_z(parallel_policy, state, qubit); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_pauli_z(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_z(state, qubit); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_PAULI_Z_HPP
