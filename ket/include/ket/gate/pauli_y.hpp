#ifndef KET_GATE_PAULI_Y_HPP
# define KET_GATE_PAULI_Y_HPP

# include <boost/config.hpp>

# include <array>
# include <iterator>
# include <algorithm>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/imaginary_unit.hpp>


namespace ket
{
  namespace gate
  {
    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::gate(
        parallel_policy, first, last,
        [](RandomAccessIterator const first, std::array<StateInteger, 2u> const& indices, int const)
        {
# ifndef BOOST_NO_CXX14_BINARY_LITERALS
          auto const zero_iter = first + indices[0b0u];
          auto const one_iter = first + indices[0b1u];
# else // BOOST_NO_CXX14_BINARY_LITERALS
          auto const zero_iter = first + indices[0u];
          auto const one_iter = first + indices[1u];
# endif // BOOST_NO_CXX14_BINARY_LITERALS
          std::iter_swap(zero_iter, one_iter);

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *zero_iter *= -::ket::utility::imaginary_unit<complex_type>();
          *one_iter *= ::ket::utility::imaginary_unit<complex_type>();
        },
        qubit);
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void pauli_y(
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_y(::ket::utility::policy::make_sequential(), first, last, qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& pauli_y(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::pauli_y(parallel_policy, std::begin(state), std::end(state), qubit);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& pauli_y(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_y(::ket::utility::policy::make_sequential(), state, qubit); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_y(parallel_policy, first, last, qubit); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_pauli_y(
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_y(first, last, qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_pauli_y(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_y(parallel_policy, state, qubit); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_pauli_y(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_y(state, qubit); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_PAULI_Y_HPP
