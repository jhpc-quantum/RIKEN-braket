#ifndef KET_GATE_PAULI_Z_HPP
# define KET_GATE_PAULI_Z_HPP

# include <cassert>
# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace pauli_z_detail
    {
      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger>
      void pauli_z_impl(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        static_assert(
          std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(
          std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(
          ::ket::utility::integer_exp2<StateInteger>(qubit)
          < static_cast<StateInteger>(last - first));
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last - first))
          == static_cast<StateInteger>(last - first));

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last - first)/2u,
          [first, qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            auto const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            auto const one_index = zero_index bitor qubit_mask;

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
            *(first + one_index) *= static_cast<real_type>(-1);
          });
      }
    } // namespace pauli_z_detail

    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::pauli_z_detail::pauli_z_impl(
        ::ket::utility::policy::make_sequential(), first, last, qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::pauli_z_detail::pauli_z_impl(
        parallel_policy, first, last, qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& pauli_z(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::pauli_z_detail::pauli_z_impl(
          ::ket::utility::policy::make_sequential(),
          std::begin(state), std::end(state), qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& pauli_z(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::pauli_z_detail::pauli_z_impl(
          parallel_policy, std::begin(state), std::end(state), qubit);
        return state;
      }
    } // namespace ranges


    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void adj_pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_z(first, last, qubit); }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void adj_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_z(parallel_policy, first, last, qubit); }

    namespace ranges
    {
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_pauli_z(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_z(state, qubit); }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_pauli_z(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_z(parallel_policy, state, qubit); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_PAULI_Z_HPP
