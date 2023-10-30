#ifndef KET_GATE_PAULI_Z_HPP
# define KET_GATE_PAULI_Z_HPP

# include <cassert>
# include <array>
# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
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
    // Z_i
    // Z_1 (a_0 |0> + a_1 |1>) = a_0 |0> - a_1 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::integer_exp2<StateInteger>(qubit) < static_cast<StateInteger>(last - first));
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
      auto const lower_bits_mask = qubit_mask - StateInteger{1u};
      auto const upper_bits_mask = compl lower_bits_mask;

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 1u,
        [first, qubit_mask, lower_bits_mask, upper_bits_mask](StateInteger const value_wo_qubit, int const)
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

    // ZZ_i = Z_i Z_j
    // ZZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{00} |00> - a_{01} |01> - a_{10} |10> + a{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      assert(::ket::utility::integer_exp2<StateInteger>(qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(qubit2) < static_cast<StateInteger>(last - first));
      assert(qubit1 != qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const minmax_qubits = std::minmax(qubit1, qubit2);
      auto const qubit1_mask = ::ket::utility::integer_exp2<StateInteger>(qubit1);
      auto const qubit2_mask = ::ket::utility::integer_exp2<StateInteger>(qubit2);
      auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
      auto const middle_bits_mask
        = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
          xor lower_bits_mask;
      auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

      using ::ket::utility::loop_n;
      loop_n(
        parallel_policy,
        static_cast<StateInteger>(last - first) >> 2u,
        [first, qubit1_mask, qubit2_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx0_1xxx0_2xxx
          auto const base_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          // xxx1_1xxx0_2xxx
          auto const qubit1_on_iter = first + (base_index bitor qubit1_mask);
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_iter = first + (base_index bitor qubit2_mask);

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
          *qubit1_on_iter *= real_type{-1.0};
          *qubit2_on_iter *= real_type{-1.0};
        });
    }

    // Z...Z_{i...j} = Z_i ... Z_j
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
      ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits)
    {
      constexpr auto num_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 3u);
      constexpr auto num_indices = ::ket::utility::integer_exp2<std::size_t>(num_qubits);

      ::ket::gate::gate(
        parallel_policy, first, last,
        [](RandomAccessIterator const first, std::array<StateInteger, num_indices> const& indices, int const)
        {
          auto const num_indices = static_cast<StateInteger>(boost::size(indices));
          for (auto i = StateInteger{0u}; i < num_indices; ++i)
          {
            auto num_ones_in_i = BitInteger{0u};
            auto i_tmp = i;
            for (auto count = BitInteger{0u}; count < num_qubits; ++count)
            {
              if (i_tmp bitand StateInteger{1u} == StateInteger{1u})
                ++num_ones_in_i;

              i_tmp >>= BitInteger{1u};
            }

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
            if (num_ones_in_i % BitInteger{2u} == BitInteger{1u})
              *(first + indices[i]) *= real_type{-1.0};
          }
        },
        qubit1, qubit2, qubit3, qubits...);
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    { ::ket::gate::pauli_z(::ket::utility::policy::make_sequential(), first, last, qubit, qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& pauli_z(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        ::ket::gate::pauli_z(parallel_policy, std::begin(state), std::end(state), qubit);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& pauli_z(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      { return ::ket::gate::ranges::pauli_z(::ket::utility::policy::make_sequential(), state, qubit); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void adj_pauli_z(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    { ::ket::gate::pauli_z(parallel_policy, first, last, qubit, qubits...); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline void adj_pauli_z(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    { ::ket::gate::pauli_z(first, last, qubit, qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& adj_pauli_z(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      { return ::ket::gate::ranges::pauli_z(parallel_policy, state, qubit, qubits...); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline RandomAccessRange& adj_pauli_z(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      { return ::ket::gate::ranges::pauli_z(state, qubit, qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_PAULI_Z_HPP
