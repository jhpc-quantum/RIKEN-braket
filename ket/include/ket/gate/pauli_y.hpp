#ifndef KET_GATE_PAULI_Y_HPP
# define KET_GATE_PAULI_Y_HPP

# include <cassert>
# include <array>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/gate.hpp>
# include <ket/gate/meta/num_control_qubits.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    // Y_i or Y1_i
    // Y_1 (a_0 |0> + a_1 |1>) = -i a_1 |0> + i a_0 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> void
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

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 1u,
        [first, qubit_mask, lower_bits_mask, upper_bits_mask](StateInteger const value_wo_qubit, int const)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((value_wo_qubit bitand upper_bits_mask) << 1u) bitor (value_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;
          auto const zero_iter = first + zero_index;
          auto const one_iter = first + one_index;

          std::iter_swap(zero_iter, one_iter);

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *zero_iter *= -::ket::utility::imaginary_unit<complex_type>();
          *one_iter *= ::ket::utility::imaginary_unit<complex_type>();
        });
    }

    // YY_{ij} = Y_i Y_j or Y2_{ij}
    // YY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = -a_{11} |00> + a_{10} |01> + a_{01} |10> - a_{00} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
    -> void
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

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 2u,
        [first, qubit1_mask, qubit2_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx0_1xxx0_2xxx
          auto const base_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          auto const off_iter = first + base_index;
          // xxx1_1xxx0_2xxx
          auto const qubit1_on_index = base_index bitor qubit1_mask;
          auto const qubit1_on_iter = first + qubit1_on_index;
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_iter = first + (base_index bitor qubit2_mask);
          // xxx1_1xxx1_2xxx
          auto const qubit12_on_iter = first + (qubit1_on_index bitor qubit2_mask);

          std::iter_swap(off_iter, qubit12_on_iter);
          std::iter_swap(qubit1_on_iter, qubit2_on_iter);

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          *off_iter *= real_type{-1.0};
          *qubit12_on_iter *= real_type{-1.0};
        });
    }

    // CY_{tc}, CY1_{tc}, C1Y_{tc}, or C1Y1_{tc}
    // CY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
    //   = a_{00} |00> + a_{01} |01> - i a_{11} |10> + i a_{10} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(control_qubit) < static_cast<StateInteger>(last - first));
      assert(target_qubit != control_qubit);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      auto const minmax_qubits = std::minmax(target_qubit, control_qubit.qubit());
      auto const target_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
      auto const control_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
      auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
      auto const middle_bits_mask
        = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
          xor lower_bits_mask;
      auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 2u,
        [first, target_qubit_mask, control_qubit_mask, lower_bits_mask, middle_bits_mask, upper_bits_mask](
          StateInteger const value_wo_qubits, int const)
        {
          // xxx0_txxx0_cxxx
          auto const base_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          // xxx0_txxx1_cxxx
          auto const control_on_index = base_index bitor control_qubit_mask;
          // xxx1_txxx1_cxxx
          auto const target_control_on_index = control_on_index bitor target_qubit_mask;
          auto const control_on_iter = first + control_on_index;
          auto const target_control_on_iter = first + target_control_on_index;

          std::iter_swap(control_on_iter, target_control_on_iter);

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          *control_on_iter *= -::ket::utility::imaginary_unit<complex_type>();
          *target_control_on_iter *= ::ket::utility::imaginary_unit<complex_type>();
        });
    }

    // C...CY...Y_{t...t'c...c'} = C...C(Y_t ... Y_t')_{c...c'}, CnY...Y_{...}, C...CYm_{...}, or CnYm_{...}
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename Qubit2, typename Qubit3, typename... Qubits>
    inline auto pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      constexpr auto num_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 3u);
      constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubit2, Qubit3, Qubits...>::value;
      constexpr auto num_target_qubits = num_qubits - num_control_qubits;
      constexpr auto num_indices = ::ket::utility::integer_exp2<std::size_t>(num_qubits);
      constexpr auto num_target_indices = ::ket::utility::integer_exp2<std::size_t>(num_target_qubits);
      constexpr auto half_num_target_indices = num_target_indices / std::size_t{2u};

      using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
      auto coefficient = complex_type{};
      switch (num_target_qubits % BitInteger{4u})
      {
       case BitInteger{0u}:
        coefficient = complex_type{1};

       case BitInteger{1u}:
        coefficient = ::ket::utility::imaginary_unit<complex_type>();

       case BitInteger{2u}:
        coefficient = complex_type{-1};

       default: //case BitInteger{3u}:
        coefficient = -::ket::utility::imaginary_unit<complex_type>();
      }

      ::ket::gate::gate(
        parallel_policy, first, last,
        [&coefficient](auto const first, std::array<StateInteger, num_indices> const& indices, int const)
        {
          // 0b1...10...0u
          constexpr auto base_indices_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

          for (auto i = std::size_t{0u}; i < half_num_target_indices; ++i)
          {
            auto const j = num_target_indices - std::size_t{1u} - i;

            auto num_ones_in_i = BitInteger{0u};
            auto num_ones_in_j = BitInteger{0u};
            auto i_tmp = i;
            auto j_tmp = j;
            for (auto count = BitInteger{0u}; count < num_target_qubits; ++count)
            {
              if ((i_tmp bitand std::size_t{1u}) == std::size_t{1u})
                ++num_ones_in_i;
              if ((j_tmp bitand std::size_t{1u}) == std::size_t{1u})
                ++num_ones_in_j;

              i_tmp >>= BitInteger{1u};
              j_tmp >>= BitInteger{1u};
            }

            auto iter1 = first + indices[base_indices_index + i];
            auto iter2 = first + indices[base_indices_index + j];
            std::iter_swap(iter1, iter2);
            *iter1 *= (num_target_qubits - num_ones_in_i) % BitInteger{2u} == BitInteger{0u} ? coefficient : -coefficient;
            *iter2 *= (num_target_qubits - num_ones_in_j) % BitInteger{2u} == BitInteger{0u} ? coefficient : -coefficient;
          }
        },
        qubit1, qubit2, qubit3, qubits...);
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto pauli_y(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    -> void
    { ::ket::gate::pauli_y(::ket::utility::policy::make_sequential(), first, last, qubit, qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto pauli_y(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::pauli_y(parallel_policy, begin(state), end(state), qubit, qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto pauli_y(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::pauli_y(::ket::utility::policy::make_sequential(), state, qubit, qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto adj_pauli_y(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    -> void
    { ::ket::gate::pauli_y(parallel_policy, first, last, qubit, qubits...); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... Qubits>
    inline auto adj_pauli_y(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    -> void
    { ::ket::gate::pauli_y(first, last, qubit, qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto adj_pauli_y(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::pauli_y(parallel_policy, state, qubit, qubits...); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto adj_pauli_y(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::pauli_y(state, qubit, qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_PAULI_Y_HPP
