#ifndef KET_GATE_NOT_HPP
# define KET_GATE_NOT_HPP

# include <cassert>
# include <array>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/pauli_x.hpp>
# include <ket/gate/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif


namespace ket
{
  namespace gate
  {
    // NOT_i
    // NOT_1 (a_0 |0> + a_1 |1>) = a_1 |0> + a_0 |1>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto not_(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit)
    -> void
    { ::ket::gate::pauli_x(parallel_policy, first, last, target_qubit); }

    // CNOT_{tc}, or C1NOT_{tc}
    // CNOT_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + a_{11} |10> + a_{10} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto not_(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
    -> void
    { ::ket::gate::pauli_x(parallel_policy, first, last, target_qubit, control_qubit); }

    // C...CNOT_{tc...c'}, or CnNOT_{tc...c'}
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto not_(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
    -> void
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((StateInteger{1u} << num_control_qubits) - StateInteger{1u}) << BitInteger{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor StateInteger{1u};

          using std::begin;
          using std::end;
          std::iter_swap(
            first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index0,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel)),
            first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index1,
                  begin(unsorted_qubits), end(unsorted_qubits),
                  begin(sorted_qubits_with_sentinel), end(sorted_qubits_with_sentinel)));
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](auto const first, StateInteger const index_wo_qubits, std::array<StateInteger, num_operated_qubits> const& qubit_masks, std::array<StateInteger, num_operated_qubits + 1u> const& index_masks, int const)
        {
          // 0b11...10u
          constexpr auto index0 = ((StateInteger{1u} << num_control_qubits) - StateInteger{1u}) << BitInteger{1u};
          // 0b11...11u
          constexpr auto index1 = index0 bitor StateInteger{1u};

          using std::begin;
          using std::end;
          std::iter_swap(
            first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index0,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks)),
            first
              + ::ket::gate::utility::index_with_qubits(
                  index_wo_qubits, index1,
                  begin(qubit_masks), end(qubit_masks), begin(index_masks), end(index_masks)));
        },
        target_qubit, control_qubit1, control_qubit2, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto not_(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit)
    -> void
    { ::ket::gate::not_(::ket::utility::policy::make_sequential(), first, last, target_qubit); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto not_(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::not_(::ket::utility::policy::make_sequential(), first, last, target_qubit, control_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto not_(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::not_(parallel_policy, begin(state), end(state), target_qubit, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto not_(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::not_(::ket::utility::policy::make_sequential(), state, target_qubit, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_not_(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::not_(parallel_policy, first, last, target_qubit, control_qubits...); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_not_(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::not_(first, last, target_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_not_(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::not_(parallel_policy, state, target_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_not_(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::not_(state, target_qubit, control_qubits...); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_NOT_HPP
