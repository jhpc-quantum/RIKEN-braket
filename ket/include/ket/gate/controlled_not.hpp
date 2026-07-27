#ifndef KET_GATE_CONTROLLED_NOT_HPP
# define KET_GATE_CONTROLLED_NOT_HPP

# include <cassert>
# include <array>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/not_.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif


namespace ket
{
  namespace gate
  {
    // CNOT_{tc} or C1NOT_{tc}
    // CNOT_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{01} |01> + a_{11} |10> + a_{10} |11>
    // C...CNOT_{tc...c'} or CnNOT_{tc...c'}
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto controlled_not(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::not_(parallel_policy, first, last, target_qubit, control_qubit, control_qubits...); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto controlled_not(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::not_(first, last, target_qubit, control_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto controlled_not(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
      { return ::ket::gate::ranges::not_(parallel_policy, state, target_qubit, control_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto controlled_not(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::not_(state, target_qubit, control_qubit, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_controlled_not(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::adj_not_(parallel_policy, first, last, target_qubit, control_qubit, control_qubits...); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_controlled_not(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::adj_not_(first, last, target_qubit, control_qubit, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_not(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
      { return ::ket::gate::ranges::adj_not_(parallel_policy, state, target_qubit, control_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_not(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::adj_not_(state, target_qubit, control_qubit, control_qubits...); }
    } // namespace ranges


    namespace runtime
    {
      // CNOT_{tc} or C1NOT_{tc}
      // CNOT_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{11} |10> + a_{10} |11>
      // C...CNOT_{tc...c'} or CnNOT_{tc...c'}
      namespace qubit_ranges
      {
        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto controlled_not(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        {
          using std::begin;
          using std::end;
          assert(begin(control_qubits) != end(control_qubits));
          ::ket::gate::runtime::qubit_ranges::not_(parallel_policy, first, last, target_qubit, control_qubits);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto controlled_not(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        {
          using std::begin;
          using std::end;
          assert(begin(control_qubits) != end(control_qubits));
          ::ket::gate::runtime::qubit_ranges::not_(first, last, target_qubit, control_qubits);
        }
      } // namespace qubit_ranges

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto controlled_not(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      {
        assert(control_qubit_first != control_qubit_last);
        ::ket::gate::runtime::not_(parallel_policy, first, last, target_qubit, control_qubit_first, control_qubit_last);
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto controlled_not(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      {
        assert(control_qubit_first != control_qubit_last);
        ::ket::gate::runtime::not_(first, last, target_qubit, control_qubit_first, control_qubit_last);
      }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto controlled_not(
          ParallelPolicy const parallel_policy, RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          assert(begin(control_qubits) != end(control_qubits));
          return ::ket::gate::runtime::ranges::not_(parallel_policy, state, target_qubit, control_qubits);
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto controlled_not(
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          assert(begin(control_qubits) != end(control_qubits));
          return ::ket::gate::runtime::ranges::not_(state, target_qubit, control_qubits);
        }
      } // namespace ranges

      namespace qubit_ranges
      {
        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_controlled_not(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        { ::ket::gate::runtime::qubit_ranges::controlled_not(parallel_policy, first, last, target_qubit, control_qubits); }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_controlled_not(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        { ::ket::gate::runtime::qubit_ranges::controlled_not(first, last, target_qubit, control_qubits); }
      } // namespace qubit_ranges

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto adj_controlled_not(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      { ::ket::gate::runtime::controlled_not(parallel_policy, first, last, target_qubit, control_qubit_first, control_qubit_last); }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto adj_controlled_not(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      { ::ket::gate::runtime::controlled_not(first, last, target_qubit, control_qubit_first, control_qubit_last); }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_controlled_not(
          ParallelPolicy const parallel_policy, RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        { return ::ket::gate::runtime::ranges::controlled_not(parallel_policy, state, target_qubit, control_qubits); }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_controlled_not(
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        { return ::ket::gate::runtime::ranges::controlled_not(state, target_qubit, control_qubits); }
      } // namespace ranges
    } // namespace runtime
  } // namespace gate
} // namespace ket


#endif // KET_GATE_CONTROLLED_NOT_HPP
