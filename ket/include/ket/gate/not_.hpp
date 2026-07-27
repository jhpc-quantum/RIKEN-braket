#ifndef KET_GATE_NOT_HPP
# define KET_GATE_NOT_HPP

# include <cassert>
# include <array>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>

# include <boost/range/iterator_range.hpp>
# include <boost/range/join.hpp>
# include <boost/range/adaptor/transformed.hpp>

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
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, unsorted_qubits, sorted_qubits_with_sentinel),
            first
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, unsorted_qubits, sorted_qubits_with_sentinel));
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
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, qubit_masks, index_masks),
            first
              + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, qubit_masks, index_masks));
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
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
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
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
      { return ::ket::gate::ranges::not_(parallel_policy, state, target_qubit, control_qubits...); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_not_(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits) -> RandomAccessRange&
      { return ::ket::gate::ranges::not_(state, target_qubit, control_qubits...); }
    } // namespace ranges


    namespace runtime
    {
      // C...CNOT_{tc...c'}, or CnNOT_{tc...c'}
      namespace qubit_ranges
      {
        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto not_(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          using control_qubit_type = ::ket::control<qubit_type>;
          static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
          static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
          static_assert(std::is_same< ::ket::utility::meta::range_value_t<ControlQubitsRange>, control_qubit_type >::value, "The value_type of ControlQubitsRange should be the same as ket::control<ket::qubit<S, B> >");

          assert(::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first));
          assert(
            ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
            == static_cast<StateInteger>(last - first));
          using std::begin;
          using std::end;
          auto const num_control_qubits = static_cast<BitInteger>(end(control_qubits) - begin(control_qubits));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
          ::ket::gate::runtime::nocache::qubit_ranges::gate(
            parallel_policy, first, last,
            [num_control_qubits](
              auto const first, StateInteger const index_wo_qubits,
              auto const& unsorted_qubits, auto const& sorted_qubits_with_sentinel,
              int const)
            {
              // 0b11...10u
              auto const index0 = ((StateInteger{1u} << num_control_qubits) - StateInteger{1u}) << BitInteger{1u};
              // 0b11...11u
              auto const index1 = index0 bitor StateInteger{1u};

              std::iter_swap(
                first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, unsorted_qubits, sorted_qubits_with_sentinel),
                first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, unsorted_qubits, sorted_qubits_with_sentinel));
            },
            boost::join(
              boost::make_iterator_range(std::addressof(target_qubit), std::next(std::addressof(target_qubit))),
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
# else // KET_USE_BIT_MASKS_EXPLICITLY
          ::ket::gate::runtime::nocache::qubit_ranges::gate(
            parallel_policy, first, last,
            [num_control_qubits](
              auto const first, StateInteger const index_wo_qubits,
              auto const& qubit_masks, auto const& index_masks,
              int const)
            {
              // 0b11...10u
              auto const index0 = ((StateInteger{1u} << num_control_qubits) - StateInteger{1u}) << BitInteger{1u};
              // 0b11...11u
              auto const index1 = index0 bitor StateInteger{1u};

              std::iter_swap(
                first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index0, qubit_masks, index_masks),
                first
                  + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index1, qubit_masks, index_masks));
            },
            boost::join(
              boost::make_iterator_range(std::addressof(target_qubit), std::next(std::addressof(target_qubit))),
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
# endif // KET_USE_BIT_MASKS_EXPLICITLY
        }

        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto not_(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          using control_qubit_type = ::ket::control<qubit_type>;
          std::array<control_qubit_type, 0u> const control_qubits{};
          ::ket::gate::runtime::qubit_ranges::not_(parallel_policy, first, last, target_qubit, control_qubits);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto not_(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        { ::ket::gate::runtime::qubit_ranges::not_(::ket::utility::policy::make_sequential(), first, last, target_qubit, control_qubits); }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto not_(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        { ::ket::gate::runtime::qubit_ranges::not_(::ket::utility::policy::make_sequential(), first, last, target_qubit); }
      } // namespace qubit_ranges

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto not_(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      {
        ::ket::gate::runtime::qubit_ranges::not_(
          parallel_policy, first, last,
          target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
      }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto not_(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit)
      -> void
      { ::ket::gate::runtime::qubit_ranges::not_(parallel_policy, first, last, target_qubit); }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto not_(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      {
        ::ket::gate::runtime::qubit_ranges::not_(
          first, last,
          target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto not_(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit)
      -> void
      { ::ket::gate::runtime::qubit_ranges::not_(first, last, target_qubit); }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto not_(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::not_(parallel_policy, begin(state), end(state), target_qubit, control_qubits);
          return state;
        }

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto not_(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::not_(parallel_policy, begin(state), end(state), target_qubit);
          return state;
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto not_(
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::not_(begin(state), end(state), target_qubit, control_qubits);
          return state;
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto not_(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const target_qubit) -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::not_(begin(state), end(state), target_qubit);
          return state;
        }
      } // namespace ranges

      namespace qubit_ranges
      {
        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_not_(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        { ::ket::gate::runtime::qubit_ranges::not_(parallel_policy, first, last, target_qubit, control_qubits); }

        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto adj_not_(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        { ::ket::gate::runtime::qubit_ranges::not_(parallel_policy, first, last, target_qubit); }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_not_(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> void
        { ::ket::gate::runtime::qubit_ranges::not_(first, last, target_qubit, control_qubits); }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto adj_not_(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        { ::ket::gate::runtime::qubit_ranges::not_(first, last, target_qubit); }
      } // namespace qubit_ranges

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto adj_not_(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      { ::ket::gate::runtime::not_(parallel_policy, first, last, target_qubit, control_qubit_first, control_qubit_last); }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto adj_not_(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit)
      -> void
      { ::ket::gate::runtime::not_(parallel_policy, first, last, target_qubit); }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto adj_not_(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      { ::ket::gate::runtime::not_(first, last, target_qubit, control_qubit_first, control_qubit_last); }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto adj_not_(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit)
      -> void
      { ::ket::gate::runtime::not_(first, last, target_qubit); }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_not_(
          ParallelPolicy const parallel_policy, RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        { return ::ket::gate::runtime::ranges::not_(parallel_policy, state, target_qubit, control_qubits); }

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_not_(
          ParallelPolicy const parallel_policy, RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        { return ::ket::gate::runtime::ranges::not_(parallel_policy, state, target_qubit); }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_not_(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits) -> RandomAccessRange&
        { return ::ket::gate::runtime::ranges::not_(state, target_qubit, control_qubits); }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_not_(RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const target_qubit) -> RandomAccessRange&
        { return ::ket::gate::runtime::ranges::not_(state, target_qubit); }
      } // namespace ranges
    } // namespace runtime
  } // namespace gate
} // namespace ket


#endif // KET_GATE_NOT_HPP
