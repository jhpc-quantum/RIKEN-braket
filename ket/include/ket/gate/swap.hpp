#ifndef KET_GATE_SWAP_HPP
# define KET_GATE_SWAP_HPP

# include <cassert>
# include <array>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <boost/range/iterator_range.hpp>
# include <boost/range/join.hpp>
# include <boost/range/adaptor/transformed.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
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
    // SWAP_{ij}
    // SWAP_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
    //   = a_{00} |00> + a_{10} |01> + a_{01} |10> + a_{11} |11>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto swap(
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
          // xxx1_1xxx0_2xxx
          auto const qubit1_on_index = base_index bitor qubit1_mask;
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_index = base_index bitor qubit2_mask;

          std::iter_swap(first + qubit1_on_index, first + qubit2_on_index);
        });
    }

    // C...CSWAP_{tt'c...c'} or CnSWAP_{tt'c...c'}
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto swap(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit1) < static_cast<StateInteger>(last - first));
      assert(::ket::utility::integer_exp2<StateInteger>(target_qubit2) < static_cast<StateInteger>(last - first));
      assert(target_qubit1 != target_qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 1u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{2u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
          auto const first, StateInteger const index_wo_qubits,
          std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
          std::array<qubit_type, num_operated_qubits + BitInteger{1u}> const& sorted_qubits_with_sentinel,
          int const)
        {
          // 0b11...100u
          constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
          // 0b11...101u
          constexpr auto index01 = base_index bitor std::size_t{1u};
          // 0b11...110u
          constexpr auto index10 = base_index bitor (std::size_t{1u} << BitInteger{1u});

          using std::begin;
          using std::end;
          std::iter_swap(
            first
            + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index01, unsorted_qubits, sorted_qubits_with_sentinel),
            first
            + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index10, unsorted_qubits, sorted_qubits_with_sentinel));
        },
        target_qubit1, target_qubit2, control_qubit, control_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
      constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 1u);
      constexpr auto num_operated_qubits = num_control_qubits + BitInteger{2u};

      ::ket::gate::nocache::gate(
        parallel_policy, first, last,
        [](
          auto const first, StateInteger const index_wo_qubits,
          std::array<StateInteger, num_operated_qubits> const& qubit_masks,
          std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
          int const)
        {
          // 0b11...100u
          constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
          // 0b11...101u
          constexpr auto index01 = base_index bitor std::size_t{1u};
          // 0b11...110u
          constexpr auto index10 = base_index bitor (std::size_t{1u} << BitInteger{1u});

          using std::begin;
          using std::end;
          std::iter_swap(
            first
            + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index01, qubit_masks, index_masks),
            first
            + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index10, qubit_masks, index_masks));
        },
        target_qubit1, target_qubit2, control_qubit, control_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto swap(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::swap(::ket::utility::policy::make_sequential(), first, last, target_qubit1, target_qubit2, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto swap(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
      {
        using std::begin;
        using std::end;
        ::ket::gate::swap(parallel_policy, begin(state), end(state), target_qubit1, target_qubit2, control_qubits...);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto swap(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::swap(::ket::utility::policy::make_sequential(), state, target_qubit1, target_qubit2, control_qubits...); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_swap(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::swap(parallel_policy, first, last, target_qubit1, target_qubit2, control_qubits...); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename... ControlQubits>
    inline auto adj_swap(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
      ControlQubits const... control_qubits)
    -> void
    { ::ket::gate::swap(first, last, target_qubit1, target_qubit2, control_qubits...); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_swap(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
      { return ::ket::gate::ranges::swap(parallel_policy, state, target_qubit1, target_qubit2, control_qubits...); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... ControlQubits>
      inline auto adj_swap(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::swap(state, target_qubit1, target_qubit2, control_qubits...); }
    } // namespace ranges


    namespace runtime
    {
      // C...CSWAP_{tt'c...c'} or CnSWAP_{tt'c...c'}
      namespace qubit_ranges
      {
        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto swap(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitsRange const& control_qubits)
        -> void
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          using control_qubit_type = ::ket::control<qubit_type>;
          static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
          static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
          static_assert(std::is_same< ::ket::utility::meta::range_value_t<ControlQubitsRange>, control_qubit_type >::value, "The value_type of ControlQubitsRange should be ket::control<ket::qubit<S, B> >");

          assert(::ket::utility::integer_exp2<StateInteger>(target_qubit1) < static_cast<StateInteger>(last - first));
          assert(::ket::utility::integer_exp2<StateInteger>(target_qubit2) < static_cast<StateInteger>(last - first));
          assert(target_qubit1 != target_qubit2);
          assert(
            ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
            == static_cast<StateInteger>(last - first));
          using std::begin;
          using std::end;
          auto const num_control_qubits = static_cast<BitInteger>(end(control_qubits) - begin(control_qubits));

          std::array<qubit_type, 2u> const target_qubits{{target_qubit1, target_qubit2}};
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
          ::ket::gate::runtime::nocache::qubit_ranges::gate(
            parallel_policy, first, last,
            [num_control_qubits](
              auto const first, StateInteger const index_wo_qubits,
              auto const& unsorted_qubits, auto const& sorted_qubits_with_sentinel,
              int const)
            {
              // 0b11...100u
              auto const base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
              // 0b11...101u
              auto const index01 = base_index bitor std::size_t{1u};
              // 0b11...110u
              auto const index10 = base_index bitor (std::size_t{1u} << BitInteger{1u});

              std::iter_swap(
                first
                + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index01, unsorted_qubits, sorted_qubits_with_sentinel),
                first
                + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index10, unsorted_qubits, sorted_qubits_with_sentinel));
            },
            boost::join(
              target_qubits,
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
              // 0b11...100u
              auto const base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
              // 0b11...101u
              auto const index01 = base_index bitor std::size_t{1u};
              // 0b11...110u
              auto const index10 = base_index bitor (std::size_t{1u} << BitInteger{1u});

              std::iter_swap(
                first
                + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index01, qubit_masks, index_masks),
                first
                + ::ket::gate::utility::ranges::index_with_qubits(index_wo_qubits, index10, qubit_masks, index_masks));
            },
            boost::join(
              target_qubits,
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));
# endif // KET_USE_BIT_MASKS_EXPLICITLY
        }

        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto swap(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          using control_qubit_type = ::ket::control<qubit_type>;
          std::array<control_qubit_type, 0u> const control_qubits{};
          ::ket::gate::runtime::qubit_ranges::swap(parallel_policy, first, last, target_qubit1, target_qubit2, control_qubits);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto swap(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitsRange const& control_qubits)
        -> void
        { ::ket::gate::runtime::qubit_ranges::swap(::ket::utility::policy::make_sequential(), first, last, target_qubit1, target_qubit2, control_qubits); }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto swap(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        { ::ket::gate::runtime::qubit_ranges::swap(::ket::utility::policy::make_sequential(), first, last, target_qubit1, target_qubit2); }
      } // namespace qubit_ranges

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto swap(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      {
        ::ket::gate::runtime::qubit_ranges::swap(
          parallel_policy, first, last,
          target_qubit1, target_qubit2,
          boost::make_iterator_range(control_qubit_first, control_qubit_last));
      }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto swap(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
      -> void
      { ::ket::gate::runtime::qubit_ranges::swap(parallel_policy, first, last, target_qubit1, target_qubit2); }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto swap(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      {
        ::ket::gate::runtime::qubit_ranges::swap(
          first, last,
          target_qubit1, target_qubit2,
          boost::make_iterator_range(control_qubit_first, control_qubit_last));
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto swap(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
      -> void
      { ::ket::gate::runtime::qubit_ranges::swap(first, last, target_qubit1, target_qubit2); }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto swap(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitsRange const& control_qubits)
        -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::swap(parallel_policy, begin(state), end(state), target_qubit1, target_qubit2, control_qubits);
          return state;
        }

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto swap(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::swap(parallel_policy, begin(state), end(state), target_qubit1, target_qubit2);
          return state;
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto swap(
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::swap(begin(state), end(state), target_qubit1, target_qubit2, control_qubits);
          return state;
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto swap(
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> RandomAccessRange&
        {
          using std::begin;
          using std::end;
          ::ket::gate::runtime::qubit_ranges::swap(begin(state), end(state), target_qubit1, target_qubit2);
          return state;
        }
      } // namespace ranges

      namespace qubit_ranges
      {
        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_swap(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitsRange const& control_qubits)
        -> void
        { ::ket::gate::runtime::qubit_ranges::swap(parallel_policy, first, last, target_qubit1, target_qubit2, control_qubits); }

        template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto adj_swap(
          ParallelPolicy const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        { ::ket::gate::runtime::qubit_ranges::swap(parallel_policy, first, last, target_qubit1, target_qubit2); }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_swap(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitsRange const& control_qubits)
        -> void
        { ::ket::gate::runtime::qubit_ranges::swap(first, last, target_qubit1, target_qubit2, control_qubits); }

        template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
        inline auto adj_swap(
          RandomAccessIterator const first, RandomAccessIterator const last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> void
        { ::ket::gate::runtime::qubit_ranges::swap(first, last, target_qubit1, target_qubit2); }
      } // namespace qubit_ranges

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto adj_swap(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      { ::ket::gate::runtime::swap(parallel_policy, first, last, target_qubit1, target_qubit2, control_qubit_first, control_qubit_last); }

      template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto adj_swap(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
      -> void
      { ::ket::gate::runtime::swap(parallel_policy, first, last, target_qubit1, target_qubit2); }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, typename ControlQubitIterator>
      inline auto adj_swap(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
      -> void
      { ::ket::gate::runtime::swap(first, last, target_qubit1, target_qubit2, control_qubit_first, control_qubit_last); }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
      inline auto adj_swap(
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
      -> void
      { ::ket::gate::runtime::swap(first, last, target_qubit1, target_qubit2); }

      namespace ranges
      {
        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_swap(
          ParallelPolicy const parallel_policy, RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitsRange const& control_qubits)
        -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
        { return ::ket::gate::runtime::ranges::swap(parallel_policy, state, target_qubit1, target_qubit2, control_qubits); }

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_swap(
          ParallelPolicy const parallel_policy, RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> std::enable_if_t< ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, RandomAccessRange& >
        { return ::ket::gate::runtime::ranges::swap(parallel_policy, state, target_qubit1, target_qubit2); }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
        inline auto adj_swap(
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubitsRange const& control_qubits)
        -> RandomAccessRange&
        { return ::ket::gate::runtime::ranges::swap(state, target_qubit1, target_qubit2, control_qubits); }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline auto adj_swap(
          RandomAccessRange& state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2)
        -> RandomAccessRange&
        { return ::ket::gate::runtime::ranges::swap(state, target_qubit1, target_qubit2); }
      } // namespace ranges
    } // namespace runtime
  } // namespace gate
} // namespace ket


#endif // KET_GATE_SWAP_HPP
