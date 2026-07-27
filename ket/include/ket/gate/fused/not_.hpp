#ifndef KET_GATE_FUSED_NOT_HPP
# define KET_GATE_FUSED_NOT_HPP

# include <cstddef>
# include <algorithm>
# include <array>
# include <iterator>
# include <memory>

# include <boost/range/iterator_range.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/fused/pauli_x.hpp>
# include <ket/gate/fused/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      // NOT_i
      // NOT_1 (a_0 |0> + a_1 |1>) = a_1 |0> + a_0 |1>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit)
      -> void
      { ::ket::gate::fused::pauli_x(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, target_qubit); }

      // CNOT_{tc}, or C1NOT_{tc}
      // CNOT_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{11} |10> + a_{10} |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      { ::ket::gate::fused::pauli_x(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, target_qubit, control_qubit); }

      // C...CNOT_{tc...c'}, or CnNOT_{tc...c'}
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... ControlQubits>
      inline auto not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};
        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
            using std::begin;
            using std::end;
            auto const iter0
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index0,
                      begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                      begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                    begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                    begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));
            // 0b11...11u
            auto const iter1
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index0 bitor std::size_t{1u},
                      begin(unsorted_operated_qubits), end(unsorted_operated_qubits),
                      begin(sorted_operated_qubits_with_sentinel), end(sorted_operated_qubits_with_sentinel)),
                    begin(unsorted_fused_qubits), end(unsorted_fused_qubits),
                    begin(sorted_fused_qubits_with_sentinel), end(sorted_fused_qubits_with_sentinel));

            std::iter_swap(iter0, iter1);
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... ControlQubits>
      inline auto adj_not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::not_(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, target_qubit, control_qubits...); }


      namespace runtime
      {
        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto not_(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          auto const target_qubit_ptr = std::addressof(target_qubit);
          ::ket::gate::fused::runtime::pauli_x(
            first, fused_index_wo_qubits,
            unsorted_fused_qubit_first, unsorted_fused_qubit_last,
            sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last,
            target_qubit_ptr, std::next(target_qubit_ptr), control_qubit_first, control_qubit_last);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename BitInteger>
        inline auto not_(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        {
          auto const target_qubit_ptr = std::addressof(target_qubit);
          ::ket::gate::fused::runtime::pauli_x(
            first, fused_index_wo_qubits,
            unsorted_fused_qubit_first, unsorted_fused_qubit_last,
            sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last,
            target_qubit_ptr, std::next(target_qubit_ptr));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto adj_not_(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          auto const target_qubit_ptr = std::addressof(target_qubit);
          ::ket::gate::fused::runtime::adj_pauli_x(
            first, fused_index_wo_qubits,
            unsorted_fused_qubit_first, unsorted_fused_qubit_last,
            sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last,
            target_qubit_ptr, std::next(target_qubit_ptr), control_qubit_first, control_qubit_last);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename QubitIterator1, typename QubitIterator2, typename BitInteger>
        inline auto adj_not_(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          QubitIterator1 const unsorted_fused_qubit_first, QubitIterator1 const unsorted_fused_qubit_last,
          QubitIterator2 const sorted_fused_qubit_with_sentinel_first, QubitIterator2 const sorted_fused_qubit_with_sentinel_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        {
          auto const target_qubit_ptr = std::addressof(target_qubit);
          ::ket::gate::fused::runtime::adj_pauli_x(
            first, fused_index_wo_qubits,
            unsorted_fused_qubit_first, unsorted_fused_qubit_last,
            sorted_fused_qubit_with_sentinel_first, sorted_fused_qubit_with_sentinel_last,
            target_qubit_ptr, std::next(target_qubit_ptr));
        }

        namespace ranges
        {
          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto not_(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            auto const target_qubit_ptr = std::addressof(target_qubit);
            ::ket::gate::fused::runtime::ranges::pauli_x(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              boost::make_iterator_range(target_qubit_ptr, std::next(target_qubit_ptr)), control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename BitInteger>
          inline auto not_(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> void
          {
            auto const target_qubit_ptr = std::addressof(target_qubit);
            ::ket::gate::fused::runtime::ranges::pauli_x(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              boost::make_iterator_range(target_qubit_ptr, std::next(target_qubit_ptr)));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto adj_not_(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            auto const target_qubit_ptr = std::addressof(target_qubit);
            ::ket::gate::fused::runtime::ranges::adj_pauli_x(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              boost::make_iterator_range(target_qubit_ptr, std::next(target_qubit_ptr)), control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename QubitsRange1, typename QubitsRange2, typename BitInteger>
          inline auto adj_not_(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            QubitsRange1 const& unsorted_fused_qubits, QubitsRange2 const& sorted_fused_qubits_with_sentinel,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> void
          {
            auto const target_qubit_ptr = std::addressof(target_qubit);
            ::ket::gate::fused::runtime::ranges::adj_pauli_x(
              first, fused_index_wo_qubits,
              unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,
              boost::make_iterator_range(target_qubit_ptr, std::next(target_qubit_ptr)));
          }
        } // namespace ranges
      } // namespace runtime
# else // KET_USE_BIT_MASKS_EXPLICITLY
      // NOT_i
      // NOT_1 (a_0 |0> + a_1 |1>) = a_1 |0> + a_0 |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit)
      -> void
      { ::ket::gate::fused::pauli_x(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, target_qubit); }

      // CNOT_{tc}, or C1NOT_{tc}
      // CNOT_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{11} |10> + a_{10} |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      { ::ket::gate::fused::pauli_x(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, target_qubit, control_qubit); }

      // C...CNOT_{tc...c'}, or CnNOT_{tc...c'}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};
        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{1u};
            using std::begin;
            using std::end;
            auto const iter0
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index0,
                      begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                    begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));
            // 0b11...11u
            auto const iter1
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(
                      operated_index_wo_qubits, index0 bitor std::size_t{1u},
                      begin(operated_qubit_masks), end(operated_qubit_masks), begin(operated_index_masks), end(operated_index_masks)),
                    begin(fused_qubit_masks), end(fused_qubit_masks), begin(fused_index_masks), end(fused_index_masks));

            std::iter_swap(iter0, iter1);
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto adj_not_(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::not_(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, target_qubit, control_qubits...); }


      namespace runtime
      {
        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto not_(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          auto const target_qubit_ptr = std::addressof(target_qubit);
          ::ket::gate::fused::runtime::pauli_x(
            first, fused_index_wo_qubits,
            fused_qubit_mask_first, fused_qubit_mask_last,
            fused_index_mask_first, fused_index_mask_last,
            target_qubit_ptr, std::next(target_qubit_ptr), control_qubit_first, control_qubit_last);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename BitInteger>
        inline auto not_(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        {
          auto const target_qubit_ptr = std::addressof(target_qubit);
          ::ket::gate::fused::runtime::pauli_x(
            first, fused_index_wo_qubits,
            fused_qubit_mask_first, fused_qubit_mask_last,
            fused_index_mask_first, fused_index_mask_last,
            target_qubit_ptr, std::next(target_qubit_ptr));
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename BitInteger, typename ControlQubitIterator>
        inline auto adj_not_(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> void
        {
          auto const target_qubit_ptr = std::addressof(target_qubit);
          ::ket::gate::fused::runtime::adj_pauli_x(
            first, fused_index_wo_qubits,
            fused_qubit_mask_first, fused_qubit_mask_last,
            fused_index_mask_first, fused_index_mask_last,
            target_qubit_ptr, std::next(target_qubit_ptr), control_qubit_first, control_qubit_last);
        }

        template <typename RandomAccessIterator, typename StateInteger, typename StateIntegerIterator1, typename StateIntegerIterator2, typename BitInteger>
        inline auto adj_not_(
          RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
          StateIntegerIterator1 const fused_qubit_mask_first, StateIntegerIterator1 const fused_qubit_mask_last,
          StateIntegerIterator2 const fused_index_mask_first, StateIntegerIterator2 const fused_index_mask_last,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> void
        {
          auto const target_qubit_ptr = std::addressof(target_qubit);
          ::ket::gate::fused::runtime::adj_pauli_x(
            first, fused_index_wo_qubits,
            fused_qubit_mask_first, fused_qubit_mask_last,
            fused_index_mask_first, fused_index_mask_last,
            target_qubit_ptr, std::next(target_qubit_ptr));
        }

        namespace ranges
        {
          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto not_(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            auto const target_qubit_ptr = std::addressof(target_qubit);
            ::ket::gate::fused::runtime::ranges::pauli_x(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              boost::make_iterator_range(target_qubit_ptr, std::next(target_qubit_ptr)), control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename BitInteger>
          inline auto not_(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> void
          {
            auto const target_qubit_ptr = std::addressof(target_qubit);
            ::ket::gate::fused::runtime::ranges::pauli_x(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              boost::make_iterator_range(target_qubit_ptr, std::next(target_qubit_ptr)));
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename BitInteger, typename ControlQubitsRange>
          inline auto adj_not_(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> void
          {
            auto const target_qubit_ptr = std::addressof(target_qubit);
            ::ket::gate::fused::runtime::ranges::adj_pauli_x(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              boost::make_iterator_range(target_qubit_ptr, std::next(target_qubit_ptr)), control_qubits);
          }

          template <typename RandomAccessIterator, typename StateInteger, typename StateIntegersRange1, typename StateIntegersRange2, typename BitInteger>
          inline auto adj_not_(
            RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
            StateIntegersRange1 const& fused_qubit_masks, StateIntegersRange2 const& fused_index_masks,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> void
          {
            auto const target_qubit_ptr = std::addressof(target_qubit);
            ::ket::gate::fused::runtime::ranges::adj_pauli_x(
              first, fused_index_wo_qubits,
              fused_qubit_masks, fused_index_masks,
              boost::make_iterator_range(target_qubit_ptr, std::next(target_qubit_ptr)));
          }
        } // namespace ranges
      } // namespace runtime
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_NOT_HPP
