#ifndef KET_GATE_FUSED_SWAP_HPP
# define KET_GATE_FUSED_SWAP_HPP

# include <cassert>
# include <cstddef>
# include <array>
# include <algorithm>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/fused/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      // SWAP_{ij}
      // SWAP_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{10} |01> + a_{01} |10> + a_{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit1 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(qubit2 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(qubit1 != qubit2);

        constexpr auto num_operated_qubits = BitInteger{2u};

        auto const minmax_qubits = std::minmax(qubit1, qubit2);
        auto const qubit1_mask = ::ket::utility::integer_exp2<StateInteger>(qubit1);
        auto const qubit2_mask = ::ket::utility::integer_exp2<StateInteger>(qubit2);
        auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_1xxx0_2xxx
          auto const base_index
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask);
          // xxx1_1xxx0_2xxx
          auto const qubit1_on_index = base_index bitor qubit1_mask;
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_index = base_index bitor qubit2_mask;

          std::iter_swap(
            first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, qubit1_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel),
            first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, qubit2_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel));
        }
      }

      // C...CSWAP_{tt'c...c'} or CnSWAP_{tt'c...c'}
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... ControlQubits>
      inline auto swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> void
      {
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 1u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{2u};
        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b11...100u
            constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
            // 0b11...101u
            constexpr auto index01 = base_index bitor std::size_t{1u};
            auto const iter01
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index01, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                    unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
            // 0b11...110u
            constexpr auto index10 = base_index bitor (std::size_t{1u} << BitInteger{1u});
            auto const iter10
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index10, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                    unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);

            std::iter_swap(iter01, iter10);
          },
          target_qubit1, target_qubit2, control_qubit, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename... ControlQubits>
      inline auto adj_swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::swap(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, target_qubit1, target_qubit2, control_qubits...); }
# else // KET_USE_BIT_MASKS_EXPLICITLY
      // SWAP_{ij}
      // SWAP_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{10} |01> + a_{01} |10> + a_{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit1 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(qubit2 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(qubit1 != qubit2);

        constexpr auto num_operated_qubits = BitInteger{2u};

        auto const minmax_qubits = std::minmax(qubit1, qubit2);
        auto const qubit1_mask = ::ket::utility::integer_exp2<StateInteger>(qubit1);
        auto const qubit2_mask = ::ket::utility::integer_exp2<StateInteger>(qubit2);
        auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_1xxx0_2xxx
          auto const base_index
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask);
          // xxx1_1xxx0_2xxx
          auto const qubit1_on_index = base_index bitor qubit1_mask;
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_index = base_index bitor qubit2_mask;

          std::iter_swap(
            first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, qubit1_on_index, fused_qubit_masks, fused_index_masks),
            first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, qubit2_on_index, fused_qubit_masks, fused_index_masks));
        }
      }

      // C...CSWAP_{tt'c...c'} or CnSWAP_{tt'c...c'}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> void
      {
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 1u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{2u};
        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...100u
            constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
            // 0b11...101u
            constexpr auto index01 = base_index bitor std::size_t{1u};
            auto const iter01
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index01, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            // 0b11...110u
            constexpr auto index10 = base_index bitor (std::size_t{1u} << BitInteger{1u});
            auto const iter10
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index10, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);

            std::iter_swap(iter01, iter10);
          },
          target_qubit1, target_qubit2, control_qubit, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto adj_swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::swap(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, target_qubit1, target_qubit2, control_qubits...); }
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_SWAP_HPP
