#ifndef KET_GATE_FUSED_TOFFOLI_HPP
# define KET_GATE_FUSED_TOFFOLI_HPP

# include <cassert>
# include <cstddef>
# include <array>
# include <algorithm>
# include <iterator>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      // TOFFOLI_{t,c1,c2}
      // TOFFOLI_{1,2,3} (a_{000} |000> + a_{001} |001> + a_{010} |010> + a_{011} |011> + a_{100} |100> + a_{101} |101> + a_{110} |110> + a_{111} |111>)
      //   = a_{000} |000> + a_{001} |001> + a_{010} |010> + a_{011} |011> + a_{100} |100> + a_{101} |101> + a_{111} |110> + a_{110} |111>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto toffoli(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit1 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit2 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit1 and target_qubit != control_qubit2 and control_qubit1 != control_qubit2);

        constexpr auto num_operated_qubits = BitInteger{3u};

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        std::array<qubit_type, 3u> sorted_qubits{target_qubit, control_qubit1.qubit(), control_qubit2.qubit()};
        using std::begin;
        using std::end;
        std::sort(begin(sorted_qubits), end(sorted_qubits));

        auto const target_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
        auto const control_qubits_mask
          = ::ket::utility::integer_exp2<StateInteger>(control_qubit1)
            bitor ::ket::utility::integer_exp2<StateInteger>(control_qubit2);

        std::array<StateInteger, 4u> bits_mask{};
        bits_mask[0u] = ::ket::utility::integer_exp2<StateInteger>(sorted_qubits[0u]) - StateInteger{1u};
        bits_mask[1u]
          = (::ket::utility::integer_exp2<StateInteger>(sorted_qubits[1u] - BitInteger{1u}) - StateInteger{1u})
            xor bits_mask[0u];
        bits_mask[2u]
          = (::ket::utility::integer_exp2<StateInteger>(sorted_qubits[2u] - BitInteger{2u}) - StateInteger{1u})
            xor (bits_mask[0u] bitor bits_mask[1u]);
        bits_mask[3u] = compl (bits_mask[0u] bitor bits_mask[1u] bitor bits_mask[2u]);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_cxxx0_txxxx0_cxxx
          auto const base_index
            = ((index_wo_qubits bitand bits_mask[3u]) << 3u)
              bitor ((index_wo_qubits bitand bits_mask[2u]) << 2u)
              bitor ((index_wo_qubits bitand bits_mask[1u]) << 1u)
              bitor (index_wo_qubits bitand bits_mask[0u]);
          // xxx1_cxxx0_txxxx1_cxxx
          auto const control_on_index = base_index bitor control_qubits_mask;
          // xxx1_cxxx1_txxxx1_cxxx
          auto const target_control_on_index = control_on_index bitor target_qubit_mask;
          auto const control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, control_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const target_control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, target_control_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);

          std::iter_swap(control_on_iter, target_control_on_iter);
        }
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits>
      inline auto adj_toffoli(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      -> void
      { ::ket::gate::fused::toffoli(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, target_qubit, control_qubit1, control_qubit2); }
# else // KET_USE_BIT_MASKS_EXPLICITLY
      // TOFFOLI_{t,c1,c2}
      // TOFFOLI_{1,2,3} (a_{000} |000> + a_{001} |001> + a_{010} |010> + a_{011} |011> + a_{100} |100> + a_{101} |101> + a_{110} |110> + a_{111} |111>)
      //   = a_{000} |000> + a_{001} |001> + a_{010} |010> + a_{011} |011> + a_{100} |100> + a_{101} |101> + a_{111} |110> + a_{110} |111>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto toffoli(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit1 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit2 < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit1 and target_qubit != control_qubit2 and control_qubit1 != control_qubit2);

        constexpr auto num_operated_qubits = BitInteger{3u};

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        std::array<qubit_type, 3u> sorted_qubits{target_qubit, control_qubit1.qubit(), control_qubit2.qubit()};
        using std::begin;
        using std::end;
        std::sort(begin(sorted_qubits), end(sorted_qubits));

        auto const target_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
        auto const control_qubits_mask
          = ::ket::utility::integer_exp2<StateInteger>(control_qubit1)
            bitor ::ket::utility::integer_exp2<StateInteger>(control_qubit2);

        std::array<StateInteger, 4u> bits_mask{};
        bits_mask[0u] = ::ket::utility::integer_exp2<StateInteger>(sorted_qubits[0u]) - StateInteger{1u};
        bits_mask[1u]
          = (::ket::utility::integer_exp2<StateInteger>(sorted_qubits[1u] - BitInteger{1u}) - StateInteger{1u})
            xor bits_mask[0u];
        bits_mask[2u]
          = (::ket::utility::integer_exp2<StateInteger>(sorted_qubits[2u] - BitInteger{2u}) - StateInteger{1u})
            xor (bits_mask[0u] bitor bits_mask[1u]);
        bits_mask[3u] = compl (bits_mask[0u] bitor bits_mask[1u] bitor bits_mask[2u]);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_cxxx0_txxxx0_cxxx
          auto const base_index
            = ((index_wo_qubits bitand bits_mask[3u]) << 3u)
              bitor ((index_wo_qubits bitand bits_mask[2u]) << 2u)
              bitor ((index_wo_qubits bitand bits_mask[1u]) << 1u)
              bitor (index_wo_qubits bitand bits_mask[0u]);
          // xxx1_cxxx0_txxxx1_cxxx
          auto const control_on_index = base_index bitor control_qubits_mask;
          // xxx1_cxxx1_txxxx1_cxxx
          auto const target_control_on_index = control_on_index bitor target_qubit_mask;
          auto const control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, control_on_index, fused_qubit_masks, fused_index_masks);
          auto const target_control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, target_control_on_index, fused_qubit_masks, fused_index_masks);

          std::iter_swap(control_on_iter, target_control_on_iter);
        }
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto adj_toffoli(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      -> void
      { ::ket::gate::fused::toffoli(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, target_qubit, control_qubit1, control_qubit2); }
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_TOFFOLI_HPP
