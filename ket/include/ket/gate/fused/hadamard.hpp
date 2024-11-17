#ifndef KET_GATE_FUSED_HADAMARD_HPP
# define KET_GATE_FUSED_HADAMARD_HPP

# include <cassert>
# include <cstddef>
# include <array>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/gate/fused/gate.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
      // H_i
      // H_1 (a_0 |0> + a_1 |1>) = (a_0 + a_1)/sqrt(2) |0> + (a_0 - a_1)/sqrt(2) |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto hadamard(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

        constexpr auto num_operated_qubits = BitInteger{1u};
        assert(qubit < ::ket::make_qubit<StateInteger>(num_fused_qubits));

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubit = std::size_t{0u}; index_wo_qubit < count; ++index_wo_qubit)
        {
          // xxxxx0xxxxxx
          auto const zero_index = ((index_wo_qubit bitand upper_bits_mask) << 1u) bitor (index_wo_qubit bitand lower_bits_mask);
          // xxxxx1xxxxxx
          auto const one_index = zero_index bitor qubit_mask;
          auto const zero_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, zero_index, fused_qubit_masks, fused_index_masks);
          auto const one_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, one_index, fused_qubit_masks, fused_index_masks);
          auto const zero_iter_value = *zero_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::one_div_root_two;
          *zero_iter += *one_iter;
          *zero_iter *= one_div_root_two<real_type>();
          *one_iter = zero_iter_value - *one_iter;
          *one_iter *= one_div_root_two<real_type>();
        }
      }

      // CH_{tc} or C1H_{tc}
      // CH_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (a_{10} + a_{11})/sqrt(2) |10> + (a_{10} - a_{11})/sqrt(2) |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger>
      inline auto hadamard(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

        constexpr auto num_operated_qubits = BitInteger{2u};
        assert(target_qubit < ::ket::make_qubit<StateInteger>(num_fused_qubits));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(num_fused_qubits));

        auto const minmax_qubits = std::minmax(target_qubit, control_qubit.qubit());
        auto const target_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
        auto const control_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
        auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_txxx0_cxxx
          auto const base_index
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask);
          // xxx0_txxx1_cxxx
          auto const control_on_index = base_index bitor control_qubit_mask;
          // xxx1_txxx1_cxxx
          auto const target_control_on_index = control_on_index bitor target_qubit_mask;
          auto const control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, control_on_index, fused_qubit_masks, fused_index_masks);
          auto const target_control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, target_control_on_index, fused_qubit_masks, fused_index_masks);
          auto const control_on_iter_value = *control_on_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          using boost::math::constants::one_div_root_two;
          *control_on_iter += *target_control_on_iter;
          *control_on_iter *= one_div_root_two<real_type>();
          *target_control_on_iter = control_on_iter_value - *target_control_on_iter;
          *target_control_on_iter *= one_div_root_two<real_type>();
        }
      }

      // C...CH_{tc...c'} or CnH_{tc...c'}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto hadamard(
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
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            auto const iter0
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index0, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            auto const iter1
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index1, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            auto const iter0_value = *iter0;

            using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            using real_type = ::ket::utility::meta::real_t<complex_type>;
            using boost::math::constants::one_div_root_two;
            *iter0 += *iter1;
            *iter0 *= one_div_root_two<real_type>();
            *iter1 = iter0_value - *iter1;
            *iter1 *= one_div_root_two<real_type>();
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename BitInteger, typename... ControlQubits>
      inline auto adj_hadamard(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::hadamard(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, target_qubit, control_qubits...); }
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_HADAMARD_HPP
