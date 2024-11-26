#ifndef KET_GATE_FUSED_CONTROLLED_V_HPP
# define KET_GATE_FUSED_CONTROLLED_V_HPP

# include <cassert>
# include <cstddef>
# include <complex>
# include <array>
# include <algorithm>
# include <iterator>
# include <type_traits>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/fused/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/exp_i.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
      // controlled_v_coeff
      // V_{tc}(theta) or CV_{tc}(theta)
      // V_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + [a_{10} (1+e^{i theta})/2 + a_{11} (1-e^{i theta})/2] |10> + [a_{10} (1-e^{i theta})/2 + a_{11} (1+e^{i theta})/2] |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger>
      inline auto controlled_v_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit);

        constexpr auto num_operated_qubits = BitInteger{2u};

        using real_type = ::ket::utility::meta::real_t<Complex>;
        auto const one_plus_phase_coefficient = Complex{real_type{1}} + phase_coefficient;
        auto const one_minus_phase_coefficient = Complex{real_type{1}} - phase_coefficient;

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

          using boost::math::constants::half;
          *control_on_iter
            = half<real_type>()
              * (one_plus_phase_coefficient * control_on_iter_value
                 + one_minus_phase_coefficient * (*target_control_on_iter));
          *target_control_on_iter
            = half<real_type>()
              * (one_minus_phase_coefficient * control_on_iter_value
                 + one_plus_phase_coefficient * (*target_control_on_iter));
        }
      }

      // C...CU_{tc...c'}(theta) or CnU_{tc...c'}(theta)
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger, typename... ControlQubits>
      inline auto controlled_v_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");

        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

        using real_type = ::ket::utility::meta::real_t<Complex>;
        auto const one_plus_phase_coefficient = Complex{real_type{1}} + phase_coefficient;
        auto const one_minus_phase_coefficient = Complex{real_type{1}} - phase_coefficient;
        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks,
           one_plus_phase_coefficient, one_minus_phase_coefficient](
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
            auto const value0 = *iter0;

            using boost::math::constants::half;
            *iter0 = half<real_type>() * (one_plus_phase_coefficient * value0 + one_minus_phase_coefficient * *iter1);
            *iter1 = half<real_type>() * (one_minus_phase_coefficient * value0 + one_plus_phase_coefficient * *iter1);
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_v_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      { using std::conj; ::ket::gate::fused::controlled_v_coeff(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, conj(phase_coefficient), target_qubit, control_qubits...); }

      // controlled_v
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto controlled_v(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        ::ket::gate::fused::controlled_v_coeff(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto adj_controlled_v(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::controlled_v(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, -phase, target_qubit, control_qubits...); }
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_CONTROLLED_V_HPP
