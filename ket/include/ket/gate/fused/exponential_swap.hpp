#ifndef KET_GATE_FUSED_EXPONENTIAL_SWAP_HPP
# define KET_GATE_FUSED_EXPONENTIAL_SWAP_HPP

# include <cassert>
# include <cstddef>
# include <cmath>
# include <complex>
# include <array>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/fused/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/exp_i.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
      // exponential_swap_coeff
      // eSWAP_{ij}(s) = exp(is SWAP_{ij}) = I cos s + i SWAP_{ij} sin s
      // eSWAP_{1,2}(s) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = e^{is} a_{00} |00> + (cos s a_{01} + i sin s a_{10}) |01> + (i sin s a_{01} + cos s a_{10}) |10> + e^{is} a_{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger>
      inline auto exponential_swap_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");
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

        using std::imag;
        auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx0_1xxx0_2xxx
          auto const base_index
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask);
          auto const off_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, base_index, fused_qubit_masks, fused_index_masks);
          // xxx1_1xxx0_2xxx
          auto const qubit1_on_index = base_index bitor qubit1_mask;
          auto const qubit1_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, qubit1_on_index, fused_qubit_masks, fused_index_masks);
          // xxx0_1xxx1_2xxx
          auto const qubit2_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, base_index bitor qubit2_mask, fused_qubit_masks, fused_index_masks);
          // xxx1_1xxx1_2xxx
          auto const qubit12_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, qubit1_on_index bitor qubit2_mask, fused_qubit_masks, fused_index_masks);

          *off_iter *= phase_coefficient;
          *qubit12_on_iter *= phase_coefficient;

          auto const qubit1_on_iter_value = *qubit1_on_iter;
          using std::real;
          using std::imag;
          *qubit1_on_iter *= real(phase_coefficient);
          *qubit1_on_iter += *qubit2_on_iter * i_sin_theta;
          *qubit2_on_iter *= real(phase_coefficient);
          *qubit2_on_iter += qubit1_on_iter_value * i_sin_theta;
        }
      }

      // C...CeSWAP_{tt'c...c'}(s) = C...C[exp(is SWAP_{tt'})]_{c...c'} = C...C[I cos s + i SWAP_{tt'} sin s]_{c...c'}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger, typename... ControlQubits>
      inline auto exponential_swap_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      -> void
      {
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");

        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 1u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{2u};

        using std::imag;
        auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * imag(phase_coefficient);

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks, &phase_coefficient, &i_sin_theta](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...100u
            constexpr auto index00 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << BitInteger{2u};
            auto const iter00
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index00, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            // 0b11...101u
            auto const iter01
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index00 bitor std::size_t{1u}, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            // 0b11...110u
            constexpr auto index10 = index00 bitor (std::size_t{1u} << BitInteger{1u});
            auto const iter10
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index10, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            // 0b11...111u
            auto const iter11
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index10 bitor std::size_t{1u}, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);

            *iter00 *= phase_coefficient;
            *iter11 *= phase_coefficient;

            auto const value01 = *iter01;
            using std::real;
            using std::imag;
            *iter01 *= real(phase_coefficient);
            *iter01 += *iter10 * i_sin_theta;
            *iter10 *= real(phase_coefficient);
            *iter10 += value01 * i_sin_theta;
          },
          target_qubit1, target_qubit2, control_qubit, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger, typename... ControlQubits>
      inline auto adj_exponential_swap_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> void
      { using std::conj; ::ket::gate::fused::exponential_swap_coeff(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, conj(phase_coefficient), target_qubit1, target_qubit2, control_qubits...); }

      // exponential_swap
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto exponential_swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        ::ket::gate::fused::exponential_swap_coeff(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, ::ket::utility::exp_i<complex_type>(phase), target_qubit1, target_qubit2, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto adj_exponential_swap(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1, ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::exponential_swap(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, -phase, target_qubit1, target_qubit2, control_qubits...); }
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_EXPONENTIAL_SWAP_HPP
