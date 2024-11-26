#ifndef KET_GATE_FUSED_EXPONENTIAL_PAULI_Y_HPP
# define KET_GATE_FUSED_EXPONENTIAL_PAULI_Y_HPP

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
# include <ket/gate/meta/num_control_qubits.hpp>
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
      // exponential_pauli_y_coeff
      // eY_i(theta) = exp(i theta Y_i) = I cos(theta) + i Y_i sin(theta), or eY1_i(theta)
      // eY_1(theta) (a_0 |0> + a_1 |1>) = (cos(theta) a_0 + sin(theta) a_1) |0> + (-sin(theta) a_0 + cos(theta) a_1) |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger>
      inline auto exponential_pauli_y_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");
        assert(qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        auto const qubit_mask = ::ket::utility::integer_exp2<StateInteger>(qubit);
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        using std::real;
        using std::imag;
        auto const cos_theta = real(phase_coefficient);
        auto const sin_theta = imag(phase_coefficient);

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

          *zero_iter *= cos_theta;
          *zero_iter += *one_iter * sin_theta;
          *one_iter *= cos_theta;
          *one_iter -= zero_iter_value * sin_theta;
        }
      }

      // eYY_{ij}(theta) = exp(i theta Y_i Y_j) = I cos(theta) + i Y_i Y_j sin(theta), or eY2_{ij}(theta)
      // eYY_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = (cos(theta) a_{00} - i sin(theta) a_{11}) |00> + (cos(theta) a_{01} + i sin(theta) a_{10}) |01>
      //     + (i sin(theta) a_{01} + cos(theta) a_{10}) |10> + (-i sin(theta) a_{00} + cos(theta) a_{11}) |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger>
      inline auto exponential_pauli_y_coeff(
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

        using std::real;
        using std::imag;
        auto const cos_theta = real(phase_coefficient);
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

          auto const off_iter_value = *off_iter;
          auto const qubit1_on_iter_value = *qubit1_on_iter;

          *off_iter *= cos_theta;
          *off_iter -= *qubit12_on_iter * i_sin_theta;
          *qubit1_on_iter *= cos_theta;
          *qubit1_on_iter += *qubit2_on_iter * i_sin_theta;
          *qubit2_on_iter *= cos_theta;
          *qubit2_on_iter += qubit1_on_iter_value * i_sin_theta;
          *qubit12_on_iter *= cos_theta;
          *qubit12_on_iter -= off_iter_value * i_sin_theta;
        }
      }

      // CeY_{tc}(theta) = C[exp(i theta Y_t)]_c = C[I cos(theta) + i Y_t sin(theta)]_c, C1eY_{tc}(theta), CeY1_{tc}(theta), or C1eY1_{tc}(theta)
      // CeY_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (cos(theta) a_{10} + sin(theta) a_{11}) |10> + (-sin(theta) a_{10} + cos(theta) a_{11}) |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger>
      inline auto exponential_pauli_y_coeff(
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

        auto const minmax_qubits = std::minmax(target_qubit, control_qubit.qubit());
        auto const target_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
        auto const control_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(control_qubit);
        auto const lower_bits_mask = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - BitInteger{1u}) - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

        using std::real;
        using std::imag;
        auto const cos_theta = real(phase_coefficient);
        auto const sin_theta = imag(phase_coefficient);

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

          *control_on_iter *= cos_theta;
          *control_on_iter += *target_control_on_iter * sin_theta;
          *target_control_on_iter *= cos_theta;
          *target_control_on_iter -= control_on_iter_value * sin_theta;
        }
      }

      // C...CeY...Y_{t...t'c...c'}(theta) = C...C[exp(i theta Y_t ... Y_t')]_{c...c'} = C...C[I cos(theta) + i Y_t ... Y_t' sin(theta)]_{c...c'}, CneY...Y_{...}, C...CeYm_{...}, or CneYm_{...}
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger, typename Qubit2, typename Qubit3, typename... Qubits>
      inline auto exponential_pauli_y_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
      -> void
      {
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");

        constexpr auto num_operated_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 3u);
        constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubit2, Qubit3, Qubits...>::value;
        constexpr auto num_target_qubits = num_operated_qubits - num_control_qubits;
        constexpr auto num_target_indices = ::ket::utility::integer_exp2<std::size_t>(num_target_qubits);
        constexpr auto half_num_target_indices = num_target_indices / std::size_t{2u};

        using std::real;
        using std::imag;
        auto const cos_theta = real(phase_coefficient);
        auto const sin_theta = static_cast<Complex>(imag(phase_coefficient));
        auto const i_sin_theta = ::ket::utility::imaginary_unit<Complex>() * sin_theta;

        constexpr auto residual = num_operated_qubits % BitInteger{4u};
        auto const sin_part
          = residual == BitInteger{0u}
            ? i_sin_theta
            : residual == BitInteger{1u}
              ? -sin_theta
              : residual == BitInteger{2u}
                ? -i_sin_theta
                : sin_theta;

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks, cos_theta, &sin_part](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b1...10...0u
            constexpr auto base_index = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << num_target_qubits;

            for (auto i = std::size_t{0u}; i < half_num_target_indices; ++i)
            {
              auto const j = num_target_indices - std::size_t{1u} - i;

              auto num_ones_in_i = BitInteger{0u};
              auto num_ones_in_j = BitInteger{0u};
              auto i_tmp = i;
              auto j_tmp = j;
              for (auto count = BitInteger{0u}; count < num_target_qubits; ++count)
              {
                if ((i_tmp bitand std::size_t{1u}) == std::size_t{1u})
                  ++num_ones_in_i;
                if ((j_tmp bitand std::size_t{1u}) == std::size_t{1u})
                  ++num_ones_in_j;

                i_tmp >>= BitInteger{1u};
                j_tmp >>= BitInteger{1u};
              }

              auto const iter1
                = first
                  + ::ket::gate::utility::index_with_qubits(
                      fused_index_wo_qubits,
                      ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, base_index + i, operated_qubit_masks, operated_index_masks),
                      fused_qubit_masks, fused_index_masks);
              auto const iter2
                = first
                  + ::ket::gate::utility::index_with_qubits(
                      fused_index_wo_qubits,
                      ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, base_index + j, operated_qubit_masks, operated_index_masks),
                      fused_qubit_masks, fused_index_masks);
              auto const value1 = *iter1;

              *iter1 *= cos_theta;
              *iter1 += (num_target_qubits - num_ones_in_i) % BitInteger{2u} == BitInteger{0u} ? *iter2 * sin_part : *iter2 * (-sin_part);
              *iter2 *= cos_theta;
              *iter2 += (num_target_qubits - num_ones_in_j) % BitInteger{2u} == BitInteger{0u} ? value1 * sin_part : value1 * (-sin_part);
            }
          },
          qubit1, qubit2, qubit3, qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger, typename... Qubits>
      inline auto adj_exponential_pauli_y_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> void
      { using std::conj; ::ket::gate::fused::exponential_pauli_y_coeff(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, conj(phase_coefficient), qubit, qubits...); }

      // exponential_pauli_y
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... Qubits>
      inline auto exponential_pauli_y(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        ::ket::gate::fused::exponential_pauli_y_coeff(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, ::ket::utility::exp_i<complex_type>(phase), qubit, qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... Qubits>
      inline auto adj_exponential_pauli_y(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> void
      { ::ket::gate::fused::exponential_pauli_y(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, -phase, qubit, qubits...); }
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_EXPONENTIAL_PAULI_Y_HPP
