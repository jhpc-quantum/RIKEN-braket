#ifndef KET_GATE_FUSED_PHASE_SHIFT_HPP
# define KET_GATE_FUSED_PHASE_SHIFT_HPP

# include <cassert>
# include <cstddef>
# include <cmath>
# include <array>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/fused/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/exp_i.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace fused
    {
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
      // phase_shift_coeff
      // U1_i(theta)
      // U1_1(theta) (a_0 |0> + a_1 |1>) = a_0 |0> + e^{i theta} a_1 |1>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Complex>
      inline auto phase_shift_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
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

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubit = std::size_t{0u}; index_wo_qubit < count; ++index_wo_qubit)
        {
          // xxxxx1xxxxxx
          auto const one_index = ((index_wo_qubit bitand upper_bits_mask) << 1u) bitor (index_wo_qubit bitand lower_bits_mask) bitor qubit_mask;
          *(first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, one_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel)) *= phase_coefficient;
        }
      }

      // CU1_{tc}(theta) or C1U1_{tc}(theta)
      // CU1_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i thta} a_{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Complex>
      inline auto phase_shift_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
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

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx1_txxx1_cxxx
          auto const target_control_on_index
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask)
              bitor control_qubit_mask bitor target_qubit_mask;
          *(first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, target_control_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel)) *= phase_coefficient;
        }
      }

      // C...CU1_{tc...c'}(theta) or CnU1_{tc...c'}(theta)
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Complex, typename... ControlQubits>
      inline auto phase_shift_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        static_assert(
          std::is_same<Complex, typename std::iterator_traits<RandomAccessIterator>::value_type>::value,
          "Complex should be the same to value_type of RandomAccessIterator");

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel, &phase_coefficient](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b11...11u
            constexpr auto index = ((std::size_t{1u} << num_operated_qubits) - std::size_t{1u});
            auto const iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                    unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
            *iter *= phase_coefficient;
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Complex, typename... ControlQubits>
      inline auto adj_phase_shift_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      { using std::conj; ::ket::gate::fused::phase_shift_coeff(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, conj(phase_coefficient), target_qubit, control_qubits...); }

      // phase_shift
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real, typename... ControlQubits>
      inline auto phase_shift(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        ::ket::gate::fused::phase_shift_coeff(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real, typename... ControlQubits>
      inline auto adj_phase_shift(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::phase_shift(first, fused_index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, -phase, target_qubit, control_qubits...); }

      // generalized phase_shift
      // U2_i(theta, theta')
      // U2_1(theta, theta') (a_0 |0> + a_1 |1>)
      //   = (a_0 - e^{i theta'} a_1)/sqrt(2) |0> + (e^{i theta} a_0 + e^{i(theta + theta')} a_1)/sqrt(2) |1>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real>
      inline auto phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

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
          auto const zero_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, zero_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const one_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, one_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const zero_iter_value = *zero_iter;

          *zero_iter -= phase_coefficient2 * *one_iter;
          *zero_iter *= one_div_root_two<Real>();
          *one_iter *= phase_coefficient2;
          *one_iter += zero_iter_value;
          *one_iter *= modified_phase_coefficient1;
        }
      }

      // CU2_{tc}(theta, theta') or C1U2_{tc}(theta, theta')
      // CU2_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (a_{10} - e^{i theta'} a_{11})/sqrt(2) |10>
      //     + (e^{i theta} a_{10} + e^{i(theta + theta')} a_{11})/sqrt(2) |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real>
      inline auto phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit);

        constexpr auto num_operated_qubits = BitInteger{2u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

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
          auto const control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, control_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const target_control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, target_control_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const control_on_iter_value = *control_on_iter;

          *control_on_iter -= phase_coefficient2 * *target_control_on_iter;
          *control_on_iter *= one_div_root_two<Real>();
          *target_control_on_iter *= phase_coefficient2;
          *target_control_on_iter += control_on_iter_value;
          *target_control_on_iter *= modified_phase_coefficient1;
        }
      }

      // C...CU2_{tc...c'}(theta, theta') or CnU2_{tc...c'}(theta, theta')
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real, typename... ControlQubits>
      inline auto phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel, &modified_phase_coefficient1, &phase_coefficient2](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            auto const control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index0, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                    unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
            auto const target_control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index1, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                    unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
            auto const control_on_iter_value = *control_on_iter;

            *control_on_iter -= phase_coefficient2 * *target_control_on_iter;
            *control_on_iter *= one_div_root_two<Real>();
            *target_control_on_iter *= phase_coefficient2;
            *target_control_on_iter += control_on_iter_value;
            *target_control_on_iter *= modified_phase_coefficient1;
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      // U2+_i(theta, theta')
      // U2+_1(theta, theta') (a_0 |0> + a_1 |1>)
      //   = (a_0 + e^{-i theta} a_1)/sqrt(2) |0>
      //     + (-e^{-i theta'} a_0 + e^{-i(theta + theta')} a_1)/sqrt(2) |1>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real>
      inline auto adj_phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

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
          auto const zero_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, zero_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const one_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, one_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const zero_iter_value = *zero_iter;

          *zero_iter += phase_coefficient1 * *one_iter;
          *zero_iter *= one_div_root_two<Real>();
          *one_iter *= phase_coefficient1;
          *one_iter -= zero_iter_value;
          *one_iter *= modified_phase_coefficient2;
        }
      }

      // CU2+_{tc}(theta, theta') or C1U2+_{tc}(theta, theta')
      // CU2+_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (a_{10} + e^{-i theta} a_{11})/sqrt(2) |10> 
      //     + (-e^{-i theta'} a_{10} + e^{-i(theta + theta')} a_{11})/sqrt(2) |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real>
      inline auto adj_phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit);

        constexpr auto num_operated_qubits = BitInteger{2u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

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
          auto const control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, control_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const target_control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, target_control_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const control_on_iter_value = *control_on_iter;

          *control_on_iter += phase_coefficient1 * *target_control_on_iter;
          *control_on_iter *= one_div_root_two<Real>();
          *target_control_on_iter *= phase_coefficient1;
          *target_control_on_iter -= control_on_iter_value;
          *target_control_on_iter *= modified_phase_coefficient2;
        }
      }

      // C...CU2+_{tc...c'}(theta, theta'), or CnU2+_{tc...c'}(theta, theta')
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real, typename... ControlQubits>
      inline auto adj_phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel, &phase_coefficient1, &modified_phase_coefficient2](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            auto const control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index0, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                    unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
            auto const target_control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index1, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                    unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
            auto const control_on_iter_value = *control_on_iter;

            *control_on_iter += phase_coefficient1 * *target_control_on_iter;
            *control_on_iter *= one_div_root_two<Real>();
            *target_control_on_iter *= phase_coefficient1;
            *target_control_on_iter -= control_on_iter_value;
            *target_control_on_iter *= modified_phase_coefficient2;
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      // U3_i(theta, theta', theta'')
      // U3_1(theta, theta', theta'') (a_0 |0> + a_1 |1>)
      //   = (cos(theta/2) a_0 - e^{i theta''} sin(theta/2) a_1) |0>
      //     + (e^{i theta'} sin(theta/2) a_0 + e^{i(theta' + theta'')} cos(theta/2) a_1) |1>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real>
      inline auto phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

        auto const sine_phase_coefficient3 = sine * phase_coefficient3;
        auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

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
          auto const zero_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, zero_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const one_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, one_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const zero_iter_value = *zero_iter;

          *zero_iter *= cosine;
          *zero_iter -= sine_phase_coefficient3 * *one_iter;
          *one_iter *= cosine_phase_coefficient3;
          *one_iter += sine * zero_iter_value;
          *one_iter *= phase_coefficient2;
        }
      }

      // CU3_{tc}(theta, theta', theta''), or C1U3_{tc}(theta, theta', theta'')
      // CU3_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} - e^{i theta''} sin(theta/2) a_{11}) |10>
      //     + (e^{i theta'} sin(theta/2) a_{10} + e^{i(theta' + theta'')} cos(theta/2) a_{11}) |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real>
      inline auto phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit);

        constexpr auto num_operated_qubits = BitInteger{2u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

        auto const sine_phase_coefficient3 = sine * phase_coefficient3;
        auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

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
          auto const control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, control_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const target_control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, target_control_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const control_on_iter_value = *control_on_iter;

          *control_on_iter *= cosine;
          *control_on_iter -= sine_phase_coefficient3 * *target_control_on_iter;
          *target_control_on_iter *= cosine_phase_coefficient3;
          *target_control_on_iter += sine * control_on_iter_value;
          *target_control_on_iter *= phase_coefficient2;
        }
      }

      // C...CU3_{tc...c'}(theta, theta', theta''), or CnU3_{tc...c'}(theta, theta', theta'')
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real, typename... ControlQubits>
      inline auto phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

        auto const sine_phase_coefficient3 = sine * phase_coefficient3;
        auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel,
           sine, cosine, &phase_coefficient2, &sine_phase_coefficient3, &cosine_phase_coefficient3](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            auto const control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index0, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                    unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
            auto const target_control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index1, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                    unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
            auto const control_on_iter_value = *control_on_iter;

            *control_on_iter *= cosine;
            *control_on_iter -= sine_phase_coefficient3 * *target_control_on_iter;
            *target_control_on_iter *= cosine_phase_coefficient3;
            *target_control_on_iter += sine * control_on_iter_value;
            *target_control_on_iter *= phase_coefficient2;
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      // U3+_i(theta, theta', theta'')
      // U3+_1(theta, theta', theta'') (a_0 |0> + a_1 |1>)
      //   = (cos(theta/2) a_0 + e^{-i theta'} sin(theta/2) a_1) |0>
      //     + (-e^{-i theta''} sin(theta/2) a_0 + e^{-i(theta' + theta'')} cos(theta/2) a_1) |1>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real>
      inline auto adj_phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

        auto const sine_phase_coefficient2 = sine * phase_coefficient2;
        auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

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
          auto const zero_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, zero_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const one_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, one_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const zero_iter_value = *zero_iter;

          *zero_iter *= cosine;
          *zero_iter += sine_phase_coefficient2 * *one_iter;
          *one_iter *= cosine_phase_coefficient2;
          *one_iter -= sine * zero_iter_value;
          *one_iter *= phase_coefficient3;
        }
      }

      // CU3+_{tc}(theta, theta', theta''), or C1U3+_{tc}(theta, theta', theta'')
      // CU3+_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} + e^{-i theta'} sin(theta/2) a_{11}) |10>
      //     + (-e^{-i theta''} sin(theta/2) a_{10} + e^{-i(theta' + theta'')} cos(theta/2) a_{11}) |11>
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real>
      inline auto adj_phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit);

        constexpr auto num_operated_qubits = BitInteger{2u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

        auto const sine_phase_coefficient2 = sine * phase_coefficient2;
        auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

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
          auto const control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, control_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const target_control_on_iter = first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, target_control_on_index, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
          auto const control_on_iter_value = *control_on_iter;

          *control_on_iter *= cosine;
          *control_on_iter += sine_phase_coefficient2 * *target_control_on_iter;
          *target_control_on_iter *= cosine_phase_coefficient2;
          *target_control_on_iter -= sine * control_on_iter_value;
          *target_control_on_iter *= phase_coefficient3;
        }
      }

      // C...CU3+_{tc...c'}(theta, theta', theta''), or CnU3+_{tc...c'}(theta, theta', theta'')
      template <typename RandomAccessIterator, typename StateInteger, typename BitInteger, std::size_t num_fused_qubits, typename Real, typename... ControlQubits>
      inline auto adj_phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits > const& unsorted_fused_qubits,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_fused_qubits + 1u> const& sorted_fused_qubits_with_sentinel,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

        auto const sine_phase_coefficient2 = sine * phase_coefficient2;
        auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &unsorted_fused_qubits, &sorted_fused_qubits_with_sentinel,
           sine, cosine, &sine_phase_coefficient2, &cosine_phase_coefficient2, phase_coefficient3](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_operated_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_operated_qubits_with_sentinel)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            auto const control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index0, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                    unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
            auto const target_control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index1, unsorted_operated_qubits, sorted_operated_qubits_with_sentinel),
                    unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);
            auto const control_on_iter_value = *control_on_iter;

            *control_on_iter *= cosine;
            *control_on_iter += sine_phase_coefficient2 * *target_control_on_iter;
            *target_control_on_iter *= cosine_phase_coefficient2;
            *target_control_on_iter -= sine * control_on_iter_value;
            *target_control_on_iter *= phase_coefficient3;
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }
# else // KET_USE_BIT_MASKS_EXPLICITLY
      // phase_shift_coeff
      // U1_i(theta)
      // U1_1(theta) (a_0 |0> + a_1 |1>) = a_0 |0> + e^{i theta} a_1 |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger>
      inline auto phase_shift_coeff(
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

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubit = std::size_t{0u}; index_wo_qubit < count; ++index_wo_qubit)
        {
          // xxxxx1xxxxxx
          auto const one_index = ((index_wo_qubit bitand upper_bits_mask) << 1u) bitor (index_wo_qubit bitand lower_bits_mask) bitor qubit_mask;
          *(first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, one_index, fused_qubit_masks, fused_index_masks)) *= phase_coefficient;
        }
      }

      // CU1_{tc}(theta) or C1U1_{tc}(theta)
      // CU1_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i thta} a_{11} |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger>
      inline auto phase_shift_coeff(
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

        constexpr auto count = ::ket::utility::integer_exp2<StateInteger>(num_fused_qubits) >> num_operated_qubits;
        for (auto index_wo_qubits = std::size_t{0u}; index_wo_qubits < count; ++index_wo_qubits)
        {
          // xxx1_txxx1_cxxx
          auto const target_control_on_index
            = ((index_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((index_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (index_wo_qubits bitand lower_bits_mask)
              bitor control_qubit_mask bitor target_qubit_mask;
          *(first + ::ket::gate::utility::index_with_qubits(fused_index_wo_qubits, target_control_on_index, fused_qubit_masks, fused_index_masks)) *= phase_coefficient;
        }
      }

      // C...CU1_{tc...c'}(theta) or CnU1_{tc...c'}(theta)
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift_coeff(
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

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks, &phase_coefficient](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...11u
            constexpr auto index = ((std::size_t{1u} << num_operated_qubits) - std::size_t{1u});
            auto const iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            *iter *= phase_coefficient;
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Complex, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift_coeff(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      { using std::conj; ::ket::gate::fused::phase_shift_coeff(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, conj(phase_coefficient), target_qubit, control_qubits...); }

      // phase_shift
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        ::ket::gate::fused::phase_shift_coeff(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits...);
      }

      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> void
      { ::ket::gate::fused::phase_shift(first, fused_index_wo_qubits, fused_qubit_masks, fused_index_masks, -phase, target_qubit, control_qubits...); }

      // generalized phase_shift
      // U2_i(theta, theta')
      // U2_1(theta, theta') (a_0 |0> + a_1 |1>)
      //   = (a_0 - e^{i theta'} a_1)/sqrt(2) |0> + (e^{i theta} a_0 + e^{i(theta + theta')} a_1)/sqrt(2) |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger>
      inline auto phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

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

          *zero_iter -= phase_coefficient2 * *one_iter;
          *zero_iter *= one_div_root_two<Real>();
          *one_iter *= phase_coefficient2;
          *one_iter += zero_iter_value;
          *one_iter *= modified_phase_coefficient1;
        }
      }

      // CU2_{tc}(theta, theta') or C1U2_{tc}(theta, theta')
      // CU2_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (a_{10} - e^{i theta'} a_{11})/sqrt(2) |10>
      //     + (e^{i theta} a_{10} + e^{i(theta + theta')} a_{11})/sqrt(2) |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger>
      inline auto phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit);

        constexpr auto num_operated_qubits = BitInteger{2u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

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

          *control_on_iter -= phase_coefficient2 * *target_control_on_iter;
          *control_on_iter *= one_div_root_two<Real>();
          *target_control_on_iter *= phase_coefficient2;
          *target_control_on_iter += control_on_iter_value;
          *target_control_on_iter *= modified_phase_coefficient1;
        }
      }

      // C...CU2_{tc...c'}(theta, theta') or CnU2_{tc...c'}(theta, theta')
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks, &modified_phase_coefficient1, &phase_coefficient2](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            auto const control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index0, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            auto const target_control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index1, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            auto const control_on_iter_value = *control_on_iter;

            *control_on_iter -= phase_coefficient2 * *target_control_on_iter;
            *control_on_iter *= one_div_root_two<Real>();
            *target_control_on_iter *= phase_coefficient2;
            *target_control_on_iter += control_on_iter_value;
            *target_control_on_iter *= modified_phase_coefficient1;
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      // U2+_i(theta, theta')
      // U2+_1(theta, theta') (a_0 |0> + a_1 |1>)
      //   = (a_0 + e^{-i theta} a_1)/sqrt(2) |0>
      //     + (-e^{-i theta'} a_0 + e^{-i(theta + theta')} a_1)/sqrt(2) |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger>
      inline auto adj_phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

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

          *zero_iter += phase_coefficient1 * *one_iter;
          *zero_iter *= one_div_root_two<Real>();
          *one_iter *= phase_coefficient1;
          *one_iter -= zero_iter_value;
          *one_iter *= modified_phase_coefficient2;
        }
      }

      // CU2+_{tc}(theta, theta') or C1U2+_{tc}(theta, theta')
      // CU2+_{1,2}(theta, theta') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (a_{10} + e^{-i theta} a_{11})/sqrt(2) |10> 
      //     + (-e^{-i theta'} a_{10} + e^{-i(theta + theta')} a_{11})/sqrt(2) |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger>
      inline auto adj_phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit);

        constexpr auto num_operated_qubits = BitInteger{2u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

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

          *control_on_iter += phase_coefficient1 * *target_control_on_iter;
          *control_on_iter *= one_div_root_two<Real>();
          *target_control_on_iter *= phase_coefficient1;
          *target_control_on_iter -= control_on_iter_value;
          *target_control_on_iter *= modified_phase_coefficient2;
        }
      }

      // C...CU2+_{tc...c'}(theta, theta'), or CnU2+_{tc...c'}(theta, theta')
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift2(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

        auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

        using boost::math::constants::one_div_root_two;
        auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks, &phase_coefficient1, &modified_phase_coefficient2](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            auto const control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index0, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            auto const target_control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index1, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            auto const control_on_iter_value = *control_on_iter;

            *control_on_iter += phase_coefficient1 * *target_control_on_iter;
            *control_on_iter *= one_div_root_two<Real>();
            *target_control_on_iter *= phase_coefficient1;
            *target_control_on_iter -= control_on_iter_value;
            *target_control_on_iter *= modified_phase_coefficient2;
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      // U3_i(theta, theta', theta'')
      // U3_1(theta, theta', theta'') (a_0 |0> + a_1 |1>)
      //   = (cos(theta/2) a_0 - e^{i theta''} sin(theta/2) a_1) |0>
      //     + (e^{i theta'} sin(theta/2) a_0 + e^{i(theta' + theta'')} cos(theta/2) a_1) |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger>
      inline auto phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

        auto const sine_phase_coefficient3 = sine * phase_coefficient3;
        auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

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

          *zero_iter *= cosine;
          *zero_iter -= sine_phase_coefficient3 * *one_iter;
          *one_iter *= cosine_phase_coefficient3;
          *one_iter += sine * zero_iter_value;
          *one_iter *= phase_coefficient2;
        }
      }

      // CU3_{tc}(theta, theta', theta''), or C1U3_{tc}(theta, theta', theta'')
      // CU3_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} - e^{i theta''} sin(theta/2) a_{11}) |10>
      //     + (e^{i theta'} sin(theta/2) a_{10} + e^{i(theta' + theta'')} cos(theta/2) a_{11}) |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger>
      inline auto phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit);

        constexpr auto num_operated_qubits = BitInteger{2u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

        auto const sine_phase_coefficient3 = sine * phase_coefficient3;
        auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

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

          *control_on_iter *= cosine;
          *control_on_iter -= sine_phase_coefficient3 * *target_control_on_iter;
          *target_control_on_iter *= cosine_phase_coefficient3;
          *target_control_on_iter += sine * control_on_iter_value;
          *target_control_on_iter *= phase_coefficient2;
        }
      }

      // C...CU3_{tc...c'}(theta, theta', theta''), or CnU3_{tc...c'}(theta, theta', theta'')
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

        auto const sine_phase_coefficient3 = sine * phase_coefficient3;
        auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks,
           sine, cosine, &phase_coefficient2, &sine_phase_coefficient3, &cosine_phase_coefficient3](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            auto const control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index0, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            auto const target_control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index1, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            auto const control_on_iter_value = *control_on_iter;

            *control_on_iter *= cosine;
            *control_on_iter -= sine_phase_coefficient3 * *target_control_on_iter;
            *target_control_on_iter *= cosine_phase_coefficient3;
            *target_control_on_iter += sine * control_on_iter_value;
            *target_control_on_iter *= phase_coefficient2;
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }

      // U3+_i(theta, theta', theta'')
      // U3+_1(theta, theta', theta'') (a_0 |0> + a_1 |1>)
      //   = (cos(theta/2) a_0 + e^{-i theta'} sin(theta/2) a_1) |0>
      //     + (-e^{-i theta''} sin(theta/2) a_0 + e^{-i(theta' + theta'')} cos(theta/2) a_1) |1>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger>
      inline auto adj_phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));

        constexpr auto num_operated_qubits = BitInteger{1u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

        auto const sine_phase_coefficient2 = sine * phase_coefficient2;
        auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

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

          *zero_iter *= cosine;
          *zero_iter += sine_phase_coefficient2 * *one_iter;
          *one_iter *= cosine_phase_coefficient2;
          *one_iter -= sine * zero_iter_value;
          *one_iter *= phase_coefficient3;
        }
      }

      // CU3+_{tc}(theta, theta', theta''), or C1U3+_{tc}(theta, theta', theta'')
      // CU3+_{1,2}(theta, theta', theta'') (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + (cos(theta/2) a_{10} + e^{-i theta'} sin(theta/2) a_{11}) |10>
      //     + (-e^{-i theta''} sin(theta/2) a_{10} + e^{-i(theta' + theta'')} cos(theta/2) a_{11}) |11>
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger>
      inline auto adj_phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
      -> void
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
        assert(target_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(control_qubit < ::ket::make_qubit<StateInteger>(static_cast<BitInteger>(num_fused_qubits)));
        assert(target_qubit != control_qubit);

        constexpr auto num_operated_qubits = BitInteger{2u};

        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

        auto const sine_phase_coefficient2 = sine * phase_coefficient2;
        auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

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

          *control_on_iter *= cosine;
          *control_on_iter += sine_phase_coefficient2 * *target_control_on_iter;
          *target_control_on_iter *= cosine_phase_coefficient2;
          *target_control_on_iter -= sine * control_on_iter_value;
          *target_control_on_iter *= phase_coefficient3;
        }
      }

      // C...CU3+_{tc...c'}(theta, theta', theta''), or CnU3+_{tc...c'}(theta, theta', theta'')
      template <typename RandomAccessIterator, typename StateInteger, std::size_t num_fused_qubits, typename Real, typename BitInteger, typename... ControlQubits>
      inline auto adj_phase_shift3(
        RandomAccessIterator const first, StateInteger const fused_index_wo_qubits,
        std::array<StateInteger, num_fused_qubits> const& fused_qubit_masks, std::array<StateInteger, num_fused_qubits + 1u> const& fused_index_masks,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2, ControlQubits const... control_qubits)
      -> void
      {
        using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
        static_assert(
          std::is_same<Real, ::ket::utility::meta::real_t<complex_type>>::value,
          "Real must be the same to \"real part\" of value_type of RandomAccessIterator");

        constexpr auto num_control_qubits = static_cast<BitInteger>(sizeof...(ControlQubits) + 2u);
        constexpr auto num_operated_qubits = num_control_qubits + BitInteger{1u};

        using std::cos;
        using std::sin;
        using boost::math::constants::half;
        auto const sine = sin(half<Real>() * phase1);
        auto const cosine = cos(half<Real>() * phase1);

        auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
        auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

        auto const sine_phase_coefficient2 = sine * phase_coefficient2;
        auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

        ::ket::gate::fused::gate<num_fused_qubits>(
          first,
          [fused_index_wo_qubits, &fused_qubit_masks, &fused_index_masks,
           sine, cosine, &sine_phase_coefficient2, &cosine_phase_coefficient2, phase_coefficient3](
            auto const first, StateInteger const operated_index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& operated_qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& operated_index_masks)
          {
            // 0b11...10u
            constexpr auto index0 = ((std::size_t{1u} << num_control_qubits) - std::size_t{1u}) << std::size_t{1u};
            // 0b11...11u
            constexpr auto index1 = index0 bitor std::size_t{1u};

            auto const control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index0, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            auto const target_control_on_iter
              = first
                + ::ket::gate::utility::index_with_qubits(
                    fused_index_wo_qubits,
                    ::ket::gate::utility::index_with_qubits(operated_index_wo_qubits, index1, operated_qubit_masks, operated_index_masks),
                    fused_qubit_masks, fused_index_masks);
            auto const control_on_iter_value = *control_on_iter;

            *control_on_iter *= cosine;
            *control_on_iter += sine_phase_coefficient2 * *target_control_on_iter;
            *target_control_on_iter *= cosine_phase_coefficient2;
            *target_control_on_iter -= sine * control_on_iter_value;
            *target_control_on_iter *= phase_coefficient3;
          },
          target_qubit, control_qubit1, control_qubit2, control_qubits...);
      }
# endif // KET_USE_BIT_MASKS_EXPLICITLY
    } // namespace fused
  } // namespace gate
} // namespace ket


#endif // KET_GATE_FUSED_PHASE_SHIFT_HPP
