#ifndef BRA_FUSED_GATE_APPLY_TWO_HADAMARDS_AND_PHASE_SHIFT_TERMS_HPP
# define BRA_FUSED_GATE_APPLY_TWO_HADAMARDS_AND_PHASE_SHIFT_TERMS_HPP

# include <algorithm>
# include <cassert>
# include <cstddef>
# include <iterator>
# include <vector>

# include <boost/math/constants/constants.hpp>
# include <boost/range/size.hpp>

# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>

# include <bra/types.hpp>
# include <bra/fused_gate/apply_phase_shift_terms.hpp>
# include <bra/fused_gate/fused_gate.hpp>


namespace bra
{
  namespace fused_gate
  {
    namespace apply_two_hadamards_and_phase_shift_terms_detail
    {
      inline auto insert_zero_bit(
        ::bra::state_integer_type const index, ::bra::bit_integer_type const bit)
      -> ::bra::state_integer_type
      {
        auto const lower_bits_mask
          = ::ket::utility::integer_exp2< ::bra::state_integer_type >(bit)
            - ::bra::state_integer_type{1u};
        return ((index bitand compl lower_bits_mask) << 1u) bitor (index bitand lower_bits_mask);
      }

      template <typename Iterator, typename QubitsRange1, typename QubitsRange2>
      inline auto apply_one(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        QubitsRange1 const& unsorted_fused_qubits_or_masks,
        QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
        bool const is_fused_index_identity,
        ::bra::bit_integer_type const lower_target_qubit,
        ::bra::bit_integer_type const upper_target_qubit,
        ::bra::state_integer_type const target_qubit_mask1,
        ::bra::state_integer_type const target_qubit_mask2,
        ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table const& phase_shift_table,
        ::bra::state_integer_type const index_wo_qubits) -> void
      {
        auto const fused_index00
          = ::bra::fused_gate::apply_two_hadamards_and_phase_shift_terms_detail::insert_zero_bit(
              ::bra::fused_gate::apply_two_hadamards_and_phase_shift_terms_detail::insert_zero_bit(
                index_wo_qubits, lower_target_qubit),
              upper_target_qubit);
        auto const fused_index10 = fused_index00 bitor target_qubit_mask1;
        auto const fused_index01 = fused_index00 bitor target_qubit_mask2;
        auto const fused_index11 = fused_index10 bitor target_qubit_mask2;

        auto const index00
          = is_fused_index_identity
            ? fused_index00
            : ::ket::gate::utility::ranges::index_with_qubits(
                fused_index_wo_qubits, fused_index00,
                unsorted_fused_qubits_or_masks,
                sorted_fused_qubits_with_sentinel_or_index_masks);
        auto const index10
          = is_fused_index_identity
            ? fused_index10
            : ::ket::gate::utility::ranges::index_with_qubits(
                fused_index_wo_qubits, fused_index10,
                unsorted_fused_qubits_or_masks,
                sorted_fused_qubits_with_sentinel_or_index_masks);
        auto const index01
          = is_fused_index_identity
            ? fused_index01
            : ::ket::gate::utility::ranges::index_with_qubits(
                fused_index_wo_qubits, fused_index01,
                unsorted_fused_qubits_or_masks,
                sorted_fused_qubits_with_sentinel_or_index_masks);
        auto const index11
          = is_fused_index_identity
            ? fused_index11
            : ::ket::gate::utility::ranges::index_with_qubits(
                fused_index_wo_qubits, fused_index11,
                unsorted_fused_qubits_or_masks,
                sorted_fused_qubits_with_sentinel_or_index_masks);

        auto const iter00 = first + index00;
        auto const iter10 = first + index10;
        auto const iter01 = first + index01;
        auto const iter11 = first + index11;
        using complex_type = typename std::iterator_traits<Iterator>::value_type;
        using real_type = ::ket::utility::meta::real_t<complex_type>;
        using boost::math::constants::one_div_root_two;
        auto const one_div_root_two_value = one_div_root_two<real_type>();

        auto value00 = (*iter00 + *iter10) * one_div_root_two_value;
        auto value10 = (*iter00 - *iter10) * one_div_root_two_value;
        auto value01 = (*iter01 + *iter11) * one_div_root_two_value;
        auto value11 = (*iter01 - *iter11) * one_div_root_two_value;

        value00 *= phase_shift_table.coefficients[static_cast<std::size_t>(
          ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_index(
            phase_shift_table, fused_index00))];
        value10 *= phase_shift_table.coefficients[static_cast<std::size_t>(
          ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_index(
            phase_shift_table, fused_index10))];
        value01 *= phase_shift_table.coefficients[static_cast<std::size_t>(
          ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_index(
            phase_shift_table, fused_index01))];
        value11 *= phase_shift_table.coefficients[static_cast<std::size_t>(
          ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_index(
            phase_shift_table, fused_index11))];

        *iter00 = (value00 + value01) * one_div_root_two_value;
        *iter01 = (value00 - value01) * one_div_root_two_value;
        *iter10 = (value10 + value11) * one_div_root_two_value;
        *iter11 = (value10 - value11) * one_div_root_two_value;
      }
    } // namespace apply_two_hadamards_and_phase_shift_terms_detail

    template <typename Iterator, typename QubitsRange1, typename QubitsRange2>
    inline auto apply_two_hadamards_and_phase_shift_terms(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      QubitsRange1 const& unsorted_fused_qubits_or_masks,
      QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
      ::bra::bit_integer_type const target_qubit1,
      ::bra::bit_integer_type const target_qubit2,
      std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms) -> void
    {
      auto const num_fused_qubits
        = static_cast< ::bra::bit_integer_type >(boost::size(unsorted_fused_qubits_or_masks));
      assert(target_qubit1 < num_fused_qubits);
      assert(target_qubit2 < num_fused_qubits);
      assert(target_qubit1 != target_qubit2);
      auto const is_fused_index_identity
        = ::bra::fused_gate::apply_phase_shift_terms_detail::is_fused_index_identity(
            fused_index_wo_qubits, unsorted_fused_qubits_or_masks);
      auto const lower_target_qubit = std::min(target_qubit1, target_qubit2);
      auto const upper_target_qubit = std::max(target_qubit1, target_qubit2);
      auto const target_qubit_mask1
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(target_qubit1);
      auto const target_qubit_mask2
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(target_qubit2);
      auto const count
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(
            num_fused_qubits - ::bra::bit_integer_type{2u});
      auto const phase_shift_table
        = ::bra::fused_gate::apply_phase_shift_terms_detail::make_phase_shift_table(phase_shift_terms);
      assert(not phase_shift_table.coefficients.empty());
      for (auto index_wo_qubits = ::bra::state_integer_type{0u}; index_wo_qubits < count; ++index_wo_qubits)
        ::bra::fused_gate::apply_two_hadamards_and_phase_shift_terms_detail::apply_one(
          first, fused_index_wo_qubits,
          unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
          is_fused_index_identity,
          lower_target_qubit, upper_target_qubit, target_qubit_mask1, target_qubit_mask2,
          phase_shift_table, index_wo_qubits);
    }

    template <typename ParallelPolicy, typename Iterator, typename QubitsRange1, typename QubitsRange2>
    inline auto apply_two_hadamards_and_phase_shift_terms(
      ParallelPolicy const parallel_policy, int const thread_index,
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      QubitsRange1 const& unsorted_fused_qubits_or_masks,
      QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
      ::bra::bit_integer_type const target_qubit1,
      ::bra::bit_integer_type const target_qubit2,
      std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms) -> void
    {
      auto const num_fused_qubits
        = static_cast< ::bra::bit_integer_type >(boost::size(unsorted_fused_qubits_or_masks));
      assert(target_qubit1 < num_fused_qubits);
      assert(target_qubit2 < num_fused_qubits);
      assert(target_qubit1 != target_qubit2);
      auto const is_fused_index_identity
        = ::bra::fused_gate::apply_phase_shift_terms_detail::is_fused_index_identity(
            fused_index_wo_qubits, unsorted_fused_qubits_or_masks);
      auto const lower_target_qubit = std::min(target_qubit1, target_qubit2);
      auto const upper_target_qubit = std::max(target_qubit1, target_qubit2);
      auto const target_qubit_mask1
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(target_qubit1);
      auto const target_qubit_mask2
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(target_qubit2);
      auto const count
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(
            num_fused_qubits - ::bra::bit_integer_type{2u});
      auto const phase_shift_table
        = ::bra::fused_gate::apply_phase_shift_terms_detail::make_phase_shift_table(phase_shift_terms);
      assert(not phase_shift_table.coefficients.empty());
      ::ket::utility::loop_n_in_execute(
        parallel_policy, count, thread_index,
        [first, fused_index_wo_qubits,
         &unsorted_fused_qubits_or_masks, &sorted_fused_qubits_with_sentinel_or_index_masks,
         is_fused_index_identity,
         lower_target_qubit, upper_target_qubit, target_qubit_mask1, target_qubit_mask2,
         &phase_shift_table](::bra::state_integer_type const index_wo_qubits, int const)
        {
          ::bra::fused_gate::apply_two_hadamards_and_phase_shift_terms_detail::apply_one(
            first, fused_index_wo_qubits,
            unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
            is_fused_index_identity,
            lower_target_qubit, upper_target_qubit, target_qubit_mask1, target_qubit_mask2,
            phase_shift_table, index_wo_qubits);
        });
    }
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_APPLY_TWO_HADAMARDS_AND_PHASE_SHIFT_TERMS_HPP
