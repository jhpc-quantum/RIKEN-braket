#ifndef BRA_FUSED_GATE_APPLY_HADAMARD_AND_PHASE_SHIFT_TERMS_HPP
# define BRA_FUSED_GATE_APPLY_HADAMARD_AND_PHASE_SHIFT_TERMS_HPP

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
    namespace apply_hadamard_and_phase_shift_terms_detail
    {
      template <typename Iterator, typename QubitsRange1, typename QubitsRange2>
      inline auto apply_one(
        Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
        QubitsRange1 const& unsorted_fused_qubits_or_masks,
        QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
        bool const is_fused_index_identity,
        ::bra::state_integer_type const target_qubit_mask,
        ::bra::state_integer_type const lower_bits_mask,
        ::bra::state_integer_type const upper_bits_mask,
        ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table const& phase_shift_table,
        ::bra::state_integer_type const index_wo_qubit) -> void
      {
        auto const fused_index0
          = ((index_wo_qubit bitand upper_bits_mask) << 1u)
            bitor (index_wo_qubit bitand lower_bits_mask);
        auto const fused_index1 = fused_index0 bitor target_qubit_mask;
        auto const index0
          = is_fused_index_identity
            ? fused_index0
            : ::ket::gate::utility::ranges::index_with_qubits(
                fused_index_wo_qubits, fused_index0,
                unsorted_fused_qubits_or_masks,
                sorted_fused_qubits_with_sentinel_or_index_masks);
        auto const index1
          = is_fused_index_identity
            ? fused_index1
            : ::ket::gate::utility::ranges::index_with_qubits(
                fused_index_wo_qubits, fused_index1,
                unsorted_fused_qubits_or_masks,
                sorted_fused_qubits_with_sentinel_or_index_masks);

        auto const iter0 = first + index0;
        auto const iter1 = first + index1;
        auto const value0 = *iter0;
        auto const value1 = *iter1;
        using complex_type = typename std::iterator_traits<Iterator>::value_type;
        using real_type = ::ket::utility::meta::real_t<complex_type>;
        using boost::math::constants::one_div_root_two;
        *iter0 = (value0 + value1) * one_div_root_two<real_type>();
        *iter1 = (value0 - value1) * one_div_root_two<real_type>();

        auto const phase_coefficient0
          = phase_shift_table.coefficients[static_cast<std::size_t>(
              ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_index(
                phase_shift_table, fused_index0))];
        if (phase_coefficient0 != ::bra::complex_type{1.0, 0.0})
          *iter0 *= phase_coefficient0;
        auto const phase_coefficient1
          = phase_shift_table.coefficients[static_cast<std::size_t>(
              ::bra::fused_gate::apply_phase_shift_terms_detail::phase_shift_table_index(
                phase_shift_table, fused_index1))];
        if (phase_coefficient1 != ::bra::complex_type{1.0, 0.0})
          *iter1 *= phase_coefficient1;
      }
    } // namespace apply_hadamard_and_phase_shift_terms_detail

    template <typename Iterator, typename QubitsRange1, typename QubitsRange2>
    inline auto apply_hadamard_and_phase_shift_terms(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      QubitsRange1 const& unsorted_fused_qubits_or_masks,
      QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
      ::bra::bit_integer_type const target_qubit,
      std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms) -> void
    {
      auto const num_fused_qubits
        = static_cast< ::bra::bit_integer_type >(boost::size(unsorted_fused_qubits_or_masks));
      assert(target_qubit < num_fused_qubits);
      auto const is_fused_index_identity
        = ::bra::fused_gate::apply_phase_shift_terms_detail::is_fused_index_identity(
            fused_index_wo_qubits, unsorted_fused_qubits_or_masks);
      auto const target_qubit_mask
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(target_qubit);
      auto const lower_bits_mask = target_qubit_mask - ::bra::state_integer_type{1u};
      auto const upper_bits_mask = compl lower_bits_mask;
      auto const count
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(num_fused_qubits - ::bra::bit_integer_type{1u});
      auto const phase_shift_table
        = ::bra::fused_gate::apply_phase_shift_terms_detail::make_phase_shift_table(phase_shift_terms);
      assert(not phase_shift_table.coefficients.empty());
      for (auto index_wo_qubit = ::bra::state_integer_type{0u}; index_wo_qubit < count; ++index_wo_qubit)
        ::bra::fused_gate::apply_hadamard_and_phase_shift_terms_detail::apply_one(
          first, fused_index_wo_qubits,
          unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
          is_fused_index_identity,
          target_qubit_mask, lower_bits_mask, upper_bits_mask,
          phase_shift_table, index_wo_qubit);
    }

    template <typename ParallelPolicy, typename Iterator, typename QubitsRange1, typename QubitsRange2>
    inline auto apply_hadamard_and_phase_shift_terms(
      ParallelPolicy const parallel_policy, int const thread_index,
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      QubitsRange1 const& unsorted_fused_qubits_or_masks,
      QubitsRange2 const& sorted_fused_qubits_with_sentinel_or_index_masks,
      ::bra::bit_integer_type const target_qubit,
      std::vector< ::bra::fused_gate::phase_shift_term > const& phase_shift_terms) -> void
    {
      auto const num_fused_qubits
        = static_cast< ::bra::bit_integer_type >(boost::size(unsorted_fused_qubits_or_masks));
      assert(target_qubit < num_fused_qubits);
      auto const is_fused_index_identity
        = ::bra::fused_gate::apply_phase_shift_terms_detail::is_fused_index_identity(
            fused_index_wo_qubits, unsorted_fused_qubits_or_masks);
      auto const target_qubit_mask
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(target_qubit);
      auto const lower_bits_mask = target_qubit_mask - ::bra::state_integer_type{1u};
      auto const upper_bits_mask = compl lower_bits_mask;
      auto const count
        = ::ket::utility::integer_exp2< ::bra::state_integer_type >(num_fused_qubits - ::bra::bit_integer_type{1u});
      auto const phase_shift_table
        = ::bra::fused_gate::apply_phase_shift_terms_detail::make_phase_shift_table(phase_shift_terms);
      assert(not phase_shift_table.coefficients.empty());
      ::ket::utility::loop_n_in_execute(
        parallel_policy, count, thread_index,
        [first, fused_index_wo_qubits,
         &unsorted_fused_qubits_or_masks, &sorted_fused_qubits_with_sentinel_or_index_masks,
         is_fused_index_identity,
         target_qubit_mask, lower_bits_mask, upper_bits_mask,
         &phase_shift_table](::bra::state_integer_type const index_wo_qubit, int const)
        {
          ::bra::fused_gate::apply_hadamard_and_phase_shift_terms_detail::apply_one(
            first, fused_index_wo_qubits,
            unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
            is_fused_index_identity,
            target_qubit_mask, lower_bits_mask, upper_bits_mask,
            phase_shift_table, index_wo_qubit);
        });
    }
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_APPLY_HADAMARD_AND_PHASE_SHIFT_TERMS_HPP
