#include <vector>

#include <ket/gate/fused/global_phase.hpp>
#if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/gate/utility/cache_aware_iterator.hpp>
#endif

#include <bra/fused_gate/fused_global_phase.hpp>
#include <bra/types.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename Iterator>
    fused_global_phase<Iterator>::fused_global_phase(::bra::real_type const phase)
      : ::bra::fused_gate::fused_gate<Iterator>{}, phase_{phase}
    { }

#ifndef KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_global_phase<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
      std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const&) const -> void
    {
      ::ket::gate::fused::runtime::ranges::global_phase(
        first, fused_index_wo_qubits,
        unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, phase_);
    }

    template <typename Iterator>
    auto fused_global_phase<Iterator>::do_call_in_execute(
      ::ket::utility::policy::parallel<unsigned int> const parallel_policy, int const thread_index,
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::qubit_type > const& unsorted_fused_qubits,
      std::vector< ::bra::qubit_type > const& sorted_fused_qubits_with_sentinel,
      std::vector< ::bra::bit_integer_type > const&,
      ::bra::state_integer_type const) const -> void
    {
      ::ket::gate::fused::runtime::ranges::global_phase(
        parallel_policy, thread_index, first, fused_index_wo_qubits,
        unsorted_fused_qubits, sorted_fused_qubits_with_sentinel, phase_);
    }
#else // KET_USE_BIT_MASKS_EXPLICITLY
    template <typename Iterator>
    auto fused_global_phase<Iterator>::do_call(
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::state_integer_type > const& qubit_masks,
      std::vector< ::bra::state_integer_type > const& index_masks,
      std::vector< ::bra::bit_integer_type > const&) const -> void
    {
      ::ket::gate::fused::runtime::ranges::global_phase(
        first, fused_index_wo_qubits, qubit_masks, index_masks, phase_);
    }

    template <typename Iterator>
    auto fused_global_phase<Iterator>::do_call_in_execute(
      ::ket::utility::policy::parallel<unsigned int> const parallel_policy, int const thread_index,
      Iterator const first, ::bra::state_integer_type const fused_index_wo_qubits,
      std::vector< ::bra::state_integer_type > const& qubit_masks,
      std::vector< ::bra::state_integer_type > const& index_masks,
      std::vector< ::bra::bit_integer_type > const&,
      ::bra::state_integer_type const) const -> void
    {
      ::ket::gate::fused::runtime::ranges::global_phase(
        parallel_policy, thread_index, first, fused_index_wo_qubits,
        qubit_masks, index_masks, phase_);
    }
#endif // KET_USE_BIT_MASKS_EXPLICITLY

    template class fused_global_phase< ::bra::data_type::iterator >;
#if !defined(BRA_NO_MPI) && (!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    template class fused_global_phase< ::bra::paged_data_type::iterator >;
#endif
#ifndef KET_USE_BIT_MASKS_EXPLICITLY
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_global_phase<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::qubit_type >>;
#   ifndef BRA_NO_MPI
    template class fused_global_phase<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::qubit_type >>;
#   endif
# endif
#else
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    template class fused_global_phase<ket::gate::utility::cache_aware_iterator< ::bra::data_type::iterator, ::bra::state_integer_type >>;
#   ifndef BRA_NO_MPI
    template class fused_global_phase<ket::gate::utility::cache_aware_iterator< ::bra::paged_data_type::iterator, ::bra::state_integer_type >>;
#   endif
# endif
#endif
  } // namespace fused_gate
} // namespace bra
