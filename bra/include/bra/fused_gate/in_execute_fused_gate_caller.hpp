#ifndef BRA_FUSED_GATE_IN_EXECUTE_FUSED_GATE_CALLER_HPP
# define BRA_FUSED_GATE_IN_EXECUTE_FUSED_GATE_CALLER_HPP

# include <ket/utility/loop_n.hpp>
# include <ket/utility/parallel/loop_n.hpp>

# include <bra/types.hpp>


namespace bra
{
  namespace fused_gate
  {
    template <typename FusedGateCaller>
    class in_execute_fused_gate_caller
    {
      ::ket::utility::policy::parallel<unsigned int> parallel_policy_;
      FusedGateCaller const& fused_gate_caller_;

     public:
      in_execute_fused_gate_caller(
        ::ket::utility::policy::parallel<unsigned int> const parallel_policy,
        FusedGateCaller const& fused_gate_caller)
        : parallel_policy_{parallel_policy}, fused_gate_caller_{fused_gate_caller}
      { }

      template <typename First, typename UnsortedFusedQubitsOrMasks, typename SortedFusedQubitsWithSentinelOrIndexMasks>
      auto operator()(
        First const first, ::bra::state_integer_type const index_wo_qubits,
        UnsortedFusedQubitsOrMasks const& unsorted_fused_qubits_or_masks,
        SortedFusedQubitsWithSentinelOrIndexMasks const& sorted_fused_qubits_with_sentinel_or_index_masks,
        int const, ::bra::state_integer_type const unit_qubit_value) const -> void
      {
        ::ket::utility::execute(
          parallel_policy_,
          [this, first, index_wo_qubits,
           &unsorted_fused_qubits_or_masks, &sorted_fused_qubits_with_sentinel_or_index_masks,
           unit_qubit_value](int const thread_index, auto& executor)
          {
            fused_gate_caller_.call_in_execute(
              parallel_policy_, executor, thread_index,
              first, index_wo_qubits,
              unsorted_fused_qubits_or_masks, sorted_fused_qubits_with_sentinel_or_index_masks,
              unit_qubit_value);
          });
      }
    };
  } // namespace fused_gate
} // namespace bra


#endif // BRA_FUSED_GATE_IN_EXECUTE_FUSED_GATE_CALLER_HPP
