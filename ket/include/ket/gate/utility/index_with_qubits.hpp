#ifndef KET_GATE_UTILITY_INDEX_WITH_QUBITS_HPP
# define KET_GATE_UTILITY_INDEX_WITH_QUBITS_HPP

# include <cassert>
# include <cstddef>
# include <array>

# include <ket/utility/integer_exp2.hpp>


namespace ket
{
  namespace gate
  {
    namespace utility
    {
      template <typename StateInteger, typename UnsignedInteger, typename RandomAccessIterator1, typename RandomAccessIterator2>
      inline constexpr auto index_with_qubits(
        StateInteger const index_wo_qubits, UnsignedInteger const qubits_value,
        RandomAccessIterator1 const qubit_masks_first, RandomAccessIterator1 const qubit_masks_last,
        RandomAccessIterator2 const index_masks_first, RandomAccessIterator2 const index_masks_last)
      -> StateInteger
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<UnsignedInteger>::value, "UnsignedInteger should be unsigned");
        assert(qubits_value < ::ket::utility::integer_exp2<UnsignedInteger>(qubit_masks_last - qubit_masks_first));

        // xx0xx0xx0xx
        auto result = StateInteger{0u};
        for (auto iter = index_masks_first; iter != index_masks_last; ++iter)
          result |= (index_wo_qubits bitand *iter) << (iter - index_masks_first);

        for (auto iter = qubit_masks_first; iter != qubit_masks_last; ++iter)
          if (((StateInteger{1u} << (iter - qubit_masks_first)) bitand static_cast<StateInteger>(qubits_value)) != StateInteger{0u})
            result |= *iter;

        return result;
      }

      template <typename StateInteger, typename UnsignedInteger, std::size_t num_operated_qubits>
      inline constexpr auto index_with_qubits(
        StateInteger const index_wo_qubits, UnsignedInteger const qubits_value,
        std::array<StateInteger, num_operated_qubits> const& qubit_masks,
        std::array<StateInteger, num_operated_qubits + 1u> const& index_masks)
      -> StateInteger
      {
        static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
        static_assert(std::is_unsigned<UnsignedInteger>::value, "UnsignedInteger should be unsigned");
        assert(qubits_value < ::ket::utility::integer_exp2<UnsignedInteger>(num_operated_qubits));

        // xx0xx0xx0xx
        auto result = StateInteger{0u};
        for (auto index_mask_index = std::size_t{0u}; index_mask_index < num_operated_qubits + std::size_t{1u}; ++index_mask_index)
          result |= (index_wo_qubits bitand index_masks[index_mask_index]) << index_mask_index;

        for (auto qubit_index = std::size_t{0u}; qubit_index < num_operated_qubits; ++qubit_index)
          if (((StateInteger{1u} << qubit_index) bitand static_cast<StateInteger>(qubits_value)) != StateInteger{0u})
            result |= qubit_masks[qubit_index];

        return result;
      }
    } // namespace utility
  } // namespace gate
} // namespace ket


#endif // KET_GATE_UTILITY_INDEX_WITH_QUBITS_HPP
