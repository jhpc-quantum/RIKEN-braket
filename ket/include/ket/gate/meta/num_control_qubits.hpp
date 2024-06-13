#ifndef KET_GATE_META_NUM_CONTROL_QUBITS_HPP
# define KET_GATE_META_NUM_CONTROL_QUBITS_HPP

# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/meta/bit_integer_of.hpp>


namespace ket
{
  namespace gate
  {
    namespace meta
    {
      namespace num_control_qubits_detail
      {
        template <typename... Qubits>
        struct all_control_qubits;

        template <>
        struct all_control_qubits<>
          : std::true_type
        { };

        template <typename StateInteger, typename BitInteger, typename... Qubits>
        struct all_control_qubits< ::ket::qubit<StateInteger, BitInteger>, Qubits... >
          : std::false_type
        { };

        template <typename StateInteger, typename BitInteger, typename... Qubits>
        struct all_control_qubits< ::ket::control< ::ket::qubit<StateInteger, BitInteger> >, Qubits... >
          : ::ket::gate::meta::num_control_qubits_detail::all_control_qubits<Qubits...>
        { };
      } // namespace num_control_qubits_detail

      template <typename BitInteger, typename... Qubits>
      struct num_control_qubits;

      template <typename BitInteger>
      struct num_control_qubits<BitInteger>
        : std::integral_constant<BitInteger, BitInteger{0u}>
      { };

      template <typename BitInteger, typename StateInteger, typename... Qubits>
      struct num_control_qubits<BitInteger, ::ket::qubit<StateInteger, BitInteger>, Qubits...>
        : ::ket::gate::meta::num_control_qubits<BitInteger, Qubits...>
      { };

      template <typename BitInteger, typename StateInteger, typename... Qubits>
      struct num_control_qubits<BitInteger, ::ket::control< ::ket::qubit<StateInteger, BitInteger> >, Qubits...>
        : std::integral_constant<BitInteger, static_cast<BitInteger>(sizeof...(Qubits) + 1u)>
      { static_assert(::ket::gate::meta::num_control_qubits_detail::all_control_qubits<Qubits...>::value, "Type sequence should be t,...t',c,...,c'"); };
    } // namespace meta
  } // namespace gate
} // namespace ket


#endif // KET_GATE_META_NUM_CONTROL_QUBITS_HPP

