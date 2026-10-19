#ifndef KET_MPI_GATE_DETAIL_DISPATCH_CONTROL_VALUES_HPP
# define KET_MPI_GATE_DETAIL_DISPATCH_CONTROL_VALUES_HPP

# include <array>
# include <cstddef>
# include <type_traits>
# include <boost/range/join.hpp>
# include <boost/range/adaptor/transformed.hpp>
# include <ket/gate/meta/num_control_qubits.hpp>
# include <ket/gate/utility/control_mask.hpp>
# include <ket/runtime_control.hpp>
# include <ket/utility/meta/ranges.hpp>

namespace ket { namespace mpi { namespace gate { namespace detail
{
  // Keep the existing optimized positive-control paths. Other control types use
  // the value-aware runtime kernels, including homogeneous compile-time zero
  // controls and heterogeneous variadic controls in their original order.
  template <typename... Qubits>
  using all_default_control_operands = ::ket::gate::utility::all_default_control_operands<Qubits...>;

  template <typename Tag, typename LocalState>
  struct value_control_dispatch;

  template <typename Function, typename Controls>
  inline auto try_positive_control_range(std::true_type, Function&& function, Controls const& controls) -> bool
  { return function(controls); }

  template <typename Function, typename Controls>
  inline auto try_positive_control_range(std::false_type, Function&&, Controls const&) -> bool
  { return false; }

  // The diagonal fast paths currently filter only all-one controls. Declining
  // them here lets zero/mixed controls use the value-aware local/page kernels.
  template <typename Function, typename Controls>
  inline auto try_positive_control_range(Function&& function, Controls const& controls) -> bool
  {
    return try_positive_control_range(
      ::ket::gate::utility::is_default_control_operand< ::ket::utility::meta::range_value_t<Controls> >{},
      function, controls);
  }

  struct hadamard_tag;
  struct adj_hadamard_tag;
  struct swap_tag;
  struct adj_swap_tag;
  struct exponential_pauli_z_coeff_tag;
  struct adj_exponential_pauli_z_coeff_tag;
  struct exponential_pauli_z_tag;
  struct adj_exponential_pauli_z_tag;
  struct sqrt_pauli_z_tag;
  struct adj_sqrt_pauli_z_tag;
  struct phase_shift_coeff_tag;
  struct adj_phase_shift_coeff_tag;
  struct phase_shift_tag;
  struct adj_phase_shift_tag;
  struct controlled_v_coeff_tag;
  struct adj_controlled_v_coeff_tag;
  struct controlled_v_tag;
  struct adj_controlled_v_tag;
  struct pauli_z_tag;
  struct adj_pauli_z_tag;
  struct not_gate_tag;
  struct adj_not_gate_tag;
  struct phase_shift2_tag;
  struct adj_phase_shift2_tag;
  struct phase_shift3_tag;
  struct adj_phase_shift3_tag;
  struct y_rotation_half_pi_tag;
  struct adj_y_rotation_half_pi_tag;
  struct x_rotation_half_pi_tag;
  struct adj_x_rotation_half_pi_tag;
  struct sqrt_pauli_y_tag;
  struct adj_sqrt_pauli_y_tag;
  struct identity_tag;
  struct adj_identity_tag;
  struct sqrt_pauli_x_tag;
  struct adj_sqrt_pauli_x_tag;
  struct pauli_y_tag;
  struct adj_pauli_y_tag;
  struct exponential_swap_coeff_tag;
  struct adj_exponential_swap_coeff_tag;
  struct exponential_swap_tag;
  struct adj_exponential_swap_tag;
  struct pauli_x_tag;
  struct adj_pauli_x_tag;
  struct exponential_pauli_y_coeff_tag;
  struct adj_exponential_pauli_y_coeff_tag;
  struct exponential_pauli_y_tag;
  struct adj_exponential_pauli_y_tag;
  struct exponential_pauli_x_coeff_tag;
  struct adj_exponential_pauli_x_coeff_tag;
  struct exponential_pauli_x_tag;
  struct adj_exponential_pauli_x_tag;

  template <typename Targets, typename Controls, typename Qubit>
  inline auto append_control_operand(
    Targets& targets, Controls&, std::size_t& target_index, std::size_t&,
    Qubit const qubit, std::false_type) -> void
  { targets[target_index++] = qubit; }

  template <typename Targets, typename Controls, typename Control>
  inline auto append_control_operand(
    Targets&, Controls& controls, std::size_t&, std::size_t& control_index,
    Control const control, std::true_type) -> void
  { controls[control_index++] = ::ket::make_runtime_control(::ket::remove_control(control), ::ket::control_value(control)); }

  template <typename Qubit, typename Legacy, typename ValueAware, typename... Qubits>
  inline auto dispatch_control_values_impl(
    std::true_type, Legacy&& legacy, ValueAware&&, Qubits const... qubits)
  -> decltype(legacy(qubits...))
  { return legacy(qubits...); }

  template <typename Qubit, typename Legacy, typename ValueAware, typename... Qubits>
  inline auto dispatch_control_values_impl(
    std::false_type, Legacy&& legacy, ValueAware&& value_aware, Qubits const... qubits)
  -> decltype(legacy(qubits...))
  {
    // num_control_qubits also enforces the targets-before-controls contract.
    constexpr auto num_controls = ::ket::gate::meta::num_control_qubits<typename Qubit::bit_integer_type, Qubits...>::value;
    std::array<Qubit, sizeof...(Qubits) - num_controls> targets{};
    std::array< ::ket::runtime_control<Qubit>, num_controls> controls{};
    auto target_index = std::size_t{0u};
    auto control_index = std::size_t{0u};
    using swallow = int[];
    (void)swallow{0, (append_control_operand(
      targets, controls, target_index, control_index,
      qubits, ::ket::meta::is_control<Qubits>{}), 0)...};
    return value_aware(targets, controls);
  }

  template <typename Qubit, typename Legacy, typename ValueAware, typename... Qubits>
  inline auto dispatch_control_values(Legacy&& legacy, ValueAware&& value_aware, Qubits const... qubits)
  -> decltype(legacy(qubits...))
  {
    return dispatch_control_values_impl<Qubit>(
      all_default_control_operands<Qubits...>{}, legacy, value_aware, qubits...);
  }
} } } }

#endif // KET_MPI_GATE_DETAIL_DISPATCH_CONTROL_VALUES_HPP
