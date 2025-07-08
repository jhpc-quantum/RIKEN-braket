#ifndef BRA_UNIT_MPI_STATE_HPP
# define BRA_UNIT_MPI_STATE_HPP

# ifndef BRA_NO_MPI
#   include <vector>
#   include <memory>

#   include <ket/gate/projective_measurement.hpp>
#   if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#     include <ket/gate/utility/cache_aware_iterator.hpp>
#   endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   include <ket/utility/parallel/loop_n.hpp>
#   include <ket/mpi/utility/unit_mpi.hpp>

#   include <yampi/rank.hpp>
#   include <yampi/communicator.hpp>
#   include <yampi/environment.hpp>

#   include <bra/types.hpp>
#   include <bra/state.hpp>
#   include <bra/fused_gate/fused_gate.hpp>


namespace bra
{
  class unit_mpi_state final
    : public ::bra::state
  {
    ket::utility::policy::parallel<unsigned int> parallel_policy_;
    using unit_mpi_policy_type
      = ket::mpi::utility::policy::unit_mpi< ::bra::state::state_integer_type, ::bra::state::bit_integer_type, unsigned int >;
    unit_mpi_policy_type mpi_policy_;

    using data_type = ::bra::data_type;
    data_type data_;

    using fused_gate_iterator = data_type::iterator;
    std::vector<std::unique_ptr< ::bra::fused_gate::fused_gate<fused_gate_iterator> >> fused_gates_; // related to begin_fusion/end_fusion
#   if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#     ifndef KET_USE_BIT_MASKS_EXPLICITLY
    using cache_aware_fused_gate_iterator = ket::gate::utility::cache_aware_iterator<fused_gate_iterator, ::bra::qubit_type>;
#     else // KET_USE_BIT_MASKS_EXPLICITLY
    using cache_aware_fused_gate_iterator = ket::gate::utility::cache_aware_iterator<fused_gate_iterator, ::bra::state_integer_type>;
#     endif // KET_USE_BIT_MASKS_EXPLICITLY
    std::vector<std::unique_ptr< ::bra::fused_gate::fused_gate<cache_aware_fused_gate_iterator> >> cache_aware_fused_gates_; // related to begin_fusion/end_fusion
#   endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

   public:
    unit_mpi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      unsigned int const num_unit_qubits,
      unsigned int const total_num_qubits,
      unsigned int const num_threads_per_process,
      unsigned int const num_processes_per_unit,
      ::bra::state::seed_type const seed,
#   ifdef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      unsigned int const num_elements_in_buffer,
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      yampi::communicator const& communicator,
      yampi::environment const& environment);

    unit_mpi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      unsigned int const num_unit_qubits,
      std::vector<permutated_qubit_type> const& initial_permutation,
      unsigned int const num_threads_per_process,
      unsigned int const num_processes_per_unit,
      ::bra::state::seed_type const seed,
#   ifdef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      unsigned int const num_elements_in_buffer,
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      yampi::communicator const& communicator,
      yampi::environment const& environment);

    ~unit_mpi_state() = default;
    unit_mpi_state(unit_mpi_state const&) = delete;
    unit_mpi_state& operator=(unit_mpi_state const&) = delete;
    unit_mpi_state(unit_mpi_state&&) = delete;
    unit_mpi_state& operator=(unit_mpi_state&&) = delete;

   private:
    data_type generate_initial_data(
      unsigned int const num_local_qubits,
      ::bra::state::state_integer_type const initial_integer,
      yampi::communicator const& communicator, yampi::environment const& environment) const;

    unsigned int do_num_page_qubits() const override;
    unsigned int do_num_pages() const override;

    void do_i_gate(qubit_type const qubit) override;
    void do_ic_gate(control_qubit_type const control_qubit) override;
    void do_ii_gate(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_in_gate(std::vector<qubit_type> const& qubits) override;
    void do_hadamard(qubit_type const qubit) override;
    void do_not_(qubit_type const qubit) override;
    void do_pauli_x(qubit_type const qubit) override;
    void do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_pauli_xn(std::vector<qubit_type> const& qubits) override;
    void do_pauli_y(qubit_type const qubit) override;
    void do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_pauli_yn(std::vector<qubit_type> const& qubits) override;
    void do_pauli_z(control_qubit_type const control_qubit) override;
    void do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_pauli_zn(std::vector<qubit_type> const& qubits) override;
    void do_swap(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_sqrt_pauli_x(qubit_type const qubit) override;
    void do_adj_sqrt_pauli_x(qubit_type const qubit) override;
    void do_sqrt_pauli_y(qubit_type const qubit) override;
    void do_adj_sqrt_pauli_y(qubit_type const qubit) override;
    void do_sqrt_pauli_z(control_qubit_type const control_qubit) override;
    void do_adj_sqrt_pauli_z(control_qubit_type const control_qubit) override;
    void do_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_sqrt_pauli_zn(std::vector<qubit_type> const& qubits) override;
    void do_adj_sqrt_pauli_zn(std::vector<qubit_type> const& qubits) override;
    void do_u1(real_type const phase, control_qubit_type const control_qubit) override;
    void do_adj_u1(real_type const phase, control_qubit_type const control_qubit) override;
    void do_u2(
      real_type const phase1, real_type const phase2,
      qubit_type const qubit) override;
    void do_adj_u2(
      real_type const phase1, real_type const phase2,
      qubit_type const qubit) override;
    void do_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const qubit) override;
    void do_adj_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const qubit) override;
    void do_phase_shift(
      complex_type const& phase_coefficient, control_qubit_type const control_qubit) override;
    void do_adj_phase_shift(
      complex_type const& phase_coefficient, control_qubit_type const control_qubit) override;
    void do_x_rotation_half_pi(qubit_type const qubit) override;
    void do_adj_x_rotation_half_pi(qubit_type const qubit) override;
    void do_y_rotation_half_pi(qubit_type const qubit) override;
    void do_adj_y_rotation_half_pi(qubit_type const qubit) override;
    void do_exponential_pauli_x(real_type const phase, qubit_type const qubit) override;
    void do_adj_exponential_pauli_x(real_type const phase, qubit_type const qubit) override;
    void do_exponential_pauli_xx(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_exponential_pauli_xx(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_adj_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_exponential_pauli_y(real_type const phase, qubit_type const qubit) override;
    void do_adj_exponential_pauli_y(real_type const phase, qubit_type const qubit) override;
    void do_exponential_pauli_yy(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_exponential_pauli_yy(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_adj_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_exponential_pauli_z(real_type const phase, qubit_type const qubit) override;
    void do_adj_exponential_pauli_z(real_type const phase, qubit_type const qubit) override;
    void do_exponential_pauli_zz(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_exponential_pauli_zz(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_adj_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& qubits) override;
    void do_exponential_swap(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_exponential_swap(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) override;
    void do_toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2) override;

    ket::gate::outcome do_projective_measurement(
      qubit_type const qubit, yampi::rank const root) override;
    void do_expectation_values(yampi::rank const root) override;
    void do_measure(yampi::rank const root) override;
    void do_generate_events(yampi::rank const root, int const num_events, int const seed) override;
    void do_shor_box(
      state_integer_type const divisor, state_integer_type const base,
      std::vector<qubit_type> const& exponent_qubits,
      std::vector<qubit_type> const& modular_exponentiation_qubits) override;
    void do_begin_fusion() override;
    void do_end_fusion() override;
    void do_clear(qubit_type const qubit) override;
    void do_set(qubit_type const qubit) override;

    void do_controlled_i_gate(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_controlled_ic_gate(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_multi_controlled_in_gate(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_ic_gate(
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_hadamard(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_hadamard(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_not(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_pauli_xn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_pauli_yn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_pauli_z(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_multi_controlled_pauli_z(std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_swap(
      qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_sqrt_pauli_z(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_adj_controlled_sqrt_pauli_z(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_sqrt_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_sqrt_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_phase_shift(
      complex_type const& phase_coefficient,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_adj_controlled_phase_shift(
      complex_type const& phase_coefficient,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      std::vector<control_qubit_type> const& control_qubit) override;
    void do_adj_multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      std::vector<control_qubit_type> const& control_qubit) override;
    void do_controlled_u1(
      real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_adj_controlled_u1(
      real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) override;
    void do_multi_controlled_u1(
      real_type const phase, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_u1(
      real_type const phase, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_exponential_pauli_x(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_exponential_pauli_x(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_exponential_pauli_y(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_exponential_pauli_y(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_exponential_swap(
      real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_exponential_swap(
      real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) override;
  }; // class unit_mpi_state
} // namespace bra


# endif // BRA_NO_MPI

#endif // BRA_UNIT_MPI_STATE_HPP
