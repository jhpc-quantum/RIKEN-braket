#ifndef BRA_SIMPLE_MPI_STATE_HPP
# define BRA_SIMPLE_MPI_STATE_HPP

# ifndef BRA_NO_MPI
#   include <vector>

#   include <ket/gate/projective_measurement.hpp>
#   include <ket/utility/parallel/loop_n.hpp>
#   include <ket/mpi/utility/simple_mpi.hpp>

#   include <yampi/allocator.hpp>
#   include <yampi/rank.hpp>
#   include <yampi/communicator.hpp>
#   include <yampi/environment.hpp>

#   include <bra/state.hpp>


namespace bra
{
  class simple_mpi_state final
    : public ::bra::state
  {
    ket::utility::policy::parallel<unsigned int> parallel_policy_;
    ket::mpi::utility::policy::simple_mpi mpi_policy_;

    using data_type = std::vector<complex_type, yampi::allocator<complex_type>>;
    data_type data_;

   public:
    simple_mpi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      unsigned int const total_num_qubits,
      unsigned int const num_threads_per_process,
      ::bra::state::seed_type const seed,
#   ifdef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      unsigned int const num_elements_in_buffer,
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      yampi::communicator const& communicator,
      yampi::environment const& environment);

    simple_mpi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      std::vector<permutated_qubit_type> const& initial_permutation,
      unsigned int const num_threads_per_process,
      ::bra::state::seed_type const seed,
#   ifdef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      unsigned int const num_elements_in_buffer,
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
      yampi::communicator const& communicator,
      yampi::environment const& environment);

    ~simple_mpi_state() = default;
    simple_mpi_state(simple_mpi_state const&) = delete;
    simple_mpi_state& operator=(simple_mpi_state const&) = delete;
    simple_mpi_state(simple_mpi_state&&) = delete;
    simple_mpi_state& operator=(simple_mpi_state&&) = delete;

   private:
    data_type generate_initial_data(
      unsigned int const num_local_qubits,
      ::bra::state::state_integer_type const initial_integer,
      yampi::communicator const& communicator, yampi::environment const& environment) const;

    unsigned int do_num_page_qubits() const override;
    unsigned int do_num_pages() const override;

    void do_hadamard(qubit_type const qubit) override;
    void do_adj_hadamard(qubit_type const qubit) override;
    void do_not_(qubit_type const qubit) override;
    void do_adj_not_(qubit_type const qubit) override;
    void do_pauli_x(qubit_type const qubit) override;
    void do_adj_pauli_x(qubit_type const qubit) override;
    void do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_pauli_xx(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_pauli_xn(std::vector<qubit_type> const& qubits) override;
    void do_adj_pauli_xn(std::vector<qubit_type> const& qubits) override;
    void do_pauli_y(qubit_type const qubit) override;
    void do_adj_pauli_y(qubit_type const qubit) override;
    void do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_pauli_yy(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_pauli_yn(std::vector<qubit_type> const& qubits) override;
    void do_adj_pauli_yn(std::vector<qubit_type> const& qubits) override;
    void do_pauli_z(qubit_type const qubit) override;
    void do_adj_pauli_z(qubit_type const qubit) override;
    void do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_pauli_zn(std::vector<qubit_type> const& qubits) override;
    void do_adj_pauli_zn(std::vector<qubit_type> const& qubits) override;
    void do_swap(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_adj_swap(qubit_type const qubit1, qubit_type const qubit2) override;
    void do_u1(real_type const phase, qubit_type const qubit) override;
    void do_adj_u1(real_type const phase, qubit_type const qubit) override;
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
      complex_type const& phase_coefficient, qubit_type const qubit) override;
    void do_adj_phase_shift(
      complex_type const& phase_coefficient, qubit_type const qubit) override;
    void do_x_rotation_half_pi(qubit_type const qubit) override;
    void do_adj_x_rotation_half_pi(qubit_type const qubit) override;
    void do_y_rotation_half_pi(qubit_type const qubit) override;
    void do_adj_y_rotation_half_pi(qubit_type const qubit) override;
    void do_controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit,
      control_qubit_type const control_qubit) override;
    void do_adj_controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit,
      control_qubit_type const control_qubit) override;
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
    void do_adj_toffoli(
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
    void do_clear(qubit_type const qubit) override;
    void do_set(qubit_type const qubit) override;
    void do_controlled_hadamard(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_hadamard(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_hadamard(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_hadamard(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_not(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_not(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_pauli_xn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_pauli_xn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_pauli_yn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_pauli_yn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_pauli_z(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_pauli_z(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_multi_controlled_swap(
      qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_swap(
      qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) override;
    void do_controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit,
      control_qubit_type const control_qubit) override;
    void do_adj_controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit,
      control_qubit_type const control_qubit) override;
    void do_multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit,
      std::vector<control_qubit_type> const& control_qubit) override;
    void do_adj_multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit,
      std::vector<control_qubit_type> const& control_qubit) override;
    void do_controlled_u1(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_u1(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_multi_controlled_u1(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_u1(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
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
    void do_multi_controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) override;
    void do_adj_multi_controlled_v(
      complex_type const& phase_coefficient,
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
  }; // class simple_mpi_state
} // namespace bra


# endif // BRA_NO_MPI

#endif // BRA_SIMPLE_MPI_STATE_HPP
