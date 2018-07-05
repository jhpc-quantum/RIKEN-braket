#ifndef BRA_GENERAL_MPI_1PAGE_STATE_HPP
# define BRA_GENERAL_MPI_1PAGE_STATE_HPP

# include <boost/config.hpp>

# include <vector>

# include <ket/gate/projective_measurement.hpp>
# include <ket/utility/parallel/loop_n.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/state.hpp>

# include <yampi/allocator.hpp>
# include <yampi/rank.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <bra/state.hpp>

# ifdef BOOST_NO_CXX11_FINAL
#   define final 
#   define override 
# endif // BOOST_NO_CXX11_FINAL


namespace bra
{
  class general_mpi_1page_state final
    : public ::bra::state
  {
    ket::utility::policy::parallel<unsigned int> parallel_policy_;
    ket::mpi::utility::policy::general_mpi mpi_policy_;

    typedef
      ket::mpi::state<complex_type, 1, yampi::allocator<complex_type> >
      data_type;
    data_type data_;

   public:
    general_mpi_1page_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      unsigned int const total_num_qubits,
      ::bra::state::seed_type const seed,
      yampi::communicator const communicator,
      yampi::environment const& environment);

    general_mpi_1page_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      std::vector<qubit_type> const& initial_permutation,
      ::bra::state::seed_type const seed,
      yampi::communicator const communicator,
      yampi::environment const& environment);

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~general_mpi_1page_state() = default;
# else
    ~general_mpi_1page_state() { }
# endif

   private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    general_mpi_1page_state(general_mpi_1page_state const&) = delete;
    general_mpi_1page_state& operator=(general_mpi_1page_state const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    general_mpi_1page_state(general_mpi_1page_state&&) = delete;
    general_mpi_1page_state& operator=(general_mpi_1page_state&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
    general_mpi_1page_state(general_mpi_1page_state const&);
    general_mpi_1page_state& operator=(general_mpi_1page_state const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    general_mpi_1page_state(general_mpi_1page_state&&);
    general_mpi_1page_state& operator=(general_mpi_1page_state&&);
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

   private:
    unsigned int do_num_page_qubits() const override;
    unsigned int do_num_pages() const override;

    void do_hadamard(qubit_type const qubit) override;
    void do_adj_hadamard(qubit_type const qubit) override;
    void do_pauli_x(qubit_type const qubit) override;
    void do_adj_pauli_x(qubit_type const qubit) override;
    void do_pauli_y(qubit_type const qubit) override;
    void do_adj_pauli_y(qubit_type const qubit) override;
    void do_pauli_z(qubit_type const qubit) override;
    void do_adj_pauli_z(qubit_type const qubit) override;
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
      complex_type const phase_coefficient, qubit_type const qubit) override;
    void do_adj_phase_shift(
      complex_type const phase_coefficient, qubit_type const qubit) override;
    void do_x_rotation_half_pi(qubit_type const qubit) override;
    void do_adj_x_rotation_half_pi(qubit_type const qubit) override;
    void do_y_rotation_half_pi(qubit_type const qubit) override;
    void do_adj_y_rotation_half_pi(qubit_type const qubit) override;
    void do_controlled_not(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit) override;
    void do_adj_controlled_not(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit) override;
    void do_controlled_phase_shift(
      complex_type const phase_coefficient,
      qubit_type const target_qubit,
      control_qubit_type const control_qubit) override;
    void do_adj_controlled_phase_shift(
      complex_type const phase_coefficient,
      qubit_type const target_qubit,
      control_qubit_type const control_qubit) override;
    void do_controlled_v(
      complex_type const phase_coefficient,
      qubit_type const target_qubit,
      control_qubit_type const control_qubit) override;
    void do_adj_controlled_v(
      complex_type const phase_coefficient,
      qubit_type const target_qubit,
      control_qubit_type const control_qubit) override;
    void do_toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2) override;
    void do_adj_toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2) override;
    KET_GATE_OUTCOME_TYPE do_projective_measurement(
      qubit_type const qubit, yampi::rank const root) override;
    void do_expectation_values(yampi::rank const root) override;
    void do_measure(yampi::rank const root) override;
    void do_generate_events(yampi::rank const root, int const num_events, int const seed) override;
    void do_shor_box(
      bit_integer_type const num_exponent_qubits,
      state_integer_type const divisor, state_integer_type const base) override;
    void do_clear(qubit_type const qubit) override;
    void do_set(qubit_type const qubit) override;
    void do_depolarizing_channel(double const px, double const py, double const pz, int const seed) override;
  };
}


# ifdef BOOST_NO_CXX11_FINAL
#   undef final 
#   undef override 
# endif // BOOST_NO_CXX11_FINAL

#endif

