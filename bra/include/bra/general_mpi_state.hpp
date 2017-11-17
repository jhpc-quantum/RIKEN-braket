#ifndef BRA_GENERAL_MPI_STATE_HPP
# define BRA_GENERAL_MPI_STATE_HPP

# include <boost/config.hpp>

# include <vector>

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
  class general_mpi_state final
    : public ::bra::state
  {
    ket::utility::policy::parallel<unsigned int> parallel_policy_;
    ket::mpi::utility::policy::general_mpi mpi_policy_;

    typedef
      ket::mpi::state<complex_type, 0, yampi::allocator<complex_type> >
      data_type;
    data_type data_;

   public:
    general_mpi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      unsigned int const total_num_qubits,
      ::bra::state::seed_type const seed,
      yampi::communicator const communicator,
      yampi::environment const& environment);

    general_mpi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const num_local_qubits,
      std::vector<qubit_type> const& initial_permutation,
      ::bra::state::seed_type const seed,
      yampi::communicator const communicator,
      yampi::environment const& environment);

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~general_mpi_state() = default;
# else
    ~general_mpi_state() { }
# endif

   private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    general_mpi_state(general_mpi_state const&) = delete;
    general_mpi_state& operator=(general_mpi_state const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    general_mpi_state(general_mpi_state&&) = delete;
    general_mpi_state& operator=(general_mpi_state&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
    general_mpi_state(general_mpi_state const&);
    general_mpi_state& operator=(general_mpi_state const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    general_mpi_state(general_mpi_state&&);
    general_mpi_state& operator=(general_mpi_state&&);
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

   private:
    unsigned int do_num_pages() const override;

    void do_hadamard(qubit_type const qubit) override;
    void do_adj_hadamard(qubit_type const qubit) override;
    void do_phase_shift(complex_type const phase_coefficient, qubit_type const qubit) override;
    void do_adj_phase_shift(complex_type const phase_coefficient, qubit_type const qubit) override;
    void do_x_rotation_half_pi(qubit_type const qubit) override;
    void do_adj_x_rotation_half_pi(qubit_type const qubit) override;
    void do_y_rotation_half_pi(qubit_type const qubit) override;
    void do_adj_y_rotation_half_pi(qubit_type const qubit) override;
    void do_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_controlled_phase_shift(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_phase_shift(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_controlled_v(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    void do_adj_controlled_v(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit) override;
    /*
    void do_toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2) override;
    void do_adj_toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2) override;
      */
    void do_expectation_values(yampi::rank const root) override;
    void do_measure(yampi::rank const root) override;
  };
}


# ifdef BOOST_NO_CXX11_FINAL
#   undef final 
#   undef override 
# endif // BOOST_NO_CXX11_FINAL

#endif

