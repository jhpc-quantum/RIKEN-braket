#ifndef BRA_NOMPI_STATE_HPP
# define BRA_NOMPI_STATE_HPP

# ifdef BRA_NO_MPI
#   include <boost/config.hpp>

#   include <vector>

#   include <boost/move/unique_ptr.hpp>

#   include <ket/gate/projective_measurement.hpp>
#   include <ket/utility/integer_exp2.hpp>
#   include <ket/utility/parallel/loop_n.hpp>

#   include <bra/state.hpp>

#   ifdef BOOST_NO_CXX11_FINAL
#     define final 
#     define override 
#   endif // BOOST_NO_CXX11_FINAL


namespace bra
{
  class nompi_state final
    : public ::bra::state
  {
    ket::utility::policy::parallel<unsigned int> parallel_policy_;

    typedef std::vector<complex_type> data_type;
    data_type data_;

   public:
    nompi_state(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const total_num_qubits,
      ::bra::state::seed_type const seed);

   private:
    data_type make_initial_data(
      ::bra::state::state_integer_type const initial_integer,
      unsigned int const total_num_qubits)
    {
      data_type result(
        ket::utility::integer_exp2<state_integer_type>(total_num_qubits),
        static_cast<complex_type>(static_cast<real_type>(0)));
      result[initial_integer] = static_cast<complex_type>(static_cast<real_type>(1));
      return result;
    }

   public:
#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~nompi_state() = default;
#   else
    ~nompi_state() { }
#   endif

   private:
#   ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    nompi_state(nompi_state const&) = delete;
    nompi_state& operator=(nompi_state const&) = delete;
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    nompi_state(nompi_state&&) = delete;
    nompi_state& operator=(nompi_state&&) = delete;
#     endif // BOOST_NO_CXX11_RVALUE_REFERENCES
#   else // BOOST_NO_CXX11_DELETED_FUNCTIONS
    nompi_state(nompi_state const&);
    nompi_state& operator=(nompi_state const&);
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    nompi_state(nompi_state&&);
    nompi_state& operator=(nompi_state&&);
#     endif // BOOST_NO_CXX11_RVALUE_REFERENCES
#   endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

   private:
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
    KET_GATE_OUTCOME_TYPE do_projective_measurement(qubit_type const qubit) override;
    void do_expectation_values() override;
    void do_measure() override;
    void do_generate_events(int const num_events, int const seed) override;
    void do_shor_box(
      bit_integer_type const num_exponent_qubits,
      state_integer_type const divisor, state_integer_type const base) override;
    void do_clear(qubit_type const qubit) override;
    void do_set(qubit_type const qubit) override;
    void do_depolarizing_channel(double const px, double const py, double const pz, int const seed) override;
  };


  inline boost::movelib::unique_ptr< ::bra::state > make_nompi_state(
    ::bra::state::state_integer_type const initial_integer,
    ::bra::state::bit_integer_type const total_num_qubits,
    ::bra::state::seed_type const seed)
  {
    return boost::movelib::unique_ptr< ::bra::state >(
      new ::bra::nompi_state(initial_integer, total_num_qubits, seed));
  }
}


#   ifdef BOOST_NO_CXX11_FINAL
#     undef final 
#     undef override 
#   endif // BOOST_NO_CXX11_FINAL
# endif // BRA_NO_MPI

#endif

