#ifndef BRA_NO_MPI
# include <vector>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/mpi/gate/hadamard.hpp>
# include <ket/mpi/gate/not_.hpp>
# include <ket/mpi/gate/pauli_x.hpp>
# include <ket/mpi/gate/pauli_y.hpp>
# include <ket/mpi/gate/pauli_z.hpp>
# include <ket/mpi/gate/swap.hpp>
# include <ket/mpi/gate/phase_shift.hpp>
# include <ket/mpi/gate/x_rotation_half_pi.hpp>
# include <ket/mpi/gate/y_rotation_half_pi.hpp>
# include <ket/mpi/gate/controlled_v.hpp>
# include <ket/mpi/gate/exponential_pauli_x.hpp>
# include <ket/mpi/gate/exponential_pauli_y.hpp>
# include <ket/mpi/gate/exponential_pauli_z.hpp>
# include <ket/mpi/gate/exponential_swap.hpp>
# include <ket/mpi/gate/toffoli.hpp>
# include <ket/mpi/gate/projective_measurement.hpp>
# include <ket/mpi/gate/clear.hpp>
# include <ket/mpi/gate/set.hpp>
# include <ket/mpi/all_spin_expectation_values.hpp>
# include <ket/mpi/measure.hpp>
# include <ket/mpi/generate_events.hpp>
# include <ket/mpi/shor_box.hpp>

# include <bra/paged_simple_mpi_state.hpp>
# include <bra/state.hpp>


namespace bra
{
  unsigned int paged_simple_mpi_state::do_num_page_qubits() const
  { return data_.num_page_qubits(); }

  unsigned int paged_simple_mpi_state::do_num_pages() const
  { return data_.num_pages(); }

  paged_simple_mpi_state::paged_simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int const num_page_qubits,
    unsigned int const num_threads_per_process,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{
        mpi_policy_, num_local_qubits, num_page_qubits, initial_integer,
        permutation_, communicator, environment}
  { }

  paged_simple_mpi_state::paged_simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int const num_page_qubits,
    unsigned int const num_threads_per_process,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{
        mpi_policy_, num_local_qubits, num_page_qubits, initial_integer,
        permutation_, communicator, environment}
  { }

  void paged_simple_mpi_state::do_hadamard(qubit_type const qubit)
  {
    ket::mpi::gate::hadamard(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_adj_hadamard(qubit_type const qubit)
  {
    ket::mpi::gate::adj_hadamard(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_not_(qubit_type const qubit)
  {
    ket::mpi::gate::not_(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_adj_not_(qubit_type const qubit)
  {
    ket::mpi::gate::adj_not_(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_pauli_x(qubit_type const qubit)
  {
    ket::mpi::gate::pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_adj_pauli_x(qubit_type const qubit)
  {
    ket::mpi::gate::adj_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_adj_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_pauli_xn(std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_pauli_xn(std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::adj_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::adj_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_pauli_y(qubit_type const qubit)
  {
    ket::mpi::gate::pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_adj_pauli_y(qubit_type const qubit)
  {
    ket::mpi::gate::adj_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_adj_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_pauli_yn(std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_pauli_yn(std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::adj_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::adj_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_pauli_z(qubit_type const qubit)
  {
    ket::mpi::gate::pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_adj_pauli_z(qubit_type const qubit)
  {
    ket::mpi::gate::adj_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_adj_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::adj_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::adj_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_swap(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::swap(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_adj_swap(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_swap(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_u1(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_simple_mpi_state::do_adj_u1(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_simple_mpi_state::do_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, qubit);
  }

  void paged_simple_mpi_state::do_adj_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, qubit);
  }

  void paged_simple_mpi_state::do_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, qubit);
  }

  void paged_simple_mpi_state::do_adj_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, qubit);
  }

  void paged_simple_mpi_state::do_phase_shift(
    complex_type const& phase_coefficient, qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, qubit);
  }

  void paged_simple_mpi_state::do_adj_phase_shift(
    complex_type const& phase_coefficient, qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, qubit);
  }

  void paged_simple_mpi_state::do_x_rotation_half_pi(qubit_type const qubit)
  {
    ket::mpi::gate::x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_adj_x_rotation_half_pi(qubit_type const qubit)
  {
    ket::mpi::gate::adj_x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_y_rotation_half_pi(qubit_type const qubit)
  {
    ket::mpi::gate::y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_adj_y_rotation_half_pi(qubit_type const qubit)
  {
    ket::mpi::gate::adj_y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::controlled_v_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_controlled_v_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_simple_mpi_state::do_adj_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_simple_mpi_state::do_exponential_pauli_xx(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_adj_exponential_pauli_xx(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_simple_mpi_state::do_adj_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_simple_mpi_state::do_exponential_pauli_yy(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_adj_exponential_pauli_yy(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_simple_mpi_state::do_adj_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_simple_mpi_state::do_exponential_pauli_zz(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_adj_exponential_pauli_zz(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_qubits = qubits.size();
    assert(num_qubits > 2u);

    switch (num_qubits)
    {
     case 3u:
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u]);
      break;

     case 6u:
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubits[0u], qubits[1u], qubits[2u], qubits[3u], qubits[4u], qubits[5u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_exponential_swap(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::exponential_swap(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_adj_exponential_swap(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_exponential_swap(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_simple_mpi_state::do_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    ket::mpi::gate::toffoli(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit1, control_qubit2);
  }

  void paged_simple_mpi_state::do_adj_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    ket::mpi::gate::adj_toffoli(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit1, control_qubit2);
  }

  ::ket::gate::outcome paged_simple_mpi_state::do_projective_measurement(
    qubit_type const qubit, yampi::rank const root)
  {
    return ket::mpi::gate::projective_measurement(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, root, communicator_, environment_, random_number_generator_, qubit);
  }

  void paged_simple_mpi_state::do_expectation_values(yampi::rank const root)
  {
    maybe_expectation_values_
      = ket::mpi::all_spin_expectation_values<spins_allocator_type>(
          mpi_policy_, parallel_policy_,
          data_, permutation_, total_num_qubits_, buffer_, root, communicator_, environment_);
  }

  void paged_simple_mpi_state::do_measure(yampi::rank const root)
  {
    measured_value_
      = ket::mpi::measure(
          mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
          data_, random_number_generator_, permutation_, communicator_, environment_);
  }

  void paged_simple_mpi_state::do_generate_events(yampi::rank const root, int const num_events, int const seed)
  {
    if (seed < 0)
      ket::mpi::generate_events(
        mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
        generated_events_, data_, num_events, random_number_generator_, permutation_,
        communicator_, environment_);
    else
      ket::mpi::generate_events(
        mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
        generated_events_, data_, num_events, random_number_generator_, static_cast<seed_type>(seed), permutation_,
        communicator_, environment_);
  }

  void paged_simple_mpi_state::do_shor_box(
    state_integer_type const divisor, state_integer_type const base,
    std::vector<qubit_type> const& exponent_qubits,
    std::vector<qubit_type> const& modular_exponentiation_qubits)
  {
    ket::mpi::shor_box(
      mpi_policy_, parallel_policy_,
      data_, base, divisor, exponent_qubits, modular_exponentiation_qubits,
      permutation_, communicator_, environment_);
  }

  void paged_simple_mpi_state::do_clear(qubit_type const qubit)
  {
    ket::mpi::gate::clear(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_set(qubit_type const qubit)
  {
    ket::mpi::gate::set(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_simple_mpi_state::do_controlled_hadamard(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::hadamard(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_hadamard(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_hadamard(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_hadamard(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + std::size_t{1u}};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_hadamard(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::adj_hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::adj_hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + std::size_t{1u}};
    }
  }

  void paged_simple_mpi_state::do_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::not_(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_not_(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_not(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + std::size_t{1u}};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_not(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::adj_not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::adj_not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + std::size_t{1u}};
    }
  }

  void paged_simple_mpi_state::do_controlled_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_pauli_xn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_pauli_xn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_controlled_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_pauli_yn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_pauli_yn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_controlled_pauli_z(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_pauli_z(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_multi_controlled_swap(
    qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    switch (num_control_qubits)
    {
     case 1u:
      ket::mpi::gate::swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit1, target_qubit2, control_qubits[0u]);
      break;

     case 2u:
      ket::mpi::gate::swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 2u};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_swap(
    qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    switch (num_control_qubits)
    {
     case 1u:
      ket::mpi::gate::adj_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit1, target_qubit2, control_qubits[0u]);
      break;

     case 2u:
      ket::mpi::gate::adj_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::adj_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 2u};
    }
  }

  void paged_simple_mpi_state::do_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 1u};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::adj_phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::adj_phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 1u};
    }
  }

  void paged_simple_mpi_state::do_controlled_u1(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::phase_shift(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_u1(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_phase_shift(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_u1(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 1u};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_u1(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::adj_phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::adj_phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 1u};
    }
  }

  void paged_simple_mpi_state::do_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 1u};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::adj_phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::adj_phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 1u};
    }
  }

  void paged_simple_mpi_state::do_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 1u};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::adj_phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::adj_phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 1u};
    }
  }

  void paged_simple_mpi_state::do_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + std::size_t{1u}};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::adj_x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::adj_x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + std::size_t{1u}};
    }
  }

  void paged_simple_mpi_state::do_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + std::size_t{1u}};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::adj_y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::adj_y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + std::size_t{1u}};
    }
  }

  void paged_simple_mpi_state::do_multi_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::controlled_v_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::controlled_v_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::controlled_v_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::controlled_v_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 1u};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    switch (num_control_qubits)
    {
     case 2u:
      ket::mpi::gate::adj_controlled_v_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::adj_controlled_v_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_controlled_v_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     case 5u:
      ket::mpi::gate::adj_controlled_v_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 1u};
    }
  }

  void paged_simple_mpi_state::do_controlled_exponential_pauli_x(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_exponential_pauli_x(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_x(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_controlled_exponential_pauli_y(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_exponential_pauli_y(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_y(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_adj_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_simple_mpi_state::do_multi_controlled_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 1u);
    assert(num_qubits > 2u);

    switch (num_target_qubits)
    {
     case 1u:
      switch (num_control_qubits)
      {
       case 2u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       case 5u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u], control_qubits[4u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 2u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       case 4u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 3u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u]);
        break;

       case 3u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], control_qubits[0u], control_qubits[1u], control_qubits[2u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 4u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u]);
        break;

       case 2u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], control_qubits[0u], control_qubits[1u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     case 5u:
      switch (num_control_qubits)
      {
       case 1u:
        ket::mpi::gate::adj_exponential_pauli_z(
          mpi_policy_, parallel_policy_,
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubits[0u], target_qubits[1u], target_qubits[2u], target_qubits[3u], target_qubits[4u], control_qubits[0u]);
        break;

       default:
        throw bra::too_many_qubits_error{num_qubits};
      }
      break;

     default:
      throw bra::too_many_qubits_error{num_qubits};
    }
  }

  void paged_simple_mpi_state::do_multi_controlled_exponential_swap(
    real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    switch (num_control_qubits)
    {
     case 1u:
      ket::mpi::gate::exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2, control_qubits[0u]);
      break;

     case 2u:
      ket::mpi::gate::exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 2u};
    }
  }

  void paged_simple_mpi_state::do_adj_multi_controlled_exponential_swap(
    real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    switch (num_control_qubits)
    {
     case 1u:
      ket::mpi::gate::adj_exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2, control_qubits[0u]);
      break;

     case 2u:
      ket::mpi::gate::adj_exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u]);
      break;

     case 3u:
      ket::mpi::gate::adj_exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u], control_qubits[2u]);
      break;

     case 4u:
      ket::mpi::gate::adj_exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2, control_qubits[0u], control_qubits[1u], control_qubits[2u], control_qubits[3u]);
      break;

     default:
      throw bra::too_many_qubits_error{num_control_qubits + 2u};
    }
  }
} // namespace bra


#endif // BRA_NO_MPI
