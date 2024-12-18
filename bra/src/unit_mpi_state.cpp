#ifndef BRA_NO_MPI
# include <vector>

# include <boost/preprocessor/arithmetic/dec.hpp>
# include <boost/preprocessor/arithmetic/inc.hpp>
# include <boost/preprocessor/punctuation/comma_if.hpp>
# include <boost/preprocessor/comparison/equal.hpp>
# include <boost/preprocessor/control/iif.hpp>
# include <boost/preprocessor/repetition/repeat.hpp>
# include <boost/preprocessor/repetition/repeat_from_to.hpp>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/mpi/utility/unit_mpi.hpp>
# include <ket/mpi/gate/gate.hpp>
# include <ket/mpi/gate/identity.hpp>
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
# include <ket/mpi/utility/simple_mpi.hpp>

# include <bra/unit_mpi_state.hpp>
# include <bra/state.hpp>

# ifndef BRA_MAX_NUM_OPERATED_QUBITS
#   define BRA_MAX_NUM_OPERATED_QUBITS 6
# endif // BRA_MAX_NUM_OPERATED_QUBITS


namespace bra
{
  unsigned int unit_mpi_state::do_num_page_qubits() const
  { return 0u; }

  unsigned int unit_mpi_state::do_num_pages() const
  { return 1u; }

# ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  unit_mpi_state::unit_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const num_unit_qubits,
    unsigned int const total_num_qubits,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{num_unit_qubits, num_processes_per_unit},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)}
  { }

  unit_mpi_state::unit_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const num_unit_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{num_unit_qubits, num_processes_per_unit},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)}
  { }
# else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  unit_mpi_state::unit_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const num_unit_qubits,
    unsigned int const total_num_qubits,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, num_elements_in_buffer, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{num_unit_qubits, num_processes_per_unit},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)}
  { }

  unit_mpi_state::unit_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const num_unit_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, num_elements_in_buffer, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{num_unit_qubits, num_processes_per_unit},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)}
  { }
# endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS

  unit_mpi_state::data_type unit_mpi_state::generate_initial_data(
    unsigned int const num_local_qubits,
    ::bra::state::state_integer_type const initial_integer,
    yampi::communicator const& communicator, yampi::environment const& environment) const
  {
    auto result
      = data_type(
          ket::utility::integer_exp2<std::size_t>(num_local_qubits)
            * ket::mpi::utility::policy::num_data_blocks(mpi_policy_, communicator, environment),
          complex_type{0});

    auto const rank_index
      = ket::mpi::utility::qubit_value_to_rank_index(
          mpi_policy_, data_, ket::mpi::permutate_bits(permutation_, initial_integer),
          communicator, environment);
    if (communicator.rank(environment) == rank_index.first)
      result[rank_index.second] = complex_type{1};

    return result;
  }

  void unit_mpi_state::do_i_gate(qubit_type const qubit)
  {
    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_adj_i_gate(qubit_type const qubit)
  {
    ket::mpi::gate::adj_identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_ii_gate(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void unit_mpi_state::do_adj_ii_gate(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void unit_mpi_state::do_in_gate(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::identity(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_adj_in_gate(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_identity(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_hadamard(qubit_type const qubit)
  {
    ket::mpi::gate::hadamard(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_adj_hadamard(qubit_type const qubit)
  {
    ket::mpi::gate::adj_hadamard(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_not_(qubit_type const qubit)
  {
    ket::mpi::gate::not_(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_adj_not_(qubit_type const qubit)
  {
    ket::mpi::gate::adj_not_(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_pauli_x(qubit_type const qubit)
  {
    ket::mpi::gate::pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_adj_pauli_x(qubit_type const qubit)
  {
    ket::mpi::gate::adj_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void unit_mpi_state::do_adj_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void unit_mpi_state::do_pauli_xn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::pauli_x(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_adj_pauli_xn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_pauli_x(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_pauli_y(qubit_type const qubit)
  {
    ket::mpi::gate::pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_adj_pauli_y(qubit_type const qubit)
  {
    ket::mpi::gate::adj_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void unit_mpi_state::do_adj_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void unit_mpi_state::do_pauli_yn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::pauli_y(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_adj_pauli_yn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_pauli_y(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_pauli_z(qubit_type const qubit)
  {
    ket::mpi::gate::pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_adj_pauli_z(qubit_type const qubit)
  {
    ket::mpi::gate::adj_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void unit_mpi_state::do_adj_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void unit_mpi_state::do_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::pauli_z(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_adj_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_pauli_z(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_swap(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::swap(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void unit_mpi_state::do_adj_swap(qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_swap(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void unit_mpi_state::do_u1(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void unit_mpi_state::do_adj_u1(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void unit_mpi_state::do_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, qubit);
  }

  void unit_mpi_state::do_adj_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, qubit);
  }

  void unit_mpi_state::do_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, qubit);
  }

  void unit_mpi_state::do_adj_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, qubit);
  }

  void unit_mpi_state::do_phase_shift(
    complex_type const& phase_coefficient, qubit_type const qubit)
  {
    ket::mpi::gate::phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, qubit);
  }

  void unit_mpi_state::do_adj_phase_shift(
    complex_type const& phase_coefficient, qubit_type const qubit)
  {
    ket::mpi::gate::adj_phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, qubit);
  }

  void unit_mpi_state::do_x_rotation_half_pi(qubit_type const qubit)
  {
    ket::mpi::gate::x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_adj_x_rotation_half_pi(qubit_type const qubit)
  {
    ket::mpi::gate::adj_x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_y_rotation_half_pi(qubit_type const qubit)
  {
    ket::mpi::gate::y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_adj_y_rotation_half_pi(qubit_type const qubit)
  {
    ket::mpi::gate::adj_y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::controlled_v_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_controlled_v_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void unit_mpi_state::do_adj_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void unit_mpi_state::do_exponential_pauli_xx(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void unit_mpi_state::do_adj_exponential_pauli_xx(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void unit_mpi_state::do_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::exponential_pauli_x(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_adj_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_exponential_pauli_x(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void unit_mpi_state::do_adj_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void unit_mpi_state::do_exponential_pauli_yy(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void unit_mpi_state::do_adj_exponential_pauli_yy(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void unit_mpi_state::do_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::exponential_pauli_y(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_adj_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_exponential_pauli_y(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void unit_mpi_state::do_adj_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void unit_mpi_state::do_exponential_pauli_zz(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void unit_mpi_state::do_adj_exponential_pauli_zz(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void unit_mpi_state::do_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::exponential_pauli_z(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_adj_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_exponential_pauli_z(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_exponential_swap(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::exponential_swap(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void unit_mpi_state::do_adj_exponential_swap(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    ket::mpi::gate::adj_exponential_swap(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void unit_mpi_state::do_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    ket::mpi::gate::toffoli(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit1, control_qubit2);
  }

  void unit_mpi_state::do_adj_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    ket::mpi::gate::adj_toffoli(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit1, control_qubit2);
  }

  ::ket::gate::outcome unit_mpi_state::do_projective_measurement(
    qubit_type const qubit, yampi::rank const root)
  {
    return ket::mpi::gate::projective_measurement(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, root, communicator_, environment_, random_number_generator_, qubit);
  }

  void unit_mpi_state::do_expectation_values(yampi::rank const root)
  {
    maybe_expectation_values_
      = ket::mpi::all_spin_expectation_values<typename spins_type::allocator_type>(
          mpi_policy_, parallel_policy_,
          data_, permutation_, total_num_qubits_, buffer_, root, communicator_, environment_);
  }

  void unit_mpi_state::do_measure(yampi::rank const root)
  {
    measured_value_
      = ket::mpi::measure(
          mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
          data_, random_number_generator_, permutation_, communicator_, environment_);
  }

  void unit_mpi_state::do_generate_events(yampi::rank const root, int const num_events, int const seed)
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

  void unit_mpi_state::do_shor_box(
    state_integer_type const divisor, state_integer_type const base,
    std::vector<qubit_type> const& exponent_qubits,
    std::vector<qubit_type> const& modular_exponentiation_qubits)
  {
    ket::mpi::shor_box(
      mpi_policy_, parallel_policy_,
      data_, base, divisor, exponent_qubits, modular_exponentiation_qubits,
      permutation_, communicator_, environment_);
  }

# ifndef BRA_MAX_NUM_FUSED_QUBITS
#   ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
#   else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS 10
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
# endif // BRA_MAX_NUM_FUSED_QUBITS
  void unit_mpi_state::do_begin_fusion()
  {
    auto const max_num_fused_qubits
      = static_cast<decltype(fused_qubits_.size())>(
          std::min(
            {BRA_MAX_NUM_FUSED_QUBITS,
             static_cast<decltype(BRA_MAX_NUM_FUSED_QUBITS)>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))}));
    if (fused_qubits_.size() > max_num_fused_qubits)
      throw ::bra::too_many_operated_qubits_error{fused_qubits_.size(), max_num_fused_qubits};
  }

  void unit_mpi_state::do_end_fusion()
  {
    switch (fused_qubits_.size())
    {
# define QUBITS(z, n, qubits) BOOST_PP_COMMA_IF(n) qubits[n]
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   define CASE_N(z, num_fused_qubits, _) \
     case num_fused_qubits:\
      ket::mpi::gate::gate(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_,\
        [this](\
          auto const first, ::bra::state_integer_type const index_wo_qubits,\
          std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
          std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
          int const)\
        {\
          for (auto const& gate_ptr: this->fused_gates_)\
            gate_ptr->call(std::addressof(*first), index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
        }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
      break;\

# else // KET_USE_BIT_MASKS_EXPLICITLY
#   define CASE_N(z, num_fused_qubits, _) \
     case num_fused_qubits:\
      ket::mpi::gate::gate(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_,\
        [this](\
          auto const first, ::bra::state_integer_type const index_wo_qubits,\
          std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
          std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
          int const)\
        {\
          for (auto const& gate_ptr: this->fused_gates_)\
            gate_ptr->call(std::addressof(*first), index_wo_qubits, qubit_masks, index_masks);\
        }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
      break;\

# endif // KET_USE_BIT_MASKS_EXPLICITLY
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), CASE_N, nil)
# undef CASE_N
# undef QUBITS
    }
  }

  void unit_mpi_state::do_clear(qubit_type const qubit)
  {
    ket::mpi::gate::clear(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_set(qubit_type const qubit)
  {
    ket::mpi::gate::set(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void unit_mpi_state::do_controlled_i_gate(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_i_gate(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_in_gate(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::identity(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_in_gate(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_identity(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_hadamard(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::hadamard(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_hadamard(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_hadamard(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_hadamard(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits  + 1u> ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::hadamard(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_hadamard(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits  + 1u> ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_hadamard(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::not_(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_not_(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_not(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits  + 1u> ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::not_(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_not(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits  + 1u> ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_not_(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_pauli_xn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::pauli_x(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_pauli_xn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_pauli_x(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_pauli_yn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::pauli_y(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_pauli_yn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_pauli_y(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_pauli_z(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_pauli_z(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_multi_controlled_swap(
    qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    if (num_control_qubits + 2u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 2u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::swap(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit1, target_qubit2, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_DEC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{2u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_swap(
    qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    if (num_control_qubits + 2u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 2u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_swap(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit1, target_qubit2, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_DEC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{2u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_phase_shift_coeff(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::phase_shift_coeff(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_phase_shift_coeff(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_u1(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::phase_shift(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_u1(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_phase_shift(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_u1(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::phase_shift(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_u1(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_phase_shift(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_phase_shift2(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::phase_shift2(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_phase_shift2(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_phase_shift3(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::phase_shift3(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_phase_shift3(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_x_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::x_rotation_half_pi(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_x_rotation_half_pi(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_y_rotation_half_pi(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::y_rotation_half_pi(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_y_rotation_half_pi(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_multi_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::controlled_v_coeff(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_controlled_v_coeff(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_exponential_pauli_x(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_exponential_pauli_x(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_x(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::exponential_pauli_x(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_exponential_pauli_x(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_exponential_pauli_y(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_exponential_pauli_y(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_y(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::exponential_pauli_y(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_exponential_pauli_y(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_adj_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    ket::mpi::gate::adj_exponential_pauli_z(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void unit_mpi_state::do_multi_controlled_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::exponential_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_exponential_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil), BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
        BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_NC, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_NC
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void unit_mpi_state::do_multi_controlled_exponential_swap(
    real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    if (num_control_qubits + 2u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 2u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::exponential_swap(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_DEC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{2u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void unit_mpi_state::do_adj_multi_controlled_exponential_swap(
    real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    if (num_control_qubits + 2u > ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 2u, ::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_exponential_swap(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_DEC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{2u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }
} // namespace bra


#endif // BRA_NO_MPI
