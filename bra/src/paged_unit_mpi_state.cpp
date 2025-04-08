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

# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
#   include <ket/control_io.hpp>
# endif // KET_PRINT_LOG
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   include <ket/gate/utility/cache_aware_iterator.hpp>
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
# include <ket/mpi/gate/gate.hpp>
# include <ket/mpi/gate/identity.hpp>
# include <ket/mpi/gate/hadamard.hpp>
# include <ket/mpi/gate/not_.hpp>
# include <ket/mpi/gate/pauli_x.hpp>
# include <ket/mpi/gate/pauli_y.hpp>
# include <ket/mpi/gate/pauli_z.hpp>
# include <ket/mpi/gate/swap.hpp>
# include <ket/mpi/gate/sqrt_pauli_x.hpp>
# include <ket/mpi/gate/sqrt_pauli_y.hpp>
# include <ket/mpi/gate/sqrt_pauli_z.hpp>
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
# include <ket/mpi/utility/unit_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>

# include <bra/paged_unit_mpi_state.hpp>
# include <bra/state.hpp>
# include <bra/fused_gate.hpp>

# ifndef BRA_MAX_NUM_OPERATED_QUBITS
#   define BRA_MAX_NUM_OPERATED_QUBITS 6
# endif // BRA_MAX_NUM_OPERATED_QUBITS


namespace bra
{
  unsigned int paged_unit_mpi_state::do_num_page_qubits() const
  { return data_.num_page_qubits(); }

  unsigned int paged_unit_mpi_state::do_num_pages() const
  { return data_.num_pages(); }

# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  paged_unit_mpi_state::paged_unit_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const num_unit_qubits,
    unsigned int const total_num_qubits,
    unsigned int const num_page_qubits,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{num_unit_qubits, num_processes_per_unit},
      data_{
        mpi_policy_, num_local_qubits, num_page_qubits, initial_integer,
        permutation_, communicator, environment},
      fused_gates_{},
      paged_fused_gates_{},
      cache_aware_fused_gates_{},
      cache_aware_paged_fused_gates_{}
  { }

  paged_unit_mpi_state::paged_unit_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const num_unit_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int const num_page_qubits,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{num_unit_qubits, num_processes_per_unit},
      data_{
        mpi_policy_, num_local_qubits, num_page_qubits, initial_integer,
        permutation_, communicator, environment},
      fused_gates_{},
      paged_fused_gates_{},
      cache_aware_fused_gates_{},
      cache_aware_paged_fused_gates_{}
  { }
# elif !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION)
  paged_unit_mpi_state::paged_unit_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const num_unit_qubits,
    unsigned int const total_num_qubits,
    unsigned int const num_page_qubits,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{num_unit_qubits, num_processes_per_unit},
      data_{
        mpi_policy_, num_local_qubits, num_page_qubits, initial_integer,
        permutation_, communicator, environment},
      fused_gates_{},
      paged_fused_gates_{}
  { }

  paged_unit_mpi_state::paged_unit_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const num_unit_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int const num_page_qubits,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{num_unit_qubits, num_processes_per_unit},
      data_{
        mpi_policy_, num_local_qubits, num_page_qubits, initial_integer,
        permutation_, communicator, environment},
      fused_gates_{},
      paged_fused_gates_{}
  { }
# else
  paged_unit_mpi_state::paged_unit_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const num_unit_qubits,
    unsigned int const total_num_qubits,
    unsigned int const num_page_qubits,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{num_unit_qubits, num_processes_per_unit},
      data_{
        mpi_policy_, num_local_qubits, num_page_qubits, initial_integer,
        permutation_, communicator, environment},
      fused_gates_{}
  { }

  paged_unit_mpi_state::paged_unit_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const num_unit_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int const num_page_qubits,
    unsigned int const num_threads_per_process,
    unsigned int const num_processes_per_unit,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{num_unit_qubits, num_processes_per_unit},
      data_{
        mpi_policy_, num_local_qubits, num_page_qubits, initial_integer,
        permutation_, communicator, environment},
      fused_gates_{}
  { }
# endif

  void paged_unit_mpi_state::do_i_gate(qubit_type const qubit)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_ic_gate(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, control_qubit);
  }

  void paged_unit_mpi_state::do_ii_gate(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_in_gate(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      return;

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_hadamard(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_hadamard<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_hadamard<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_hadamard<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_hadamard<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_not_(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_not_<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_not_<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_not_<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_not_<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_x<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_x<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_x<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_x<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xx<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xx<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xx<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xx<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_pauli_xn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      auto qubits_in_fused_gate = std::vector<qubit_type>{};
      qubits_in_fused_gate.reserve(qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(qubits), end(qubits), std::back_inserter(qubits_in_fused_gate),
        [this](::bra::qubit_type const qubit) { return this->to_qubit_in_fused_gate_.at(qubit); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xn<paged_fused_gate_iterator> >(qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xn<cache_aware_fused_gate_iterator> >(qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xn<cache_aware_paged_fused_gate_iterator> >(qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xn<fused_gate_iterator> >(std::move(qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_y<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_y<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_y<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_y<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yy<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yy<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yy<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yy<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_pauli_yn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      auto qubits_in_fused_gate = std::vector<qubit_type>{};
      qubits_in_fused_gate.reserve(qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(qubits), end(qubits), std::back_inserter(qubits_in_fused_gate),
        [this](::bra::qubit_type const qubit) { return this->to_qubit_in_fused_gate_.at(qubit); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yn<paged_fused_gate_iterator> >(qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yn<cache_aware_fused_gate_iterator> >(qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yn<cache_aware_paged_fused_gate_iterator> >(qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yn<fused_gate_iterator> >(std::move(qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_z<fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_z<paged_fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_z<cache_aware_fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_z<cache_aware_paged_fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit);
  }

  void paged_unit_mpi_state::do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zz<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zz<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zz<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zz<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      auto qubits_in_fused_gate = std::vector<qubit_type>{};
      qubits_in_fused_gate.reserve(qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(qubits), end(qubits), std::back_inserter(qubits_in_fused_gate),
        [this](::bra::qubit_type const qubit) { return this->to_qubit_in_fused_gate_.at(qubit); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zn<paged_fused_gate_iterator> >(qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zn<cache_aware_fused_gate_iterator> >(qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zn<cache_aware_paged_fused_gate_iterator> >(qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zn<fused_gate_iterator> >(std::move(qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_swap(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_swap<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_swap<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_swap<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_swap<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_sqrt_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_x<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_x<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_x<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_adj_sqrt_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_x<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_x<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_x<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_sqrt_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_y<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_y<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_y<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_adj_sqrt_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_y<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_y<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_y<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_sqrt_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_z<fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_z<paged_fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_z<cache_aware_paged_fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_sqrt_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_z<fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_z<paged_fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_z<cache_aware_paged_fused_gate_iterator> >(ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit);
  }

  void paged_unit_mpi_state::do_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zz<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zz<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zz<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zz<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_adj_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zz<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zz<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zz<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zz<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_sqrt_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      auto qubits_in_fused_gate = std::vector<qubit_type>{};
      qubits_in_fused_gate.reserve(qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(qubits), end(qubits), std::back_inserter(qubits_in_fused_gate),
        [this](::bra::qubit_type const qubit) { return this->to_qubit_in_fused_gate_.at(qubit); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zn<paged_fused_gate_iterator> >(qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zn<cache_aware_paged_fused_gate_iterator> >(qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zn<fused_gate_iterator> >(std::move(qubits_in_fused_gate)));

      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::sqrt_pauli_z(\
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

  void paged_unit_mpi_state::do_adj_sqrt_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      auto qubits_in_fused_gate = std::vector<qubit_type>{};
      qubits_in_fused_gate.reserve(qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(qubits), end(qubits), std::back_inserter(qubits_in_fused_gate),
        [this](::bra::qubit_type const qubit) { return this->to_qubit_in_fused_gate_.at(qubit); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zn<paged_fused_gate_iterator> >(qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zn<cache_aware_paged_fused_gate_iterator> >(qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zn<fused_gate_iterator> >(std::move(qubits_in_fused_gate)));

      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_sqrt_pauli_z(\
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

  void paged_unit_mpi_state::do_u1(real_type const phase, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u1<fused_gate_iterator> >(phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u1<paged_fused_gate_iterator> >(phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u1<cache_aware_fused_gate_iterator> >(phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u1<cache_aware_paged_fused_gate_iterator> >(phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_u1(real_type const phase, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u1<fused_gate_iterator> >(phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u1<paged_fused_gate_iterator> >(phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u1<cache_aware_fused_gate_iterator> >(phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u1<cache_aware_paged_fused_gate_iterator> >(phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, control_qubit);
  }

  void paged_unit_mpi_state::do_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u2<fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u2<paged_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u2<cache_aware_paged_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, qubit);
  }

  void paged_unit_mpi_state::do_adj_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u2<fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u2<paged_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u2<cache_aware_paged_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, qubit);
  }

  void paged_unit_mpi_state::do_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u3<fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u3<paged_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u3<cache_aware_paged_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, qubit);
  }

  void paged_unit_mpi_state::do_adj_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u3<fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u3<paged_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u3<cache_aware_paged_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, qubit);
  }

  void paged_unit_mpi_state::do_phase_shift(
    complex_type const& phase_coefficient, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_phase_shift<fused_gate_iterator> >(phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_phase_shift<paged_fused_gate_iterator> >(phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_phase_shift<cache_aware_paged_fused_gate_iterator> >(phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_phase_shift(
    complex_type const& phase_coefficient, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_phase_shift<fused_gate_iterator> >(phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_phase_shift<paged_fused_gate_iterator> >(phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_phase_shift<cache_aware_paged_fused_gate_iterator> >(phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, control_qubit);
  }

  void paged_unit_mpi_state::do_x_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_x_rotation_half_pi<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_x_rotation_half_pi<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_x_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_adj_x_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_x_rotation_half_pi<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_x_rotation_half_pi<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_x_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_y_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_y_rotation_half_pi<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_y_rotation_half_pi<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_y_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_adj_y_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_y_rotation_half_pi<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_y_rotation_half_pi<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_y_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_v<fused_gate_iterator> >(
          phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_v<paged_fused_gate_iterator> >(
          phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_v<cache_aware_fused_gate_iterator> >(
          phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_v<cache_aware_paged_fused_gate_iterator> >(
          phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::controlled_v_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_v<fused_gate_iterator> >(
          phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_v<paged_fused_gate_iterator> >(
          phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_v<cache_aware_fused_gate_iterator> >(
          phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_v<cache_aware_paged_fused_gate_iterator> >(
          phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_controlled_v_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_x<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_x<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_x<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_x<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_unit_mpi_state::do_adj_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_x<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_x<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_x<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_x<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_unit_mpi_state::do_exponential_pauli_xx(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xx<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xx<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xx<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xx<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_adj_exponential_pauli_xx(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xx<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xx<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xx<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xx<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      auto qubits_in_fused_gate = std::vector<qubit_type>{};
      qubits_in_fused_gate.reserve(qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(qubits), end(qubits), std::back_inserter(qubits_in_fused_gate),
        [this](::bra::qubit_type const qubit) { return this->to_qubit_in_fused_gate_.at(qubit); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xn<paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xn<cache_aware_paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xn<fused_gate_iterator> >(phase, std::move(qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      auto qubits_in_fused_gate = std::vector<qubit_type>{};
      qubits_in_fused_gate.reserve(qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(qubits), end(qubits), std::back_inserter(qubits_in_fused_gate),
        [this](::bra::qubit_type const qubit) { return this->to_qubit_in_fused_gate_.at(qubit); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xn<paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xn<cache_aware_paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xn<fused_gate_iterator> >(phase, std::move(qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_y<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_y<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_y<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_y<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_unit_mpi_state::do_adj_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_y<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_y<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_y<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_y<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_unit_mpi_state::do_exponential_pauli_yy(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yy<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yy<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yy<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yy<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_adj_exponential_pauli_yy(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yy<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yy<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yy<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yy<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      auto qubits_in_fused_gate = std::vector<qubit_type>{};
      qubits_in_fused_gate.reserve(qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(qubits), end(qubits), std::back_inserter(qubits_in_fused_gate),
        [this](::bra::qubit_type const qubit) { return this->to_qubit_in_fused_gate_.at(qubit); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yn<paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yn<cache_aware_paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yn<fused_gate_iterator> >(phase, std::move(qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      auto qubits_in_fused_gate = std::vector<qubit_type>{};
      qubits_in_fused_gate.reserve(qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(qubits), end(qubits), std::back_inserter(qubits_in_fused_gate),
        [this](::bra::qubit_type const qubit) { return this->to_qubit_in_fused_gate_.at(qubit); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yn<paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yn<cache_aware_paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yn<fused_gate_iterator> >(phase, std::move(qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_z<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_z<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_z<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_unit_mpi_state::do_adj_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_z<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_z<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_z<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void paged_unit_mpi_state::do_exponential_pauli_zz(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zz<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zz<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zz<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zz<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_adj_exponential_pauli_zz(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zz<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zz<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zz<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zz<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      auto qubits_in_fused_gate = std::vector<qubit_type>{};
      qubits_in_fused_gate.reserve(qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(qubits), end(qubits), std::back_inserter(qubits_in_fused_gate),
        [this](::bra::qubit_type const qubit) { return this->to_qubit_in_fused_gate_.at(qubit); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zn<paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zn<cache_aware_paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zn<fused_gate_iterator> >(phase, std::move(qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      auto qubits_in_fused_gate = std::vector<qubit_type>{};
      qubits_in_fused_gate.reserve(qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(qubits), end(qubits), std::back_inserter(qubits_in_fused_gate),
        [this](::bra::qubit_type const qubit) { return this->to_qubit_in_fused_gate_.at(qubit); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zn<paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zn<cache_aware_paged_fused_gate_iterator> >(phase, qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zn<fused_gate_iterator> >(phase, std::move(qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_exponential_swap(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_swap<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_swap<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_swap<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_swap<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_adj_exponential_swap(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_swap<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_swap<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_swap<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_swap<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(qubit1), to_qubit_in_fused_gate_.at(qubit2)));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void paged_unit_mpi_state::do_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_toffoli<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit),
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())),
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_toffoli<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit),
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())),
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_toffoli<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit),
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())),
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_toffoli<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit),
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())),
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::toffoli(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit1, control_qubit2);
  }

  ::ket::gate::outcome paged_unit_mpi_state::do_projective_measurement(
    qubit_type const qubit, yampi::rank const root)
  {
    return ket::mpi::gate::projective_measurement(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, root, communicator_, environment_, random_number_generator_, qubit);
  }

  void paged_unit_mpi_state::do_expectation_values(yampi::rank const root)
  {
    maybe_expectation_values_
      = ket::mpi::all_spin_expectation_values<typename spins_type::allocator_type>(
          mpi_policy_, parallel_policy_,
          data_, permutation_, total_num_qubits_, buffer_, root, communicator_, environment_);
  }

  void paged_unit_mpi_state::do_measure(yampi::rank const root)
  {
    measured_value_
      = ket::mpi::measure(
          mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
          data_, random_number_generator_, permutation_, communicator_, environment_);
  }

  void paged_unit_mpi_state::do_generate_events(yampi::rank const root, int const num_events, int const seed)
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

  void paged_unit_mpi_state::do_shor_box(
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
  void paged_unit_mpi_state::do_begin_fusion()
  {
    auto const max_num_fused_qubits
      = static_cast<decltype(fused_qubits_.size())>(
          std::min(
            {BRA_MAX_NUM_FUSED_QUBITS,
             static_cast<decltype(BRA_MAX_NUM_FUSED_QUBITS)>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))}));
    if (fused_qubits_.size() > max_num_fused_qubits)
      throw ::bra::too_many_operated_qubits_error{fused_qubits_.size(), max_num_fused_qubits};
  }

  void paged_unit_mpi_state::do_end_fusion()
  {
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
    constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
    constexpr auto cache_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);
    assert(fused_gates_.size() == paged_fused_gates_.size());
    assert(fused_gates_.size() == cache_aware_fused_gates_.size());
    assert(fused_gates_.size() == cache_aware_paged_fused_gates_.size());

    ::ket::mpi::utility::logger logger{environment_};

# elif !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION)
    assert(fused_gates_.size() == paged_fused_gates_.size());

    ::ket::mpi::utility::logger logger{environment_};

# endif
    switch (fused_qubits_.size())
    {
# define QUBITS_OF_PERMUTATED_QUBITS(z, n, qubits) BOOST_PP_COMMA_IF(n) permutation_[qubits[n]].qubit()
# define PERMUTATED_QUBITS(z, n, qubits) BOOST_PP_COMMA_IF(n) permutation_[qubits[n]]
# define QUBITS(z, n, qubits) BOOST_PP_COMMA_IF(n) qubits[n]
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_N(z, num_fused_qubits, _) \
     case num_fused_qubits:\
      logger.print("[start] " + ::ket::mpi::gate::detail::append_qubits_string(std::string{"Gate"}, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_)), environment_);\
\
      ket::mpi::utility::maybe_interchange_qubits(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_,\
        BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
\
      if (ket::mpi::page::none_on_page(data_, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_)))\
        if (ket::utility::all_in_state_vector(num_on_cache_qubits, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS_OF_PERMUTATED_QUBITS, fused_qubits_)))\
          if (ket::mpi::page::page_size(mpi_policy_, data_, communicator_, environment_) <= cache_size)\
            ket::mpi::gate::local::nopage::all_on_cache::small::gate(\
              mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
              [this](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
                std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
                int const)\
              {\
                for (auto const& gate_ptr: this->fused_gates_)\
                  gate_ptr->call(first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
              }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
          else\
            ket::mpi::gate::local::nopage::all_on_cache::gate(\
              mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
              [this](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
                std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
                int const)\
              {\
                for (auto const& gate_ptr: this->fused_gates_)\
                  gate_ptr->call(first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
              }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
        else if (ket::utility::none_in_state_vector(num_on_cache_qubits, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS_OF_PERMUTATED_QUBITS, fused_qubits_)))\
          ket::mpi::gate::local::nopage::none_on_cache::gate(\
            mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
            [this](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                gate_ptr->call(first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
            }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
        else\
          ket::mpi::gate::local::nopage::some_on_cache::gate(\
            mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
            [this](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                gate_ptr->call(first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
            }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
      else\
        if (ket::utility::all_in_state_vector(num_on_cache_qubits, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS_OF_PERMUTATED_QUBITS, fused_qubits_)))\
          if (ket::mpi::page::page_size(mpi_policy_, data_, communicator_, environment_) <= cache_size)\
            ket::mpi::gate::local::nopage::all_on_cache::small::gate(\
              mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
              [this](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
                std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
                int const)\
              {\
                for (auto const& gate_ptr: this->paged_fused_gates_)\
                  gate_ptr->call(first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
              }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
          else\
            ket::mpi::gate::local::nopage::all_on_cache::gate(\
              mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
              [this](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
                std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
                int const)\
              {\
                for (auto const& gate_ptr: this->paged_fused_gates_)\
                  gate_ptr->call(first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
              }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
        else if (ket::utility::none_in_state_vector(num_on_cache_qubits, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS_OF_PERMUTATED_QUBITS, fused_qubits_)))\
          ket::mpi::gate::local::nopage::none_on_cache::gate(\
            mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
            [this](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_paged_fused_gates_)\
                gate_ptr->call(first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
            }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
        else\
          ket::mpi::gate::local::nopage::some_on_cache::gate(\
            mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
            [this](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
              std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_paged_fused_gates_)\
                gate_ptr->call(first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
            }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
\
      logger.print_with_time("[end] " + ::ket::mpi::gate::detail::append_qubits_string(std::string{"Gate"}, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_)), environment_);\
      break;\

#   else // KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_N(z, num_fused_qubits, _) \
     case num_fused_qubits:\
      logger.print("[start] " + ::ket::mpi::gate::detail::append_qubits_string(std::string{"Gate"}, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_)), environment_);\
\
      ket::mpi::utility::maybe_interchange_qubits(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_,\
        BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
\
      if (ket::mpi::page::none_on_page(data_, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_)))\
        if (ket::utility::all_in_state_vector(num_on_cache_qubits, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS_OF_PERMUTATED_QUBITS, fused_qubits_)))\
          if (ket::mpi::page::page_size(mpi_policy_, data_, communicator_, environment_) <= cache_size)\
            ket::mpi::gate::local::nopage::all_on_cache::small::gate(\
              mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
              [this](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
                std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
                int const)\
              {\
                for (auto const& gate_ptr: this->fused_gates_)\
                  gate_ptr->call(first, index_wo_qubits, qubit_masks, index_masks);\
              }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
          else\
            ket::mpi::gate::local::nopage::all_on_cache::gate(\
              mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
              [this](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
                std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
                int const)\
              {\
                for (auto const& gate_ptr: this->fused_gates_)\
                  gate_ptr->call(first, index_wo_qubits, qubit_masks, index_masks);\
              }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
        else if (ket::utility::none_in_state_vector(num_on_cache_qubits, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS_OF_PERMUTATED_QUBITS, fused_qubits_)))\
          ket::mpi::gate::local::nopage::none_on_cache::gate(\
            mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
            [this](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
              std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                gate_ptr->call(first, index_wo_qubits, qubit_masks, index_masks);\
            }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
        else\
          ket::mpi::gate::local::nopage::some_on_cache::gate(\
            mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
            [this](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
              std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                gate_ptr->call(first, index_wo_qubits, qubit_masks, index_masks);\
            }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
      else\
        if (ket::utility::all_in_state_vector(num_on_cache_qubits, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS_OF_PERMUTATED_QUBITS, fused_qubits_)))\
          if (ket::mpi::page::page_size(mpi_policy_, data_, communicator_, environment_) <= cache_size)\
            ket::mpi::gate::local::nopage::all_on_cache::small::gate(\
              mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
              [this](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
                std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
                int const)\
              {\
                for (auto const& gate_ptr: this->paged_fused_gates_)\
                  gate_ptr->call(first, index_wo_qubits, qubit_masks, index_masks);\
              }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
          else\
            ket::mpi::gate::local::nopage::all_on_cache::gate(\
              mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
              [this](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
                std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
                int const)\
              {\
                for (auto const& gate_ptr: this->paged_fused_gates_)\
                  gate_ptr->call(first, index_wo_qubits, qubit_masks, index_masks);\
              }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
        else if (ket::utility::none_in_state_vector(num_on_cache_qubits, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS_OF_PERMUTATED_QUBITS, fused_qubits_)))\
          ket::mpi::gate::local::nopage::none_on_cache::gate(\
            mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
            [this](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
              std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_paged_fused_gates_)\
                gate_ptr->call(first, index_wo_qubits, qubit_masks, index_masks);\
            }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
        else\
          ket::mpi::gate::local::nopage::some_on_cache::gate(\
            mpi_policy_, parallel_policy_, data_, communicator_, environment_,\
            [this](\
              auto const first, ::bra::state_integer_type const index_wo_qubits,\
              std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
              std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
              int const)\
            {\
              for (auto const& gate_ptr: this->cache_aware_paged_fused_gates_)\
                gate_ptr->call(first, index_wo_qubits, qubit_masks, index_masks);\
            }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
\
      logger.print_with_time("[end] " + ::ket::mpi::gate::detail::append_qubits_string(std::string{"Gate"}, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_)), environment_);\
      break;\

#   endif // KET_USE_BIT_MASKS_EXPLICITLY
# elif !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION)
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_N(z, num_fused_qubits, _) \
     case num_fused_qubits:\
      logger.print("[start] " + ::ket::mpi::gate::detail::append_qubits_string(std::string{"Gate"}, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_)), environment_);\
\
      ket::mpi::utility::maybe_interchange_qubits(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_,\
        BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
\
      if (ket::mpi::page::none_on_page(data_, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_)))\
        ket::mpi::gate::local::nopage::gate(\
          mpi_policy_, parallel_policy_, data_, buffer_, communicator_, environment_,\
          [this](\
            auto const first, ::bra::state_integer_type const index_wo_qubits,\
            std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
            std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
            int const)\
          {\
            for (auto const& gate_ptr: this->fused_gates_)\
              gate_ptr->call(first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
          }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
      else\
        ket::mpi::gate::local::page::gate(\
          mpi_policy_, parallel_policy_, data_, buffer_, communicator_, environment_,\
          [this](\
            auto const first, ::bra::state_integer_type const index_wo_qubits,\
            std::array< ::bra::qubit_type, num_fused_qubits > const& unsorted_fused_qubits,\
            std::array< ::bra::qubit_type, num_fused_qubits + 1u > const& sorted_fused_qubits_with_sentinel,\
            int const)\
          {\
            for (auto const& gate_ptr: this->paged_fused_gates_)\
              gate_ptr->call(first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
          }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
\
      logger.print_with_time("[end] " + ::ket::mpi::gate::detail::append_qubits_string(std::string{"Gate"}, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_)), environment_);\
      break;\

#   else // KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_N(z, num_fused_qubits, _) \
     case num_fused_qubits:\
      logger.print("[start] " + ::ket::mpi::gate::detail::append_qubits_string(std::string{"Gate"}, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_)), environment_);\
\
      ket::mpi::utility::maybe_interchange_qubits(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_,\
        BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
\
      if (ket::mpi::page::none_on_page(data_, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_)))\
        ket::mpi::gate::local::nopage::gate(\
          mpi_policy_, parallel_policy_, data_, buffer_, communicator_, environment_,\
          [this](\
            auto const first, ::bra::state_integer_type const index_wo_qubits,\
            std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
            std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
            int const)\
          {\
            for (auto const& gate_ptr: this->fused_gates_)\
              gate_ptr->call(first, index_wo_qubits, qubit_masks, index_masks);\
          }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
      else\
        ket::mpi::gate::local::page::gate(\
          mpi_policy_, parallel_policy_, data_, buffer_, communicator_, environment_,\
          [this](\
            auto const first, ::bra::state_integer_type const index_wo_qubits,\
            std::array< ::bra::state_integer_type, num_fused_qubits > const& qubit_masks,\
            std::array< ::bra::state_integer_type, num_fused_qubits + 1u > const& index_masks,\
            int const)\
          {\
            for (auto const& gate_ptr: this->paged_fused_gates_)\
              gate_ptr->call(first, index_wo_qubits, qubit_masks, index_masks);\
          }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, PERMUTATED_QUBITS, fused_qubits_));\
\
      logger.print_with_time("[end] " + ::ket::mpi::gate::detail::append_qubits_string(std::string{"Gate"}, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_)), environment_);\
      break;\

#   endif // KET_USE_BIT_MASKS_EXPLICITLY
# else
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_N(z, num_fused_qubits, _) \
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
            gate_ptr->call(first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel);\
        }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
      break;\

#   else // KET_USE_BIT_MASKS_EXPLICITLY
#     define CASE_N(z, num_fused_qubits, _) \
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
            gate_ptr->call(first, index_wo_qubits, qubit_masks, index_masks);\
        }, BOOST_PP_REPEAT_ ## z(num_fused_qubits, QUBITS, fused_qubits_));\
      break;\

#   endif // KET_USE_BIT_MASKS_EXPLICITLY
# endif
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), CASE_N, nil)
# undef CASE_N
# undef QUBITS
# undef PERMUTATED_QUBITS
# undef QUBITS_OF_PERMUTATED_QUBITS
    }

    fused_gates_.clear();
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
    paged_fused_gates_.clear();
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    cache_aware_fused_gates_.clear();
    cache_aware_paged_fused_gates_.clear();
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  }

  void paged_unit_mpi_state::do_clear(qubit_type const qubit)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"CLEAR"};

    ket::mpi::gate::clear(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_set(qubit_type const qubit)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"SET"};

    ket::mpi::gate::set(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void paged_unit_mpi_state::do_controlled_i_gate(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_controlled_ic_gate(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, control_qubit1, control_qubit2);
  }

  void paged_unit_mpi_state::do_multi_controlled_in_gate(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      return;

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_multi_controlled_ic_gate(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      return;

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::identity(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void paged_unit_mpi_state::do_controlled_hadamard(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_hadamard<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_hadamard<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_hadamard<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_hadamard<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_hadamard(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_hadamard<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_hadamard<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_hadamard<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_hadamard<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_not<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_not<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_not<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_not<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_not(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_not<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_not<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_not<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_not<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_controlled_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_x<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_x<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_x<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_x<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_pauli_xn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto target_qubits_in_fused_gate = std::vector<qubit_type>{};
      target_qubits_in_fused_gate.reserve(target_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(target_qubits), end(target_qubits), std::back_inserter(target_qubits_in_fused_gate),
        [this](::bra::qubit_type const target_qubit) { return this->to_qubit_in_fused_gate_.at(target_qubit); });

      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_xn<paged_fused_gate_iterator> >(target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_xn<cache_aware_fused_gate_iterator> >(target_qubits_in_fused_gate, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_xn<cache_aware_paged_fused_gate_iterator> >(target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_xn<fused_gate_iterator> >(std::move(target_qubits_in_fused_gate), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_controlled_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_y<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_y<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_y<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_y<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_pauli_yn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto target_qubits_in_fused_gate = std::vector<qubit_type>{};
      target_qubits_in_fused_gate.reserve(target_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(target_qubits), end(target_qubits), std::back_inserter(target_qubits_in_fused_gate),
        [this](::bra::qubit_type const target_qubit) { return this->to_qubit_in_fused_gate_.at(target_qubit); });

      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_yn<paged_fused_gate_iterator> >(target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_yn<cache_aware_fused_gate_iterator> >(target_qubits_in_fused_gate, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_yn<cache_aware_paged_fused_gate_iterator> >(target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_yn<fused_gate_iterator> >(std::move(target_qubits_in_fused_gate), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_controlled_pauli_z(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_z<fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_z<paged_fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_z<cache_aware_fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_z<cache_aware_paged_fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit1, control_qubit2);
  }

  void paged_unit_mpi_state::do_multi_controlled_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_z<paged_fused_gate_iterator> >(control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_z<cache_aware_fused_gate_iterator> >(control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_z<cache_aware_paged_fused_gate_iterator> >(control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_z<fused_gate_iterator> >(std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_operated_qubits, num_target_qubits) \
       case num_operated_qubits:\
        ket::mpi::gate::pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void paged_unit_mpi_state::do_multi_controlled_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto target_qubits_in_fused_gate = std::vector<qubit_type>{};
      target_qubits_in_fused_gate.reserve(target_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(target_qubits), end(target_qubits), std::back_inserter(target_qubits_in_fused_gate),
        [this](::bra::qubit_type const target_qubit) { return this->to_qubit_in_fused_gate_.at(target_qubit); });

      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_zn<paged_fused_gate_iterator> >(target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_zn<cache_aware_fused_gate_iterator> >(target_qubits_in_fused_gate, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_zn<cache_aware_paged_fused_gate_iterator> >(target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_zn<fused_gate_iterator> >(std::move(target_qubits_in_fused_gate), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_multi_controlled_swap(
    qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_swap<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_swap<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_swap<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_swap<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    if (num_control_qubits + 2u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 2u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_x<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_x<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_x<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_x<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_x<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_x<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_x<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_x<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_x<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));

      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::sqrt_pauli_x(\
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

  void paged_unit_mpi_state::do_adj_multi_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_x<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_x<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_x<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));

      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_sqrt_pauli_x(\
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

  void paged_unit_mpi_state::do_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_y<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_y<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_y<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_y<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_y<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_y<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_y<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_y<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_y<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));

      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::sqrt_pauli_y(\
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

  void paged_unit_mpi_state::do_adj_multi_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_y<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_y<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_y<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));

      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_sqrt_pauli_y(\
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

  void paged_unit_mpi_state::do_controlled_sqrt_pauli_z(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_z<fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_z<paged_fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_z<cache_aware_paged_fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit1, control_qubit2);
  }

  void paged_unit_mpi_state::do_adj_controlled_sqrt_pauli_z(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_z<fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_z<paged_fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_z<cache_aware_paged_fused_gate_iterator> >(
          ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit1, control_qubit2);
  }

  void paged_unit_mpi_state::do_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_z<paged_fused_gate_iterator> >(control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_z<cache_aware_paged_fused_gate_iterator> >(control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_z<fused_gate_iterator> >(std::move(control_qubits_in_fused_gate)));

      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::sqrt_pauli_z(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void paged_unit_mpi_state::do_adj_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_z<paged_fused_gate_iterator> >(control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_z<cache_aware_paged_fused_gate_iterator> >(control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_z<fused_gate_iterator> >(std::move(control_qubits_in_fused_gate)));

      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_sqrt_pauli_z(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void paged_unit_mpi_state::do_multi_controlled_sqrt_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto target_qubits_in_fused_gate = std::vector<qubit_type>{};
      target_qubits_in_fused_gate.reserve(target_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(target_qubits), end(target_qubits), std::back_inserter(target_qubits_in_fused_gate),
        [this](::bra::qubit_type const target_qubit) { return this->to_qubit_in_fused_gate_.at(target_qubit); });

      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_zn<paged_fused_gate_iterator> >(
          target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(
          target_qubits_in_fused_gate, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_zn<cache_aware_paged_fused_gate_iterator> >(
          target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_zn<fused_gate_iterator> >(
          std::move(target_qubits_in_fused_gate), std::move(control_qubits_in_fused_gate)));

      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::sqrt_pauli_z(\
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

  void paged_unit_mpi_state::do_adj_multi_controlled_sqrt_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto target_qubits_in_fused_gate = std::vector<qubit_type>{};
      target_qubits_in_fused_gate.reserve(target_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(target_qubits), end(target_qubits), std::back_inserter(target_qubits_in_fused_gate),
        [this](::bra::qubit_type const target_qubit) { return this->to_qubit_in_fused_gate_.at(target_qubit); });

      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_zn<paged_fused_gate_iterator> >(
          target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(
          target_qubits_in_fused_gate, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_zn<cache_aware_paged_fused_gate_iterator> >(
          target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_zn<fused_gate_iterator> >(
          std::move(target_qubits_in_fused_gate), std::move(control_qubits_in_fused_gate)));

      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_target_qubits)
    {
# define TARGET_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) target_qubits[n]
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_NC(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_sqrt_pauli_z(\
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

  void paged_unit_mpi_state::do_controlled_phase_shift(
    complex_type const& phase_coefficient,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_phase_shift<fused_gate_iterator> >(
          phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_phase_shift<paged_fused_gate_iterator> >(
          phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_phase_shift<cache_aware_fused_gate_iterator> >(
          phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_phase_shift<cache_aware_paged_fused_gate_iterator> >(
          phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, control_qubit1, control_qubit2);
  }

  void paged_unit_mpi_state::do_adj_controlled_phase_shift(
    complex_type const& phase_coefficient,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_phase_shift<fused_gate_iterator> >(
          phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_phase_shift<paged_fused_gate_iterator> >(
          phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_phase_shift<cache_aware_fused_gate_iterator> >(
          phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_phase_shift<cache_aware_paged_fused_gate_iterator> >(
          phase_coefficient, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, control_qubit1, control_qubit2);
  }

  void paged_unit_mpi_state::do_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_phase_shift<paged_fused_gate_iterator> >(phase_coefficient, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_phase_shift<cache_aware_paged_fused_gate_iterator> >(phase_coefficient, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_phase_shift<fused_gate_iterator> >(phase_coefficient, std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 1u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::phase_shift_coeff(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void paged_unit_mpi_state::do_adj_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_phase_shift<paged_fused_gate_iterator> >(phase_coefficient, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_phase_shift<cache_aware_paged_fused_gate_iterator> >(phase_coefficient, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_phase_shift<fused_gate_iterator> >(phase_coefficient, std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 1u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_phase_shift_coeff(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void paged_unit_mpi_state::do_controlled_u1(
    real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u1<fused_gate_iterator> >(
          phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u1<paged_fused_gate_iterator> >(
          phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u1<cache_aware_fused_gate_iterator> >(
          phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u1<cache_aware_paged_fused_gate_iterator> >(
          phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, control_qubit1, control_qubit2);
  }

  void paged_unit_mpi_state::do_adj_controlled_u1(
    real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u1<fused_gate_iterator> >(
          phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u1<paged_fused_gate_iterator> >(
          phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u1<cache_aware_fused_gate_iterator> >(
          phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u1<cache_aware_paged_fused_gate_iterator> >(
          phase, ket::make_control(to_qubit_in_fused_gate_.at(control_qubit1.qubit())), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit2.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, control_qubit1, control_qubit2);
  }

  void paged_unit_mpi_state::do_multi_controlled_u1(
    real_type const phase, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u1<paged_fused_gate_iterator> >(phase, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u1<cache_aware_fused_gate_iterator> >(phase, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u1<cache_aware_paged_fused_gate_iterator> >(phase, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u1<fused_gate_iterator> >(phase, std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::phase_shift(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void paged_unit_mpi_state::do_adj_multi_controlled_u1(
    real_type const phase, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u1<paged_fused_gate_iterator> >(phase, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u1<cache_aware_fused_gate_iterator> >(phase, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u1<cache_aware_paged_fused_gate_iterator> >(phase, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u1<fused_gate_iterator> >(phase, std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_phase_shift(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void paged_unit_mpi_state::do_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u2<fused_gate_iterator> >(
          phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u2<paged_fused_gate_iterator> >(
          phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u2<cache_aware_fused_gate_iterator> >(
          phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u2<cache_aware_paged_fused_gate_iterator> >(
          phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u2<fused_gate_iterator> >(
          phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u2<paged_fused_gate_iterator> >(
          phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u2<cache_aware_fused_gate_iterator> >(
          phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u2<cache_aware_paged_fused_gate_iterator> >(
          phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u2<paged_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u2<cache_aware_paged_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u2<fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_multi_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u2<paged_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u2<cache_aware_paged_fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u2<fused_gate_iterator> >(phase1, phase2, to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u3<fused_gate_iterator> >(
          phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u3<paged_fused_gate_iterator> >(
          phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u3<cache_aware_fused_gate_iterator> >(
          phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u3<cache_aware_paged_fused_gate_iterator> >(
          phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u3<fused_gate_iterator> >(
          phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u3<paged_fused_gate_iterator> >(
          phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u3<cache_aware_fused_gate_iterator> >(
          phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u3<cache_aware_paged_fused_gate_iterator> >(
          phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u3<paged_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u3<cache_aware_paged_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u3<fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_multi_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u3<paged_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u3<cache_aware_paged_fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u3<fused_gate_iterator> >(phase1, phase2, phase3, to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_x_rotation_half_pi<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_x_rotation_half_pi<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_x_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_x_rotation_half_pi<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_x_rotation_half_pi<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_x_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_x_rotation_half_pi<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_x_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_x_rotation_half_pi<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_multi_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_x_rotation_half_pi<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_x_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_x_rotation_half_pi<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_y_rotation_half_pi<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_y_rotation_half_pi<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_y_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_y_rotation_half_pi<fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_y_rotation_half_pi<paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_y_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(
          to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_y_rotation_half_pi<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_y_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_y_rotation_half_pi<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_multi_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_y_rotation_half_pi<paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_y_rotation_half_pi<cache_aware_paged_fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_y_rotation_half_pi<fused_gate_iterator> >(to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_multi_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_v<paged_fused_gate_iterator> >(phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_v<cache_aware_fused_gate_iterator> >(phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_v<cache_aware_paged_fused_gate_iterator> >(phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_v<fused_gate_iterator> >(phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_multi_controlled_v(
    complex_type const& phase_coefficient,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_v<paged_fused_gate_iterator> >(phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_v<cache_aware_fused_gate_iterator> >(phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_v<cache_aware_paged_fused_gate_iterator> >(phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_v<fused_gate_iterator> >(phase_coefficient, to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_controlled_exponential_pauli_x(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_x<fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_x<paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_x<cache_aware_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_x<cache_aware_paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_controlled_exponential_pauli_x(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_x<fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_x<paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_x<cache_aware_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_x<cache_aware_paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto target_qubits_in_fused_gate = std::vector<qubit_type>{};
      target_qubits_in_fused_gate.reserve(target_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(target_qubits), end(target_qubits), std::back_inserter(target_qubits_in_fused_gate),
        [this](::bra::qubit_type const target_qubit) { return this->to_qubit_in_fused_gate_.at(target_qubit); });

      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_xn<paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_xn<cache_aware_paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_xn<fused_gate_iterator> >(phase, std::move(target_qubits_in_fused_gate), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_multi_controlled_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto target_qubits_in_fused_gate = std::vector<qubit_type>{};
      target_qubits_in_fused_gate.reserve(target_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(target_qubits), end(target_qubits), std::back_inserter(target_qubits_in_fused_gate),
        [this](::bra::qubit_type const target_qubit) { return this->to_qubit_in_fused_gate_.at(target_qubit); });

      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_xn<paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_xn<cache_aware_paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_xn<fused_gate_iterator> >(phase, std::move(target_qubits_in_fused_gate), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_controlled_exponential_pauli_y(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_y<fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_y<paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_y<cache_aware_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_y<cache_aware_paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_controlled_exponential_pauli_y(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_y<fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_y<paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_y<cache_aware_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_y<cache_aware_paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto target_qubits_in_fused_gate = std::vector<qubit_type>{};
      target_qubits_in_fused_gate.reserve(target_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(target_qubits), end(target_qubits), std::back_inserter(target_qubits_in_fused_gate),
        [this](::bra::qubit_type const target_qubit) { return this->to_qubit_in_fused_gate_.at(target_qubit); });

      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_yn<paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_yn<cache_aware_paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_yn<fused_gate_iterator> >(phase, std::move(target_qubits_in_fused_gate), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_multi_controlled_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto target_qubits_in_fused_gate = std::vector<qubit_type>{};
      target_qubits_in_fused_gate.reserve(target_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(target_qubits), end(target_qubits), std::back_inserter(target_qubits_in_fused_gate),
        [this](::bra::qubit_type const target_qubit) { return this->to_qubit_in_fused_gate_.at(target_qubit); });

      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_yn<paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_yn<cache_aware_paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_yn<fused_gate_iterator> >(phase, std::move(target_qubits_in_fused_gate), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_z<fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_z<paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_z<cache_aware_paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_adj_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_z<fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_z<paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
      cache_aware_paged_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_z<cache_aware_paged_fused_gate_iterator> >(
          phase, to_qubit_in_fused_gate_.at(target_qubit), ket::make_control(to_qubit_in_fused_gate_.at(control_qubit.qubit()))));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void paged_unit_mpi_state::do_multi_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_z<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_z<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_z<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    constexpr auto num_target_qubits = 1u;
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::exponential_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void paged_unit_mpi_state::do_adj_multi_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_z<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_z<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_z<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit), std::move(control_qubits_in_fused_gate)));
      return;
    }

    constexpr auto num_target_qubits = 1u;
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) BOOST_PP_COMMA_IF(n) control_qubits[n]
# define CASE_N(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_exponential_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void paged_unit_mpi_state::do_multi_controlled_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto target_qubits_in_fused_gate = std::vector<qubit_type>{};
      target_qubits_in_fused_gate.reserve(target_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(target_qubits), end(target_qubits), std::back_inserter(target_qubits_in_fused_gate),
        [this](::bra::qubit_type const target_qubit) { return this->to_qubit_in_fused_gate_.at(target_qubit); });

      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_zn<paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_zn<cache_aware_paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_zn<fused_gate_iterator> >(phase, std::move(target_qubits_in_fused_gate), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_multi_controlled_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto target_qubits_in_fused_gate = std::vector<qubit_type>{};
      target_qubits_in_fused_gate.reserve(target_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(target_qubits), end(target_qubits), std::back_inserter(target_qubits_in_fused_gate),
        [this](::bra::qubit_type const target_qubit) { return this->to_qubit_in_fused_gate_.at(target_qubit); });

      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_zn<paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_zn<cache_aware_paged_fused_gate_iterator> >(phase, target_qubits_in_fused_gate, control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_zn<fused_gate_iterator> >(phase, std::move(target_qubits_in_fused_gate), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_target_qubits = target_qubits.size();
    auto const num_control_qubits = control_qubits.size();
    auto const num_operated_qubits = num_target_qubits + num_control_qubits;
    assert(num_target_qubits > 0u);
    assert(num_control_qubits > 0u);
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_multi_controlled_exponential_swap(
    real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_swap<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_swap<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_swap<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_swap<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    if (num_control_qubits + 2u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 2u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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

  void paged_unit_mpi_state::do_adj_multi_controlled_exponential_swap(
    real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      auto control_qubits_in_fused_gate = std::vector<control_qubit_type>{};
      control_qubits_in_fused_gate.reserve(control_qubits.size());
      using std::begin;
      using std::end;
      std::transform(
        begin(control_qubits), end(control_qubits), std::back_inserter(control_qubits_in_fused_gate),
        [this](::bra::control_qubit_type const control_qubit) { return ket::make_control(this->to_qubit_in_fused_gate_.at(control_qubit.qubit())); });

# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
      paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_swap<paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), control_qubits_in_fused_gate));
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR))
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_swap<cache_aware_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), control_qubits_in_fused_gate));
      cache_aware_paged_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_swap<cache_aware_paged_fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), control_qubits_in_fused_gate));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_swap<fused_gate_iterator> >(phase, to_qubit_in_fused_gate_.at(target_qubit1), to_qubit_in_fused_gate_.at(target_qubit2), std::move(control_qubits_in_fused_gate)));
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    if (num_control_qubits + 2u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 2u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

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
