#ifndef BRA_NO_MPI
# include <vector>
# include <array>
# include <iterator>
# include <algorithm>
# include <numeric>
# include <utility>

# include <boost/preprocessor/arithmetic/dec.hpp>
# include <boost/preprocessor/arithmetic/inc.hpp>
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
# include <ket/utility/all_in_state_vector.hpp>
# include <ket/utility/none_in_state_vector.hpp>
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
# include <ket/mpi/page/page_size.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/apply_local_gate.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>

# include <bra/simple_mpi_state.hpp>
# include <bra/state.hpp>
# include <bra/fused_gate.hpp>

# ifndef BRA_MAX_NUM_OPERATED_QUBITS
#   define BRA_MAX_NUM_OPERATED_QUBITS 6
# endif // BRA_MAX_NUM_OPERATED_QUBITS


namespace bra
{
  unsigned int simple_mpi_state::do_num_page_qubits() const
  { return 0u; }

  unsigned int simple_mpi_state::do_num_pages() const
  { return 1u; }

# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)},
      fused_gates_{},
      cache_aware_fused_gates_{}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)},
      fused_gates_{},
      cache_aware_fused_gates_{}
  { }
#   else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, num_elements_in_buffer, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)},
      fused_gates_{},
      cache_aware_fused_gates_{}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, num_elements_in_buffer, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)},
      fused_gates_{},
      cache_aware_fused_gates_{}
  { }
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
# else // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
#   ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)},
      fused_gates_{}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)},
      fused_gates_{}
  { }
#   else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    unsigned int const total_num_qubits,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{total_num_qubits, seed, num_elements_in_buffer, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)},
      fused_gates_{}
  { }

  simple_mpi_state::simple_mpi_state(
    ::bra::state::state_integer_type const initial_integer,
    unsigned int const num_local_qubits,
    std::vector<permutated_qubit_type> const& initial_permutation,
    unsigned int num_threads_per_process,
    ::bra::state::seed_type const seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : ::bra::state{initial_permutation, seed, num_elements_in_buffer, communicator, environment},
      parallel_policy_{num_threads_per_process},
      mpi_policy_{},
      data_{generate_initial_data(num_local_qubits, initial_integer, communicator, environment)},
      fused_gates_{}
  { }
#   endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

  simple_mpi_state::data_type simple_mpi_state::generate_initial_data(
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

  void simple_mpi_state::do_i_gate(qubit_type const qubit)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_ic_gate(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, control_qubit);
  }

  void simple_mpi_state::do_ii_gate(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_in_gate(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      return;

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::identity(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_hadamard(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_hadamard<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_hadamard<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_not_(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_not_<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_not_<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_x<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_x<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xx<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xx<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_pauli_xn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xn<fused_gate_iterator> >(qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_xn<cache_aware_fused_gate_iterator> >(qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::pauli_x(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_y<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_y<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yy<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yy<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_pauli_yn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yn<fused_gate_iterator> >(qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_yn<cache_aware_fused_gate_iterator> >(qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::pauli_y(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_z<fused_gate_iterator> >(control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_z<cache_aware_fused_gate_iterator> >(control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit);
  }

  void simple_mpi_state::do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zz<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zz<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zn<fused_gate_iterator> >(qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_pauli_zn<cache_aware_fused_gate_iterator> >(qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::pauli_z(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_swap(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_swap<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_swap<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_sqrt_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_x<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_adj_sqrt_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_x<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_sqrt_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_y<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_adj_sqrt_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_y<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_sqrt_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_z<fused_gate_iterator> >(control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit);
  }

  void simple_mpi_state::do_adj_sqrt_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_z<fused_gate_iterator> >(control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit);
  }

  void simple_mpi_state::do_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zz<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zz<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_adj_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zz<fused_gate_iterator> >(qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zz<cache_aware_fused_gate_iterator> >(qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit1, qubit2);
  }

  void simple_mpi_state::do_sqrt_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zn<fused_gate_iterator> >(qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::sqrt_pauli_z(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_adj_sqrt_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zn<fused_gate_iterator> >(qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_sqrt_pauli_z(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_u1(real_type const phase, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u1<fused_gate_iterator> >(phase, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u1<cache_aware_fused_gate_iterator> >(phase, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, control_qubit);
  }

  void simple_mpi_state::do_adj_u1(real_type const phase, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u1<fused_gate_iterator> >(phase, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u1<cache_aware_fused_gate_iterator> >(phase, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, control_qubit);
  }

  void simple_mpi_state::do_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u2<fused_gate_iterator> >(phase1, phase2, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, qubit);
  }

  void simple_mpi_state::do_adj_u2(
    real_type const phase1, real_type const phase2, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u2<fused_gate_iterator> >(phase1, phase2, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, qubit);
  }

  void simple_mpi_state::do_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u3<fused_gate_iterator> >(phase1, phase2, phase3, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, qubit);
  }

  void simple_mpi_state::do_adj_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u3<fused_gate_iterator> >(phase1, phase2, phase3, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, qubit);
  }

  void simple_mpi_state::do_phase_shift(
    complex_type const& phase_coefficient, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_phase_shift<fused_gate_iterator> >(phase_coefficient, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, control_qubit);
  }

  void simple_mpi_state::do_adj_phase_shift(
    complex_type const& phase_coefficient, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_phase_shift<fused_gate_iterator> >(phase_coefficient, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, control_qubit);
  }

  void simple_mpi_state::do_x_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_x_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_adj_x_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_x_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_y_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_y_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_adj_y_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_y_rotation_half_pi<fused_gate_iterator> >(qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_x<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_x<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_adj_exponential_pauli_x(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_x<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_x<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_exponential_pauli_xx(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xx<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xx<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_adj_exponential_pauli_xx(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xx<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xx<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::exponential_pauli_x(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_adj_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_exponential_pauli_x(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_y<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_y<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_adj_exponential_pauli_y(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_y<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_y<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_exponential_pauli_yy(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yy<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yy<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_adj_exponential_pauli_yy(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yy<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yy<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::exponential_pauli_y(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_adj_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_exponential_pauli_y(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_z<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_adj_exponential_pauli_z(real_type const phase, qubit_type const qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_z<fused_gate_iterator> >(phase, qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit);
  }

  void simple_mpi_state::do_exponential_pauli_zz(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zz<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zz<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_adj_exponential_pauli_zz(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zz<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zz<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::exponential_pauli_z(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_adj_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zn<fused_gate_iterator> >(phase, qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define QUBITS(z, n, _) , qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_exponential_pauli_z(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef QUBITS
    }
  }

  void simple_mpi_state::do_exponential_swap(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_swap<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_exponential_swap<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_adj_exponential_swap(
    real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_swap<fused_gate_iterator> >(phase, qubit1, qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_exponential_swap<cache_aware_fused_gate_iterator> >(phase, qubit1, qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_swap(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, qubit1, qubit2);
  }

  void simple_mpi_state::do_toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_toffoli<fused_gate_iterator> >(
          target_qubit, control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_toffoli<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::toffoli(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit1, control_qubit2);
  }

  ket::gate::outcome simple_mpi_state::do_projective_measurement(
    qubit_type const qubit, yampi::rank const root)
  {
    return ket::mpi::gate::projective_measurement(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, root, communicator_, environment_, random_number_generator_, qubit);
  }

  void simple_mpi_state::do_expectation_values(yampi::rank const root)
  {
    maybe_expectation_values_
      = ket::mpi::all_spin_expectation_values<typename spins_type::allocator_type>(
          mpi_policy_, parallel_policy_,
          data_, permutation_, total_num_qubits_, buffer_, root, communicator_, environment_);
  }

  void simple_mpi_state::do_measure(yampi::rank const root)
  {
    measured_value_
      = ket::mpi::measure(
          mpi_policy_, ket::utility::policy::make_sequential(), // parallel_policy_,
          data_, random_number_generator_, permutation_, communicator_, environment_);
  }

  void simple_mpi_state::do_generate_events(yampi::rank const root, int const num_events, int const seed)
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

  void simple_mpi_state::do_shor_box(
    state_integer_type const divisor, state_integer_type const base,
    std::vector<qubit_type> const& exponent_qubits,
    std::vector<qubit_type> const& modular_exponentiation_qubits)
  {
    ket::mpi::shor_box(
      mpi_policy_, parallel_policy_,
      data_, base, divisor, exponent_qubits, modular_exponentiation_qubits,
      permutation_, communicator_, environment_);
  }

  void simple_mpi_state::do_begin_fusion()
  { }

  void simple_mpi_state::do_end_fusion()
  {
# if !(!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && defined(KET_USE_ON_CACHE_STATE_VECTOR)))
    assert(fused_gates_.size() == cache_aware_fused_gates_.size());
# endif // !(!defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && defined(KET_USE_ON_CACHE_STATE_VECTOR)))

    // generate fused_control_qubits, fused_ez_qubits, fused_cez_qubits, and fused_qubits from found_qubits_
    auto fused_control_qubits = std::vector< ::bra::control_qubit_type >{};
    fused_control_qubits.reserve(total_num_qubits_);
    auto fused_ez_qubits = std::vector< ::bra::qubit_type >{};
    fused_ez_qubits.reserve(total_num_qubits_);
    auto fused_cez_qubits = std::vector< ::bra::qubit_type >{};
    fused_cez_qubits.reserve(total_num_qubits_);
    auto fused_qubits = std::vector< ::bra::qubit_type >{};
    fused_qubits.reserve(total_num_qubits_);
    for (auto index = ::bra::bit_integer_type{0}; index < total_num_qubits_; ++index)
      switch (found_qubits_[index])
      {
       case ::bra::found_qubit::control_qubit:
        fused_control_qubits.push_back(ket::make_control(ket::make_qubit< ::bra::state_integer_type >(index)));
        break;

       case ::bra::found_qubit::ez_qubit:
        fused_ez_qubits.push_back(ket::make_qubit< ::bra::state_integer_type >(index));
        break;

       case ::bra::found_qubit::cez_qubit:
        fused_cez_qubits.push_back(ket::make_qubit< ::bra::state_integer_type >(index));
        break;

       case ::bra::found_qubit::qubit:
        fused_qubits.push_back(ket::make_qubit< ::bra::state_integer_type >(index));
        break;

       case ::bra::found_qubit::not_found:
        break;
      }

    auto const data_block_size = ket::mpi::utility::policy::data_block_size(mpi_policy_, data_, communicator_, environment_);
    auto const least_permutated_unit_qubit
      = ket::mpi::make_permutated(ket::make_qubit< ::bra::state_integer_type >(
          static_cast< ::bra::bit_integer_type >(ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_block_size))));
    auto const least_permutated_global_qubit
      = ket::mpi::make_permutated(ket::make_qubit< ::bra::state_integer_type >(
          static_cast< ::bra::bit_integer_type >(ket::mpi::utility::policy::num_nonglobal_qubits(mpi_policy_, data_block_size))));

    // generate local_fused_control_qubits and nonlocal_fused_control_qubits by using std::partition
    using std::begin;
    using std::end;
    auto const local_fused_control_qubit_first = begin(fused_control_qubits);
    auto const nonlocal_fused_control_qubit_last = end(fused_control_qubits);
    auto const local_fused_control_qubit_last
      = std::partition(
          local_fused_control_qubit_first, nonlocal_fused_control_qubit_last,
          [this, least_permutated_unit_qubit](::bra::control_qubit_type const control_qubit)
          { return this->permutation_[control_qubit] < least_permutated_unit_qubit; });
    auto const nonlocal_fused_control_qubit_first = local_fused_control_qubit_last;

    // generate local_fused_ez_qubits, unit_fused_ez_qubits, global_fused_unit_qubits, and nonlocal_fused_ez_qubits by using std::partition
    auto const local_fused_ez_qubit_first = begin(fused_ez_qubits);
    auto const global_fused_ez_qubit_last = end(fused_ez_qubits);
    auto const nonlocal_fused_ez_qubit_last = global_fused_ez_qubit_last;
    auto const local_fused_ez_qubit_last
      = std::partition(
          local_fused_ez_qubit_first, global_fused_ez_qubit_last,
          [this, least_permutated_unit_qubit](::bra::qubit_type const qubit)
          { return this->permutation_[qubit] < least_permutated_unit_qubit; });
    auto const unit_fused_ez_qubit_first = local_fused_ez_qubit_last;
    auto const nonlocal_fused_ez_qubit_first = unit_fused_ez_qubit_first;
    auto const unit_fused_ez_qubit_last
      = std::partition(
          unit_fused_ez_qubit_first, global_fused_ez_qubit_last,
          [this, least_permutated_global_qubit](::bra::qubit_type const qubit)
          { return this->permutation_[qubit] < least_permutated_global_qubit; });
    auto const global_fused_ez_qubit_first = unit_fused_ez_qubit_last;

    // generate nonglobal_fused_cez_qubits and global_fused_cez_qubits by using std::partition
    auto const nonglobal_fused_cez_qubit_first = begin(fused_cez_qubits);
    auto const global_fused_cez_qubit_last = end(fused_cez_qubits);
    auto const nonglobal_fused_cez_qubit_last
      = std::partition(
          nonglobal_fused_cez_qubit_first, global_fused_cez_qubit_last,
          [this, least_permutated_global_qubit](::bra::qubit_type const qubit)
          { return this->permutation_[qubit] < least_permutated_global_qubit; });
    auto const global_fused_cez_qubit_first = nonglobal_fused_cez_qubit_last;

    // generate ez_qubit_states and cez_qubit_states
    auto const global_qubit_value = static_cast< ::bra::state_integer_type >(::ket::mpi::utility::policy::global_qubit_value(mpi_policy_, communicator_, environment_));
    auto ez_qubit_states = std::vector< ::bra::fused_gate::cez_qubit_state >(global_fused_ez_qubit_last - global_fused_ez_qubit_first, ::bra::fused_gate::cez_qubit_state::not_global);
    auto cez_qubit_states = std::vector< ::bra::fused_gate::cez_qubit_state >(global_fused_cez_qubit_last - global_fused_cez_qubit_first, ::bra::fused_gate::cez_qubit_state::not_global);
    for (auto iter = global_fused_ez_qubit_first; iter != global_fused_ez_qubit_last; ++iter)
    {
      constexpr auto zero = ::bra::state_integer_type{0u};
      constexpr auto one = ::bra::state_integer_type{1u};
      if ((global_qubit_value bitand (one << (permutation_[*iter] - least_permutated_global_qubit))) == zero)
        ez_qubit_states[iter - global_fused_ez_qubit_first] = ::bra::fused_gate::cez_qubit_state::global_zero;
      else
        ez_qubit_states[iter - global_fused_ez_qubit_first] = ::bra::fused_gate::cez_qubit_state::global_one;
    }
    for (auto iter = global_fused_cez_qubit_first; iter != global_fused_cez_qubit_last; ++iter)
    {
      constexpr auto zero = ::bra::state_integer_type{0u};
      constexpr auto one = ::bra::state_integer_type{1u};
      if ((global_qubit_value bitand (one << (permutation_[*iter] - least_permutated_global_qubit))) == zero)
        cez_qubit_states[iter - global_fused_cez_qubit_first] = ::bra::fused_gate::cez_qubit_state::global_zero;
      else
        cez_qubit_states[iter - global_fused_cez_qubit_first] = ::bra::fused_gate::cez_qubit_state::global_one;
    }

    // modify fused_gate's in fused_gates_ and its variants, and calculate global_phase if needed
    auto new_fused_control_qubits = std::vector< ::bra::control_qubit_type >{};
    new_fused_control_qubits.reserve(total_num_qubits_);
    auto exists_global_phase = false;
    auto global_phase = ::bra::real_type{0};
    for (auto& fused_gate_ptr: fused_gates_)
    {
      fused_gate_ptr->disable_control_qubits(nonlocal_fused_control_qubit_first, nonlocal_fused_control_qubit_last);
      fused_gate_ptr->disable_control_qubits(nonlocal_fused_ez_qubit_first, nonlocal_fused_ez_qubit_last);
      fused_gate_ptr->disable_control_qubits(global_fused_cez_qubit_first, global_fused_cez_qubit_last);

      fused_gate_ptr->modify_cez(global_fused_ez_qubit_first, global_fused_ez_qubit_last, begin(ez_qubit_states));
      fused_gate_ptr->modify_cez(global_fused_cez_qubit_first, global_fused_cez_qubit_last, begin(cez_qubit_states));

      auto const maybe_cqubit_global_phase = fused_gate_ptr->maybe_phase_shiftize_ez(unit_fused_ez_qubit_first, unit_fused_ez_qubit_last);
      if (maybe_cqubit_global_phase)
      {
        new_fused_control_qubits.push_back(maybe_cqubit_global_phase->first);
        exists_global_phase = true;
        global_phase += maybe_cqubit_global_phase->second;
      }
    }
#   if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    for (auto& fused_gate_ptr: cache_aware_fused_gates_)
    {
      fused_gate_ptr->disable_control_qubits(nonlocal_fused_control_qubit_first, nonlocal_fused_control_qubit_last);
      fused_gate_ptr->disable_control_qubits(nonlocal_fused_ez_qubit_first, nonlocal_fused_ez_qubit_last);
      fused_gate_ptr->disable_control_qubits(global_fused_cez_qubit_first, global_fused_cez_qubit_last);

      fused_gate_ptr->modify_cez(global_fused_ez_qubit_first, global_fused_ez_qubit_last, begin(ez_qubit_states));
      fused_gate_ptr->modify_cez(global_fused_cez_qubit_first, global_fused_cez_qubit_last, begin(cez_qubit_states));

      fused_gate_ptr->maybe_phase_shiftize_ez(unit_fused_ez_qubit_first, unit_fused_ez_qubit_last);
    }
#   endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)

    // modify fused_qubits and fused_control_qubits
    std::copy(nonglobal_fused_cez_qubit_first, nonglobal_fused_cez_qubit_last, std::back_inserter(fused_qubits));
    std::copy(local_fused_ez_qubit_first, local_fused_ez_qubit_last, std::back_inserter(fused_qubits));
    std::sort(begin(new_fused_control_qubits), end(new_fused_control_qubits));
    new_fused_control_qubits.resize(std::unique(begin(new_fused_control_qubits), end(new_fused_control_qubits)) - begin(new_fused_control_qubits));
    std::copy(local_fused_control_qubit_first, local_fused_control_qubit_last, std::back_inserter(new_fused_control_qubits));
    fused_control_qubits.swap(new_fused_control_qubits);

    // generate to_qubit_index_in_fused_gates
    auto to_qubit_index_in_fused_gates = std::vector< ::bra::bit_integer_type >(total_num_qubits_);
    std::iota(begin(to_qubit_index_in_fused_gates), end(to_qubit_index_in_fused_gates), ::bra::bit_integer_type{0u});
    auto present_qubit_index = ::bra::bit_integer_type{0u};
    for (auto const fused_qubit: fused_qubits)
      to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(fused_qubit)] = present_qubit_index++;
    for (auto const fused_control_qubit: fused_control_qubits)
      to_qubit_index_in_fused_gates[static_cast< ::bra::bit_integer_type >(fused_control_qubit.qubit())] = present_qubit_index++;

    ket::mpi::utility::logger logger{environment_};

    if (exists_global_phase)
      ::ket::mpi::gate::phase_shift(mpi_policy_, parallel_policy_, data_, permutation_, buffer_, communicator_, environment_, global_phase);

    switch (fused_qubits.size())
    {
# if !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && defined(KET_USE_ON_CACHE_STATE_VECTOR))
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
#     define LOCAL_GATE \
          [this, &to_qubit_index_in_fused_gates](\
            ket::mpi::utility::policy::simple_mpi const mpi_policy,\
            ket::utility::policy::parallel<unsigned int> const parallel_policy,\
            data_type& data, ::bra::data_type& buffer,\
            yampi::communicator const& communicator, yampi::environment const& environment,\
            ::bra::state_integer_type const unit_control_qubit_mask, auto&&... permutated_qubits)\
          {\
            return ::ket::mpi::gate::local::gate(\
              mpi_policy, parallel_policy,\
              data, buffer, communicator, environment, unit_control_qubit_mask,\
              [this, &to_qubit_index_in_fused_gates](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::qubit_type, sizeof...(permutated_qubits) > const& unsorted_fused_qubits,\
                std::array< ::bra::qubit_type, sizeof...(permutated_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
                int const)\
              {\
                for (auto const& gate_ptr: this->fused_gates_)\
                  gate_ptr->call(\
                    first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                    to_qubit_index_in_fused_gates);\
              }, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);\
          }
#   else // KET_USE_BIT_MASKS_EXPLICITLY
#     define LOCAL_GATE \
          [this, &to_qubit_index_in_fused_gates](\
            ket::mpi::utility::policy::simple_mpi const mpi_policy,\
            ket::utility::policy::parallel<unsigned int> const parallel_policy,\
            data_type& data, ::bra::data_type& buffer,\
            yampi::communicator const& communicator, yampi::environment const& environment,\
            ::bra::state_integer_type const unit_control_qubit_mask, auto&&... permutated_qubits)\
          {\
            return ::ket::mpi::gate::local::gate(\
              mpi_policy, parallel_policy,\
              data, buffer, communicator, environment, unit_control_qubit_mask,\
              [this, &to_qubit_index_in_fused_gates](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::state_integer_type, sizeof...(permutated_qubits) > const& qubit_masks,\
                std::array< ::bra::state_integer_type, sizeof...(permutated_qubits) + 1u > const& index_masks,\
                int const)\
              {\
                for (auto const& gate_ptr: this->fused_gates_)\
                  gate_ptr->call(\
                    first, index_wo_qubits, qubit_masks, index_masks,\
                    to_qubit_index_in_fused_gates);\
              }, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);\
          }
#   endif // KET_USE_BIT_MASKS_EXPLICITLY
# else // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && defined(KET_USE_ON_CACHE_STATE_VECTOR))
#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
#     define LOCAL_GATE \
          [this, &to_qubit_index_in_fused_gates](\
            ket::mpi::utility::policy::simple_mpi const mpi_policy,\
            ket::utility::policy::parallel<unsigned int> const parallel_policy,\
            data_type& data, ::bra::data_type& buffer,\
            yampi::communicator const& communicator, yampi::environment const& environment,\
            ::bra::state_integer_type const unit_control_qubit_mask, auto&&... permutated_qubits)\
          {\
            constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};\
            constexpr auto cache_size = ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);\
\
            if (ket::utility::all_in_state_vector(num_on_cache_qubits, permutated_qubits.qubit()...))\
            {\
              if (ket::mpi::page::page_size(mpi_policy, data, communicator, environment) <= cache_size)\
                return ket::mpi::gate::local::nopage::all_on_cache::small::gate(\
                  mpi_policy, parallel_policy,\
                  data, communicator, environment, unit_control_qubit_mask,\
                  [this, &to_qubit_index_in_fused_gates](\
                    auto const first, ::bra::state_integer_type const index_wo_qubits,\
                    std::array< ::bra::qubit_type, sizeof...(permutated_qubits) > const& unsorted_fused_qubits,\
                    std::array< ::bra::qubit_type, sizeof...(permutated_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
                    int const)\
                  {\
                    for (auto const& gate_ptr: this->fused_gates_)\
                      gate_ptr->call(\
                        first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                        to_qubit_index_in_fused_gates);\
                  }, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);\
\
              return ket::mpi::gate::local::nopage::all_on_cache::gate(\
                mpi_policy, parallel_policy,\
                data, communicator, environment, unit_control_qubit_mask,\
                [this, &to_qubit_index_in_fused_gates](\
                  auto const first, ::bra::state_integer_type const index_wo_qubits,\
                  std::array< ::bra::qubit_type, sizeof...(permutated_qubits) > const& unsorted_fused_qubits,\
                  std::array< ::bra::qubit_type, sizeof...(permutated_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
                  int const)\
                {\
                  for (auto const& gate_ptr: this->fused_gates_)\
                    gate_ptr->call(\
                      first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                      to_qubit_index_in_fused_gates);\
                }, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);\
            }\
\
            if (ket::utility::none_in_state_vector(num_on_cache_qubits, permutated_qubits.qubit()...))\
              return ket::mpi::gate::local::nopage::none_on_cache::gate(\
                mpi_policy, parallel_policy,\
                data, communicator, environment, unit_control_qubit_mask,\
                [this, &to_qubit_index_in_fused_gates](\
                  auto const first, ::bra::state_integer_type const index_wo_qubits,\
                  std::array< ::bra::qubit_type, sizeof...(permutated_qubits) > const& unsorted_fused_qubits,\
                  std::array< ::bra::qubit_type, sizeof...(permutated_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
                  int const)\
                {\
                  for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                    gate_ptr->call(\
                      first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                      to_qubit_index_in_fused_gates);\
                }, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);\
\
            return ket::mpi::gate::local::nopage::some_on_cache::gate(\
              mpi_policy, parallel_policy,\
              data, communicator, environment, unit_control_qubit_mask,\
              [this, &to_qubit_index_in_fused_gates](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::qubit_type, sizeof...(permutated_qubits) > const& unsorted_fused_qubits,\
                std::array< ::bra::qubit_type, sizeof...(permutated_qubits) + 1u > const& sorted_fused_qubits_with_sentinel,\
                int const)\
              {\
                for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                  gate_ptr->call(\
                    first, index_wo_qubits, unsorted_fused_qubits, sorted_fused_qubits_with_sentinel,\
                    to_qubit_index_in_fused_gates);\
              }, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);\
          }
#   else // KET_USE_BIT_MASKS_EXPLICITLY
#     define LOCAL_GATE \
          [this, &to_qubit_index_in_fused_gates](\
            ket::mpi::utility::policy::simple_mpi const mpi_policy,\
            ket::utility::policy::parallel<unsigned int> const parallel_policy,\
            data_type& data, ::bra::data_type& buffer,\
            yampi::communicator const& communicator, yampi::environment const& environment,\
            ::bra::state_integer_type const unit_control_qubit_mask, auto&&... permutated_qubits)\
          {\
            constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};\
            constexpr auto cache_size = ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);\
\
            if (ket::utility::all_in_state_vector(num_on_cache_qubits, permutated_qubits.qubit()...))\
            {\
              if (ket::mpi::page::page_size(mpi_policy, data, communicator, environment) <= cache_size)\
                return ket::mpi::gate::local::nopage::all_on_cache::small::gate(\
                  mpi_policy, parallel_policy,\
                  data, communicator, environment, unit_control_qubit_mask,\
                  [this, &to_qubit_index_in_fused_gates](\
                    auto const first, ::bra::state_integer_type const index_wo_qubits,\
                    std::array< ::bra::state_integer_type, sizeof...(permutated_qubits) > const& qubit_masks,\
                    std::array< ::bra::state_integer_type, sizeof...(permutated_qubits) + 1u > const& index_masks,\
                    int const)\
                  {\
                    for (auto const& gate_ptr: this->fused_gates_)\
                      gate_ptr->call(\
                        first, index_wo_qubits, qubit_masks, index_masks,\
                        to_qubit_index_in_fused_gates);\
                  }, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);\
\
              return ket::mpi::gate::local::nopage::all_on_cache::gate(\
                mpi_policy, parallel_policy,\
                data, communicator, environment, unit_control_qubit_mask,\
                [this, &to_qubit_index_in_fused_gates](\
                  auto const first, ::bra::state_integer_type const index_wo_qubits,\
                  std::array< ::bra::state_integer_type, sizeof...(permutated_qubits) > const& qubit_masks,\
                  std::array< ::bra::state_integer_type, sizeof...(permutated_qubits) + 1u > const& index_masks,\
                  int const)\
                {\
                  for (auto const& gate_ptr: this->fused_gates_)\
                    gate_ptr->call(\
                      first, index_wo_qubits, qubit_masks, index_masks,\
                      to_qubit_index_in_fused_gates);\
                }, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);\
            }\
\
            if (ket::utility::none_in_state_vector(num_on_cache_qubits, permutated_qubits.qubit()...))\
              return ket::mpi::gate::local::nopage::none_on_cache::gate(\
                mpi_policy, parallel_policy,\
                data, communicator, environment, unit_control_qubit_mask,\
                [this, &to_qubit_index_in_fused_gates](\
                  auto const first, ::bra::state_integer_type const index_wo_qubits,\
                  std::array< ::bra::state_integer_type, sizeof...(permutated_qubits) > const& qubit_masks,\
                  std::array< ::bra::state_integer_type, sizeof...(permutated_qubits) + 1u > const& index_masks,\
                  int const)\
                {\
                  for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                    gate_ptr->call(\
                      first, index_wo_qubits, qubit_masks, index_masks,\
                      to_qubit_index_in_fused_gates);\
                }, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);\
\
            return ket::mpi::gate::local::nopage::some_on_cache::gate(\
              mpi_policy, parallel_policy,\
              data, communicator, environment, unit_control_qubit_mask,\
              [this, &to_qubit_index_in_fused_gates](\
                auto const first, ::bra::state_integer_type const index_wo_qubits,\
                std::array< ::bra::state_integer_type, sizeof...(permutated_qubits) > const& qubit_masks,\
                std::array< ::bra::state_integer_type, sizeof...(permutated_qubits) + 1u > const& index_masks,\
                int const)\
              {\
                for (auto const& gate_ptr: this->cache_aware_fused_gates_)\
                  gate_ptr->call(\
                    first, index_wo_qubits, qubit_masks, index_masks,\
                    to_qubit_index_in_fused_gates);\
              }, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);\
          }
#   endif // KET_USE_BIT_MASKS_EXPLICITLY
# endif // !defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) || (defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && defined(KET_USE_ON_CACHE_STATE_VECTOR))
# define FUSED_QUBITS(z, n, _) , fused_qubits[n]
# define FUSED_CONTROL_QUBITS(z, n, _) , fused_control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        logger.print(\
          "[start] " + ket::mpi::gate::detail::append_qubits_string(\
            std::string(num_control_qubits, 'C').append("Gate") BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil)),\
          environment_);\
\
        ket::mpi::utility::apply_local_gate(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_,\
LOCAL_GATE BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil));\
\
        logger.print_with_time(\
          "[end] " + ket::mpi::gate::detail::append_qubits_string(\
            std::string(num_control_qubits, 'C').append("Gate") BOOST_PP_REPEAT_ ## z(num_target_qubits, FUSED_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, FUSED_CONTROL_QUBITS, nil)),\
          environment_);\
        break;\

# ifndef BRA_MAX_NUM_FUSED_QUBITS
#   ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
#   else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS 10
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
# endif // BRA_MAX_NUM_FUSED_QUBITS
# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (fused_control_qubits.size())\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(0, BOOST_PP_INC(BOOST_PP_SUB(BRA_MAX_NUM_FUSED_QUBITS, num_target_qubits)), CASE_CN, num_target_qubits)\
      }\
      break;\

     case 0:
      switch (fused_control_qubits.size())
      {
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), CASE_CN, 0)
      }
      break;

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BRA_MAX_NUM_FUSED_QUBITS), CASE_N, nil)
# undef CASE_N
# undef CASE_CN
# undef FUSED_CONTROL_QUBITS
# undef FUSED_QUBITS
# undef LOCAL_GATE
    }

    fused_gates_.clear();
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    cache_aware_fused_gates_.clear();
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
  }

  void simple_mpi_state::do_clear(qubit_type const qubit)
  {
    ket::mpi::gate::clear(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_set(qubit_type const qubit)
  {
    ket::mpi::gate::set(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, qubit);
  }

  void simple_mpi_state::do_controlled_i_gate(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_controlled_ic_gate(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
      return;

    ket::mpi::gate::identity(
      mpi_policy_, parallel_policy_,
      data_, permutation_, buffer_, communicator_, environment_, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_multi_controlled_in_gate(
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::identity(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_multi_controlled_ic_gate(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      return;

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::identity(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_hadamard(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_hadamard<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_hadamard<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::hadamard(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_hadamard(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_hadamard<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_hadamard<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::hadamard(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_not(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_not<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_not<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::not_(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_not(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_not<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_not<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::not_(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_x<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_x<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_pauli_xn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_xn<fused_gate_iterator> >(target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_xn<cache_aware_fused_gate_iterator> >(target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::pauli_x(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_y<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_y<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_pauli_yn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_yn<fused_gate_iterator> >(target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_yn<cache_aware_fused_gate_iterator> >(target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::pauli_y(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_pauli_z(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_z<fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_pauli_z<cache_aware_fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_multi_controlled_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_z<fused_gate_iterator> >(control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_z<cache_aware_fused_gate_iterator> >(control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, num_target_qubits) \
       case num_operated_qubits:\
        ket::mpi::gate::pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_multi_controlled_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_zn<fused_gate_iterator> >(target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_pauli_zn<cache_aware_fused_gate_iterator> >(target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_multi_controlled_swap(
    qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_swap<fused_gate_iterator> >(target_qubit1, target_qubit2, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_swap<cache_aware_fused_gate_iterator> >(target_qubit1, target_qubit2, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    if (num_control_qubits + 2u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 2u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::swap(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit1, target_qubit2 BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_DEC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{2u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_x<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_x<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_x<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::sqrt_pauli_x(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_sqrt_pauli_x(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_x<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_x<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_sqrt_pauli_x(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_y<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_y<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_y<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::sqrt_pauli_y(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_sqrt_pauli_y(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_y<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_y<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_sqrt_pauli_y(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_sqrt_pauli_z(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_z<fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_adj_controlled_sqrt_pauli_z(
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_z<fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(
          control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_sqrt_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_z<fused_gate_iterator> >(control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
       case num_operated_qubits:\
        ket::mpi::gate::sqrt_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_z<fused_gate_iterator> >(control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_z<cache_aware_fused_gate_iterator> >(control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
       case num_operated_qubits:\
        ket::mpi::gate::adj_sqrt_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_multi_controlled_sqrt_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_zn<fused_gate_iterator> >(target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::sqrt_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_sqrt_pauli_zn(
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_zn<fused_gate_iterator> >(target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_sqrt_pauli_zn<cache_aware_fused_gate_iterator> >(target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_sqrt_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_ BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_phase_shift(
    complex_type const& phase_coefficient,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_phase_shift<fused_gate_iterator> >(
          phase_coefficient, control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_phase_shift<cache_aware_fused_gate_iterator> >(
          phase_coefficient, control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_adj_controlled_phase_shift(
    complex_type const& phase_coefficient,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_phase_shift<fused_gate_iterator> >(
          phase_coefficient, control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_phase_shift<cache_aware_fused_gate_iterator> >(
          phase_coefficient, control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift_coeff(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_phase_shift<fused_gate_iterator> >(phase_coefficient, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::phase_shift_coeff(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_phase_shift(
    complex_type const& phase_coefficient,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_phase_shift<fused_gate_iterator> >(phase_coefficient, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_phase_shift<cache_aware_fused_gate_iterator> >(phase_coefficient, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 1u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_phase_shift_coeff(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase_coefficient BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_u1(
    real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u1<fused_gate_iterator> >(
          phase, control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u1<cache_aware_fused_gate_iterator> >(
          phase, control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_adj_controlled_u1(
    real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u1<fused_gate_iterator> >(
          phase, control_qubit1, control_qubit2));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u1<cache_aware_fused_gate_iterator> >(
          phase, control_qubit1, control_qubit2));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, control_qubit1, control_qubit2);
  }

  void simple_mpi_state::do_multi_controlled_u1(
    real_type const phase, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u1<fused_gate_iterator> >(phase, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u1<cache_aware_fused_gate_iterator> >(phase, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::phase_shift(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_u1(
    real_type const phase, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u1<fused_gate_iterator> >(phase, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u1<cache_aware_fused_gate_iterator> >(phase, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_operated_qubits = control_qubits.size();
    assert(num_operated_qubits > 2u);

    if (num_operated_qubits > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_operated_qubits, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_operated_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_operated_qubits, _) \
     case num_operated_qubits:\
      ket::mpi::gate::adj_phase_shift(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_operated_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(3, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u2<fused_gate_iterator> >(
          phase1, phase2, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u2<cache_aware_fused_gate_iterator> >(
          phase1, phase2, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u2<fused_gate_iterator> >(
          phase1, phase2, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u2<cache_aware_fused_gate_iterator> >(
          phase1, phase2, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift2(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u2<fused_gate_iterator> >(phase1, phase2, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::phase_shift2(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_u2(
    real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u2<fused_gate_iterator> >(phase1, phase2, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u2<cache_aware_fused_gate_iterator> >(phase1, phase2, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_phase_shift2(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u3<fused_gate_iterator> >(
          phase1, phase2, phase3, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_u3<cache_aware_fused_gate_iterator> >(
          phase1, phase2, phase3, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u3<fused_gate_iterator> >(
          phase1, phase2, phase3, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_u3<cache_aware_fused_gate_iterator> >(
          phase1, phase2, phase3, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_phase_shift3(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u3<fused_gate_iterator> >(phase1, phase2, phase3, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::phase_shift3(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_u3(
    real_type const phase1, real_type const phase2, real_type const phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u3<fused_gate_iterator> >(phase1, phase2, phase3, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_u3<cache_aware_fused_gate_iterator> >(phase1, phase2, phase3, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_phase_shift3(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase1, phase2, phase3, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_x_rotation_half_pi<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_x_rotation_half_pi<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_x_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_x_rotation_half_pi<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::x_rotation_half_pi(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_x_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_x_rotation_half_pi<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_x_rotation_half_pi<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_x_rotation_half_pi(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_y_rotation_half_pi<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_y_rotation_half_pi<fused_gate_iterator> >(
          target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(
          target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_y_rotation_half_pi(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_y_rotation_half_pi<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::y_rotation_half_pi(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_y_rotation_half_pi(
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_y_rotation_half_pi<fused_gate_iterator> >(target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_y_rotation_half_pi<cache_aware_fused_gate_iterator> >(target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 1u);

    if (num_control_qubits + 1u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 1u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_y_rotation_half_pi(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{1u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_exponential_pauli_x(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_x<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_x<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_exponential_pauli_x(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_x<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_x<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_x(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_xn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::exponential_pauli_x(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_exponential_pauli_xn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_xn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_xn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_exponential_pauli_x(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_exponential_pauli_y(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_y<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_y<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_exponential_pauli_y(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_y<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_y<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_y(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_yn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::exponential_pauli_y(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_exponential_pauli_yn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_yn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_yn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_exponential_pauli_y(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_z<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_adj_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_z<fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(
        std::make_unique< ::bra::fused_gate::fused_adj_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(
          phase, target_qubit, control_qubit));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
    }
    else
      ket::mpi::gate::adj_exponential_pauli_z(
        mpi_policy_, parallel_policy_,
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit, control_qubit);
  }

  void simple_mpi_state::do_multi_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_z<fused_gate_iterator> >(phase, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
       case num_control_qubits:\
        ket::mpi::gate::exponential_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_exponential_pauli_z(
    real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_z<fused_gate_iterator> >(phase, target_qubit, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_z<cache_aware_fused_gate_iterator> >(phase, target_qubit, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
       case num_control_qubits:\
        ket::mpi::gate::adj_exponential_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

BOOST_PP_REPEAT_FROM_TO(2, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_multi_controlled_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_zn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::exponential_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_exponential_pauli_zn(
    real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_zn<fused_gate_iterator> >(phase, target_qubits, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_pauli_zn<cache_aware_fused_gate_iterator> >(phase, target_qubits, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
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
# define TARGET_QUBITS(z, n, _) , target_qubits[n]
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_CN(z, num_control_qubits, num_target_qubits) \
       case num_control_qubits:\
        ket::mpi::gate::adj_exponential_pauli_z(\
          mpi_policy_, parallel_policy_,\
          data_, permutation_, buffer_, communicator_, environment_, phase BOOST_PP_REPEAT_ ## z(num_target_qubits, TARGET_QUBITS, nil) BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
        break;\

# define CASE_N(z, num_target_qubits, _) \
     case num_target_qubits:\
      switch (num_control_qubits)\
      {\
BOOST_PP_REPEAT_FROM_TO_ ## z(BOOST_PP_IIF(BOOST_PP_EQUAL(num_target_qubits, 1), 2, 1), BOOST_PP_SUB(BOOST_PP_INC(BRA_MAX_NUM_OPERATED_QUBITS), num_target_qubits), CASE_CN, num_target_qubits)\
       default:\
        throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};\
      }\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BRA_MAX_NUM_OPERATED_QUBITS, CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_operated_qubits, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CASE_CN
# undef CONTROL_QUBITS
# undef TARGET_QUBITS
    }
  }

  void simple_mpi_state::do_multi_controlled_exponential_swap(
    real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_swap<fused_gate_iterator> >(phase, target_qubit1, target_qubit2, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_multi_controlled_exponential_swap<cache_aware_fused_gate_iterator> >(phase, target_qubit1, target_qubit2, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    if (num_control_qubits + 2u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 2u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::exponential_swap(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2 BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
      break;\

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_DEC(BRA_MAX_NUM_OPERATED_QUBITS), CASE_N, nil)
     default:
      throw bra::too_many_operated_qubits_error{num_control_qubits + std::size_t{2u}, BRA_MAX_NUM_OPERATED_QUBITS};
# undef CASE_N
# undef CONTROL_QUBITS
    }
  }

  void simple_mpi_state::do_adj_multi_controlled_exponential_swap(
    real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_swap<fused_gate_iterator> >(phase, target_qubit1, target_qubit2, control_qubits));
# if defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      cache_aware_fused_gates_.push_back(std::make_unique< ::bra::fused_gate::fused_adj_multi_controlled_exponential_swap<cache_aware_fused_gate_iterator> >(phase, target_qubit1, target_qubit2, control_qubits));
# endif // defined(KET_ENABLE_CACHE_AWARE_GATE_FUNCTION) && !defined(KET_USE_ON_CACHE_STATE_VECTOR)
      return;
    }

    auto const num_control_qubits = control_qubits.size();
    assert(num_control_qubits > 0u);

    if (num_control_qubits + 2u > ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_))
      throw ::bra::too_many_operated_qubits_error{num_control_qubits + 2u, ket::mpi::utility::policy::num_local_qubits(mpi_policy_, data_, communicator_, environment_)};

    switch (num_control_qubits)
    {
# define CONTROL_QUBITS(z, n, _) , control_qubits[n]
# define CASE_N(z, num_control_qubits, _) \
     case num_control_qubits:\
      ket::mpi::gate::adj_exponential_swap(\
        mpi_policy_, parallel_policy_,\
        data_, permutation_, buffer_, communicator_, environment_, phase, target_qubit1, target_qubit2 BOOST_PP_REPEAT_ ## z(num_control_qubits, CONTROL_QUBITS, nil));\
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
