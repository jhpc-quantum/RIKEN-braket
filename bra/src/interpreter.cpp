#include <istream>
#include <string>
#include <vector>
#include <tuple>
#include <utility>
#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
#if __cplusplus >= 201703L
# include <type_traits>
#else
# include <boost/type_traits/is_nothrow_swappable.hpp>
#endif

#include <boost/lexical_cast.hpp>

#include <boost/range/empty.hpp>
#include <boost/range/size.hpp>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <boost/preprocessor/arithmetic/dec.hpp>

#ifndef BRA_NO_MPI
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
#endif // BRA_NO_MPI

#include <ket/qubit.hpp>
#include <ket/control.hpp>
#include <ket/utility/integer_log2.hpp>
#include <ket/utility/integer_exp2.hpp>

#include <bra/types.hpp>
#include <bra/interpreter.hpp>
#include <bra/state.hpp>
#include <bra/utility/to_integer.hpp>
#include <bra/gate/gate.hpp>
#include <bra/gate/var_op.hpp>
#include <bra/gate/let_op.hpp>
#include <bra/gate/jump_op.hpp>
#include <bra/gate/jumpif_op.hpp>
#include <bra/gate/i_gate.hpp>
#include <bra/gate/ic_gate.hpp>
#include <bra/gate/ii_gate.hpp>
#include <bra/gate/in_gate.hpp>
#include <bra/gate/hadamard.hpp>
#include <bra/gate/not_.hpp>
#include <bra/gate/pauli_x.hpp>
#include <bra/gate/pauli_xx.hpp>
#include <bra/gate/pauli_xn.hpp>
#include <bra/gate/pauli_y.hpp>
#include <bra/gate/pauli_yy.hpp>
#include <bra/gate/pauli_yn.hpp>
#include <bra/gate/pauli_z.hpp>
#include <bra/gate/pauli_zz.hpp>
#include <bra/gate/pauli_zn.hpp>
#include <bra/gate/swap.hpp>
#include <bra/gate/sqrt_pauli_x.hpp>
#include <bra/gate/adj_sqrt_pauli_x.hpp>
#include <bra/gate/sqrt_pauli_y.hpp>
#include <bra/gate/adj_sqrt_pauli_y.hpp>
#include <bra/gate/sqrt_pauli_z.hpp>
#include <bra/gate/adj_sqrt_pauli_z.hpp>
#include <bra/gate/sqrt_pauli_zz.hpp>
#include <bra/gate/adj_sqrt_pauli_zz.hpp>
#include <bra/gate/sqrt_pauli_zn.hpp>
#include <bra/gate/adj_sqrt_pauli_zn.hpp>
#include <bra/gate/s_gate.hpp>
#include <bra/gate/adj_s_gate.hpp>
#include <bra/gate/t_gate.hpp>
#include <bra/gate/adj_t_gate.hpp>
#include <bra/gate/u1.hpp>
#include <bra/gate/adj_u1.hpp>
#include <bra/gate/u2.hpp>
#include <bra/gate/adj_u2.hpp>
#include <bra/gate/u3.hpp>
#include <bra/gate/adj_u3.hpp>
#include <bra/gate/phase_shift.hpp>
#include <bra/gate/adj_phase_shift.hpp>
#include <bra/gate/x_rotation_half_pi.hpp>
#include <bra/gate/adj_x_rotation_half_pi.hpp>
#include <bra/gate/y_rotation_half_pi.hpp>
#include <bra/gate/adj_y_rotation_half_pi.hpp>
#include <bra/gate/controlled_phase_shift.hpp>
#include <bra/gate/adj_controlled_phase_shift.hpp>
#include <bra/gate/exponential_pauli_x.hpp>
#include <bra/gate/adj_exponential_pauli_x.hpp>
#include <bra/gate/exponential_pauli_xx.hpp>
#include <bra/gate/adj_exponential_pauli_xx.hpp>
#include <bra/gate/exponential_pauli_xn.hpp>
#include <bra/gate/adj_exponential_pauli_xn.hpp>
#include <bra/gate/exponential_pauli_y.hpp>
#include <bra/gate/adj_exponential_pauli_y.hpp>
#include <bra/gate/exponential_pauli_yy.hpp>
#include <bra/gate/adj_exponential_pauli_yy.hpp>
#include <bra/gate/exponential_pauli_yn.hpp>
#include <bra/gate/adj_exponential_pauli_yn.hpp>
#include <bra/gate/exponential_pauli_z.hpp>
#include <bra/gate/adj_exponential_pauli_z.hpp>
#include <bra/gate/exponential_pauli_zz.hpp>
#include <bra/gate/adj_exponential_pauli_zz.hpp>
#include <bra/gate/exponential_pauli_zn.hpp>
#include <bra/gate/adj_exponential_pauli_zn.hpp>
#include <bra/gate/exponential_swap.hpp>
#include <bra/gate/adj_exponential_swap.hpp>
#include <bra/gate/toffoli.hpp>
#include <bra/gate/projective_measurement.hpp>
#include <bra/gate/amplitudes.hpp>
#include <bra/gate/measurement.hpp>
#include <bra/gate/generate_events.hpp>
#include <bra/gate/shor_box.hpp>
#include <bra/gate/begin_fusion.hpp>
#include <bra/gate/end_fusion.hpp>
#include <bra/gate/clear.hpp>
#include <bra/gate/set.hpp>
#include <bra/gate/depolarizing_channel.hpp>
#include <bra/gate/exit.hpp>
#include <bra/gate/controlled_i_gate.hpp>
#include <bra/gate/controlled_ic_gate.hpp>
#include <bra/gate/multi_controlled_in_gate.hpp>
#include <bra/gate/multi_controlled_ic_gate.hpp>
#include <bra/gate/controlled_hadamard.hpp>
#include <bra/gate/multi_controlled_hadamard.hpp>
#include <bra/gate/controlled_not.hpp>
#include <bra/gate/multi_controlled_not.hpp>
#include <bra/gate/controlled_pauli_x.hpp>
#include <bra/gate/multi_controlled_pauli_xn.hpp>
#include <bra/gate/controlled_pauli_y.hpp>
#include <bra/gate/multi_controlled_pauli_yn.hpp>
#include <bra/gate/controlled_pauli_z.hpp>
#include <bra/gate/multi_controlled_pauli_zn.hpp>
#include <bra/gate/multi_controlled_pauli_z.hpp>
#include <bra/gate/multi_controlled_swap.hpp>
#include <bra/gate/controlled_sqrt_pauli_x.hpp>
#include <bra/gate/adj_controlled_sqrt_pauli_x.hpp>
#include <bra/gate/multi_controlled_sqrt_pauli_x.hpp>
#include <bra/gate/adj_multi_controlled_sqrt_pauli_x.hpp>
#include <bra/gate/controlled_sqrt_pauli_y.hpp>
#include <bra/gate/adj_controlled_sqrt_pauli_y.hpp>
#include <bra/gate/multi_controlled_sqrt_pauli_y.hpp>
#include <bra/gate/adj_multi_controlled_sqrt_pauli_y.hpp>
#include <bra/gate/controlled_sqrt_pauli_z.hpp>
#include <bra/gate/adj_controlled_sqrt_pauli_z.hpp>
#include <bra/gate/multi_controlled_sqrt_pauli_zn.hpp>
#include <bra/gate/adj_multi_controlled_sqrt_pauli_zn.hpp>
#include <bra/gate/multi_controlled_sqrt_pauli_z.hpp>
#include <bra/gate/adj_multi_controlled_sqrt_pauli_z.hpp>
#include <bra/gate/controlled_s_gate.hpp>
#include <bra/gate/multi_controlled_s_gate.hpp>
#include <bra/gate/adj_controlled_s_gate.hpp>
#include <bra/gate/adj_multi_controlled_s_gate.hpp>
#include <bra/gate/controlled_t_gate.hpp>
#include <bra/gate/multi_controlled_t_gate.hpp>
#include <bra/gate/adj_controlled_t_gate.hpp>
#include <bra/gate/adj_multi_controlled_t_gate.hpp>
#include <bra/gate/controlled_u1.hpp>
#include <bra/gate/adj_controlled_u1.hpp>
#include <bra/gate/multi_controlled_u1.hpp>
#include <bra/gate/adj_multi_controlled_u1.hpp>
#include <bra/gate/controlled_u2.hpp>
#include <bra/gate/adj_controlled_u2.hpp>
#include <bra/gate/multi_controlled_u2.hpp>
#include <bra/gate/adj_multi_controlled_u2.hpp>
#include <bra/gate/controlled_u3.hpp>
#include <bra/gate/adj_controlled_u3.hpp>
#include <bra/gate/multi_controlled_u3.hpp>
#include <bra/gate/adj_multi_controlled_u3.hpp>
#include <bra/gate/controlled_phase_shift_.hpp>
#include <bra/gate/adj_controlled_phase_shift_.hpp>
#include <bra/gate/multi_controlled_phase_shift.hpp>
#include <bra/gate/adj_multi_controlled_phase_shift.hpp>
#include <bra/gate/controlled_x_rotation_half_pi.hpp>
#include <bra/gate/multi_controlled_x_rotation_half_pi.hpp>
#include <bra/gate/adj_controlled_x_rotation_half_pi.hpp>
#include <bra/gate/adj_multi_controlled_x_rotation_half_pi.hpp>
#include <bra/gate/controlled_y_rotation_half_pi.hpp>
#include <bra/gate/multi_controlled_y_rotation_half_pi.hpp>
#include <bra/gate/adj_controlled_y_rotation_half_pi.hpp>
#include <bra/gate/adj_multi_controlled_y_rotation_half_pi.hpp>
#include <bra/gate/controlled_exponential_pauli_x.hpp>
#include <bra/gate/adj_controlled_exponential_pauli_x.hpp>
#include <bra/gate/multi_controlled_exponential_pauli_xn.hpp>
#include <bra/gate/adj_multi_controlled_exponential_pauli_xn.hpp>
#include <bra/gate/controlled_exponential_pauli_y.hpp>
#include <bra/gate/adj_controlled_exponential_pauli_y.hpp>
#include <bra/gate/multi_controlled_exponential_pauli_yn.hpp>
#include <bra/gate/adj_multi_controlled_exponential_pauli_yn.hpp>
#include <bra/gate/controlled_exponential_pauli_z.hpp>
#include <bra/gate/adj_controlled_exponential_pauli_z.hpp>
#include <bra/gate/multi_controlled_exponential_pauli_z.hpp>
#include <bra/gate/adj_multi_controlled_exponential_pauli_z.hpp>
#include <bra/gate/multi_controlled_exponential_pauli_zn.hpp>
#include <bra/gate/adj_multi_controlled_exponential_pauli_zn.hpp>
#include <bra/gate/multi_controlled_exponential_swap.hpp>
#include <bra/gate/adj_multi_controlled_exponential_swap.hpp>

# if __cplusplus >= 201703L
#   define BRA_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define BRA_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace bra
{
  unsupported_mnemonic_error::unsupported_mnemonic_error(std::string const& mnemonic)
    : std::runtime_error{(mnemonic + " is not supported").c_str()}
  { }

  wrong_mnemonics_error::wrong_mnemonics_error(::bra::interpreter::columns_type const& columns)
    : std::runtime_error{generate_what_string(columns).c_str()}
  { }

  std::string wrong_mnemonics_error::generate_what_string(::bra::interpreter::columns_type const& columns)
  {
    auto result = std::string{};

    auto const last = columns.end();
    for (auto iter = columns.begin(); iter != last; ++iter)
    {
      result += *iter;
      result += " ";
    }

    return result;
  }

#ifndef BRA_NO_MPI
  wrong_mpi_communicator_size_error::wrong_mpi_communicator_size_error()
    : std::runtime_error{"communicator size is wrong"}
  { }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  interpreter::interpreter()
    : circuits_(1u), label_maps_(1u), num_qubits_{}, num_lqubits_{}, num_uqubits_{}, num_processes_per_unit_{1u},
      initial_state_value_{}, initial_permutation_{}, root_{}, circuit_index_{0}, is_in_circuit_{false}
  { }
#else // BRA_NO_MPI
  interpreter::interpreter()
    : circuits_(1u), label_maps_(1u), num_qubits_{},
      initial_state_value_{}, circuit_index_{0}, is_in_circuit_{false}
  { }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  interpreter::interpreter(
    std::istream& input_stream,
    ::bra::bit_integer_type num_uqubits, unsigned int num_processes_per_unit,
    yampi::environment const& environment,
    yampi::rank const root, yampi::communicator const& total_communicator,
    size_type const num_reserved_gates)
    : circuits_(1u), label_maps_(1u), num_qubits_{}, num_lqubits_{},
      num_uqubits_{num_uqubits}, num_processes_per_unit_{num_processes_per_unit},
      initial_state_value_{}, initial_permutation_{}, root_{root}, circuit_index_{0}, is_in_circuit_{false}
  {
    assert(num_processes_per_unit >= 1u);
    invoke(input_stream, environment, total_communicator, num_reserved_gates);
  }
#else // BRA_NO_MPI
  interpreter::interpreter(std::istream& input_stream)
    : circuits_(1u), label_maps_(1u), num_qubits_{},
      initial_state_value_{}, circuit_index_{0}, is_in_circuit_{false}
  { invoke(input_stream, size_type{0u}); }

  interpreter::interpreter(std::istream& input_stream, size_type const num_reserved_gates)
    : circuits_(1u), label_maps_(1u), num_qubits_{},
      initial_state_value_{}, circuit_index_{0}, is_in_circuit_{false}
  { invoke(input_stream, num_reserved_gates); }
#endif // BRA_NO_MPI

  bool interpreter::operator==(interpreter const& other) const
  {
#ifndef BRA_NO_MPI
    return circuits_ == other.circuits_
      and label_maps_ == other.label_maps_
      and num_qubits_ == other.num_qubits_
      and num_lqubits_ == other.num_lqubits_
      and num_uqubits_ == other.num_uqubits_
      and num_processes_per_unit_ == other.num_processes_per_unit_
      and initial_state_value_ == other.initial_state_value_
      and initial_permutation_ == other.initial_permutation_
      and root_ == other.root_
      and circuit_index_ == other.circuit_index_
      and is_in_circuit_ == other.is_in_circuit_;
#else // BRA_NO_MPI
    return circuits_ == other.circuits_
      and label_maps_ == other.label_maps_
      and num_qubits_ == other.num_qubits_
      and initial_state_value_ == other.initial_state_value_
      and circuit_index_ == other.circuit_index_
      and is_in_circuit_ == other.is_in_circuit_;
#endif // BRA_NO_MPI
  }

#ifndef BRA_NO_MPI
  void interpreter::num_qubits(
    ::bra::bit_integer_type const new_num_qubits,
    yampi::communicator const& total_communicator, yampi::environment const& environment)
  {
    auto const num_gqubits
      = ket::utility::integer_log2< ::bra::bit_integer_type >(
          (total_communicator.size(environment) / circuits_.size()) / num_processes_per_unit_);
    set_num_qubits_params(new_num_qubits - num_gqubits - num_uqubits_, num_gqubits, total_communicator, environment);
  }

  void interpreter::num_lqubits(
    ::bra::bit_integer_type const new_num_lqubits,
    yampi::communicator const& total_communicator, yampi::environment const& environment)
  {
    set_num_qubits_params(
      new_num_lqubits,
      ket::utility::integer_log2< ::bra::bit_integer_type >(
        (total_communicator.size(environment) / circuits_.size()) / num_processes_per_unit_),
      total_communicator, environment);
  }
#else // BRA_NO_MPI
  void interpreter::num_qubits(::bra::bit_integer_type const new_num_qubits)
  { set_num_qubits_params(new_num_qubits); }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  void interpreter::set_num_qubits_params(
    ::bra::bit_integer_type const new_num_lqubits, ::bra::bit_integer_type const num_gqubits,
    yampi::communicator const& total_communicator, yampi::environment const& environment)
  {
    if (ket::utility::integer_exp2< ::bra::bit_integer_type >(num_gqubits) * num_processes_per_unit_ * circuits_.size()
        != static_cast< ::bra::bit_integer_type >(total_communicator.size(environment)))
      throw wrong_mpi_communicator_size_error{};

    num_lqubits_ = new_num_lqubits;
    num_qubits_ = new_num_lqubits + num_uqubits_ + num_gqubits;

    initial_permutation_.clear();
    initial_permutation_.reserve(num_qubits_);
    for (auto bit = ::bra::bit_integer_type{0u}; bit < num_qubits_; ++bit)
      initial_permutation_.push_back(::bra::permutated_qubit_type{bit});
  }
#else // BRA_NO_MPI
  void interpreter::set_num_qubits_params(::bra::bit_integer_type const new_num_qubits)
  {
    num_qubits_ = new_num_qubits;
  }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  void interpreter::invoke(
    std::istream& input_stream, yampi::environment const& environment,
    yampi::communicator const& total_communicator, size_type const num_reserved_gates)
#else // BRA_NO_MPI
  void interpreter::invoke(std::istream& input_stream, size_type const num_reserved_gates)
#endif // BRA_NO_MPI
  {
    for (auto& circuit: circuits_)
    {
      circuit.clear();
      circuit.reserve(num_reserved_gates);
    }
    for (auto& label_map: label_maps_)
      label_map.clear();

    auto line = std::string{};
    auto columns = columns_type{};
    columns.reserve(10u);

    while (std::getline(input_stream, line))
    {
      if (line.empty())
        continue;

      line.erase(std::find(line.begin(), line.end(), '!'), line.end());
      boost::algorithm::trim(line);
      if (line.empty())
        continue;

      boost::algorithm::split(
        columns, line, boost::algorithm::is_space(),
        boost::algorithm::token_compress_on);

      if (boost::empty(columns))
        continue;

      boost::algorithm::to_upper(columns.front());
      auto const& mnemonic = columns.front();
      using std::begin;
      using std::end;
      if (mnemonic == "CIRCUITS")
      {
        circuits_.resize(read_num_circuits(columns));
        label_maps_.resize(read_num_circuits(columns));
        for (auto& circuit: circuits_)
          circuit.reserve(num_reserved_gates);
      }
      else if (mnemonic == "QUBITS")
      {
#ifndef BRA_NO_MPI
        num_qubits(
          static_cast< ::bra::bit_integer_type >(read_num_qubits(columns)),
          total_communicator, environment);
#else // BRA_NO_MPI
        num_qubits(
          static_cast< ::bra::bit_integer_type >(read_num_qubits(columns)));
#endif // BRA_NO_MPI
      }
      else if (mnemonic == "INITIAL") // INITIAL STATE
        initial_state_value_
          = static_cast< ::bra::state_integer_type >(read_initial_state_value(columns));
      else if (mnemonic == "MPIPROCESSES")
      {
        read_num_mpi_processes(columns);
        // ignore this statement
      }
      else if (mnemonic == "MPISWAPBUFFER")
      {
        read_mpi_buffer_size(columns);
        // ignore this statement
      }
      else if (mnemonic == "BIT") // BIT ASSIGNMENT
      {
        if (boost::size(columns) <= 1u)
          throw wrong_mnemonics_error{columns};
        boost::algorithm::to_upper(columns[1u]);

        auto const statement = read_bit_statement(columns);

        if (statement == ::bra::bit_statement::assignment)
        {
#ifndef BRA_NO_MPI
          initial_permutation_ = read_initial_permutation(columns);
#endif
        }
      }
      else if (mnemonic == "PERMUTATION")
        throw unsupported_mnemonic_error{mnemonic};
      else if (mnemonic == "RANDOM") // RANDOM PERMUTATION
        throw unsupported_mnemonic_error{mnemonic};
      else if (mnemonic == "VAR")
        add_var(columns);
      else if (mnemonic == "LET")
        add_let(columns);
      else if (mnemonic.front() == '@')
        add_label(columns, mnemonic);
      else if (mnemonic == "JUMP")
        add_jump(columns);
      else if (mnemonic == "JUMPIF")
        add_jumpif(columns);
      else if (mnemonic == "I")
        add_i(columns);
      else if (mnemonic == "IC")
        add_ic(columns);
      else if (mnemonic == "II")
        add_ii(columns);
      else if (mnemonic.size() >= 3u
               and std::all_of(begin(mnemonic), end(mnemonic), [](char const character) { return character == 'I'; }))
        add_is(columns, mnemonic);
      else if (mnemonic.size() >= 2u and mnemonic.front() == 'I')
        add_in(columns, mnemonic);
      else if (mnemonic == "H")
        add_h(columns);
      else if (mnemonic == "NOT")
        add_not(columns);
      else if (mnemonic == "X")
        add_x(columns);
      else if (mnemonic == "XX")
        add_xx(columns);
      else if (mnemonic.size() >= 3u
               and std::all_of(begin(mnemonic), end(mnemonic), [](char const character) { return character == 'X'; }))
        add_xs(columns, mnemonic);
      else if (mnemonic.size() >= 2u and mnemonic.front() == 'X')
        add_xn(columns, mnemonic);
      else if (mnemonic == "Y")
        add_y(columns);
      else if (mnemonic == "YY")
        add_yy(columns);
      else if (mnemonic.size() >= 3u
               and std::all_of(begin(mnemonic), end(mnemonic), [](char const character) { return character == 'Y'; }))
        add_ys(columns, mnemonic);
      else if (mnemonic.size() >= 2u and mnemonic.front() == 'Y')
        add_yn(columns, mnemonic);
      else if (mnemonic == "Z")
        add_z(columns);
      else if (mnemonic == "ZZ")
        add_zz(columns);
      else if (mnemonic.size() >= 3u
               and std::all_of(begin(mnemonic), end(mnemonic), [](char const character) { return character == 'Z'; }))
        add_zs(columns, mnemonic);
      else if (mnemonic.size() >= 2u and mnemonic.front() == 'Z')
        add_zn(columns, mnemonic);
      else if (mnemonic == "SWAP")
        add_swap(columns);
      else if (mnemonic == "S")
        add_s(columns);
      else if (mnemonic == "S+")
        add_adj_s(columns);
      else if (mnemonic == "T")
        add_t(columns);
      else if (mnemonic == "T+")
        add_adj_t(columns);
      else if (mnemonic == "U1")
        add_u1(columns);
      else if (mnemonic == "U1+")
        add_adj_u1(columns);
      else if (mnemonic == "U2")
        add_u2(columns);
      else if (mnemonic == "U2+")
        add_adj_u2(columns);
      else if (mnemonic == "U3")
        add_u3(columns);
      else if (mnemonic == "U3+")
        add_adj_u3(columns);
      else if (mnemonic == "R")
        add_r(columns);
      else if (mnemonic == "R+")
        add_adj_r(columns);
      else if (mnemonic == "+X")
        add_rotx(columns);
      else if (mnemonic == "-X")
        add_adj_rotx(columns);
      else if (mnemonic == "+Y")
        add_roty(columns);
      else if (mnemonic == "-Y")
        add_adj_roty(columns);
      else if (mnemonic == "U")
        add_u(columns);
      else if (mnemonic == "U+")
        add_adj_u(columns);
      else if (mnemonic == "EXIT")
      {
        if (boost::size(columns) != 1u)
          throw wrong_mnemonics_error{columns};

#ifndef BRA_NO_MPI
        circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exit >(root_));
#else // BRA_NO_MPI
        circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exit >());
#endif // BRA_NO_MPI
        break;
      }
      else if (mnemonic == "EX")
        add_ex(columns);
      else if (mnemonic == "EX+")
        add_adj_ex(columns);
      else if (mnemonic == "EXX")
        add_exx(columns);
      else if (mnemonic == "EXX+")
        add_adj_exx(columns);
      else if (mnemonic.size() >= 4u and mnemonic.front() == 'E'
               and std::all_of(next(begin(mnemonic)), end(mnemonic), [](char const character) { return character == 'X'; }))
        add_exs(columns, mnemonic);
      else if (mnemonic.size() >= 5u and mnemonic.front() == 'E'
               and std::all_of(next(begin(mnemonic)), std::prev(end(mnemonic)), [](char const character) { return character == 'X'; })
               and mnemonic.back() == '+')
        add_adj_exs(columns, mnemonic);
      else if (mnemonic.size() >= 4u and mnemonic[0] == 'E' and mnemonic[1] == 'X' and mnemonic.back() == '+')
        add_adj_exn(columns, mnemonic);
      else if (mnemonic.size() >= 3u and mnemonic[0] == 'E' and mnemonic[1] == 'X')
        add_exn(columns, mnemonic);
      else if (mnemonic == "EY")
        add_ey(columns);
      else if (mnemonic == "EY+")
        add_adj_ey(columns);
      else if (mnemonic == "EYY")
        add_eyy(columns);
      else if (mnemonic == "EYY+")
        add_adj_eyy(columns);
      else if (mnemonic.size() >= 4u and mnemonic.front() == 'E'
               and std::all_of(std::next(begin(mnemonic)), end(mnemonic), [](char const character) { return character == 'Y'; }))
        add_eys(columns, mnemonic);
      else if (mnemonic.size() >= 5u and mnemonic.front() == 'E'
               and std::all_of(std::next(begin(mnemonic)), std::prev(end(mnemonic)), [](char const character) { return character == 'Y'; })
               and mnemonic.back() == '+')
        add_adj_eys(columns, mnemonic);
      else if (mnemonic.size() >= 4u and mnemonic[0] == 'E' and mnemonic[1] == 'Y' and mnemonic.back() == '+')
        add_adj_eyn(columns, mnemonic);
      else if (mnemonic.size() >= 3u and mnemonic[0] == 'E' and mnemonic[1] == 'Y')
        add_eyn(columns, mnemonic);
      else if (mnemonic == "EZ")
        add_ez(columns);
      else if (mnemonic == "EZ+")
        add_adj_ez(columns);
      else if (mnemonic == "EZZ")
        add_ezz(columns);
      else if (mnemonic == "EZZ+")
        add_adj_ezz(columns);
      else if (mnemonic.size() >= 4u and mnemonic.front() == 'E'
               and std::all_of(std::next(begin(mnemonic)), end(mnemonic), [](char const character) { return character == 'Z'; }))
        add_ezs(columns, mnemonic);
      else if (mnemonic.size() >= 5u and mnemonic.front() == 'E'
               and std::all_of(std::next(begin(mnemonic)), std::prev(end(mnemonic)), [](char const character) { return character == 'Z'; })
               and mnemonic.back() == '+')
        add_adj_ezs(columns, mnemonic);
      else if (mnemonic.size() >= 4u and mnemonic[0] == 'E' and mnemonic[1] == 'Z' and mnemonic.back() == '+')
        add_adj_ezn(columns, mnemonic);
      else if (mnemonic.size() >= 3u and mnemonic[0] == 'E' and mnemonic[1] == 'Z')
        add_ezn(columns, mnemonic);
      else if (mnemonic == "ESWAP")
        add_eswap(columns);
      else if (mnemonic == "ESWAP+")
        add_adj_eswap(columns);
      else if (mnemonic == "TOFFOLI")
        add_toffoli(columns);
      else if (mnemonic == "M")
        add_m(columns);
      else if (mnemonic == "SHORBOX")
        add_shor_box(columns);
      else if (mnemonic == "BEGIN") // BEGIN MEASUREMENT/LEARNING MACHINE/FUSION/CIRCUIT
      {
        if (columns.size() <= 1u)
          throw wrong_mnemonics_error{columns};

        std::for_each(
          std::next(begin(columns)), end(columns),
          [](std::string& str) { boost::algorithm::to_upper(str); });

        auto const statement = read_begin_statement(columns);

        if (statement == ::bra::begin_statement::measurement)
        {
#ifndef BRA_NO_MPI
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::measurement >(root_));
#else // BRA_NO_MPI
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::measurement >());
#endif // BRA_NO_MPI
        }
        else if (statement == ::bra::begin_statement::fusion)
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::begin_fusion >());
        else if (statement == ::bra::begin_statement::circuit)
        {
          if (columns.size() != 3u)
            throw wrong_mnemonics_error{columns};

          if (is_in_circuit_)
            throw wrong_mnemonics_error{columns};

          using std::begin;
          auto iter = begin(columns);
          ++iter;
          circuit_index_ = boost::lexical_cast<int>(*++iter);
          is_in_circuit_ = true;
          if (circuit_index_ < 0 or circuit_index_ >= static_cast<int>(circuits_.size()))
            throw wrong_mnemonics_error{columns};
        }
        else if (statement == ::bra::begin_statement::learning_machine)
          throw unsupported_mnemonic_error{mnemonic};
        else
          throw unsupported_mnemonic_error{mnemonic};
      }
      else if (mnemonic == "DO") // DO MEASUREMENT
      {
        if (columns.size() <= 1u)
          throw wrong_mnemonics_error{columns};

        std::for_each(
          std::next(begin(columns)), end(columns),
          [](std::string& str) { boost::algorithm::to_upper(str); });

        auto const statement = read_do_statement(columns);

        if (statement == ::bra::do_statement::measurement)
        {
#ifndef BRA_NO_MPI
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::measurement >(root_));
#else // BRA_NO_MPI
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::measurement >());
#endif // BRA_NO_MPI
        }
        else if (statement == ::bra::do_statement::amplitudes)
        {
#ifndef BRA_NO_MPI
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::amplitudes >(root_));
#else // BRA_NO_MPI
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::amplitudes >());
#endif // BRA_NO_MPI
        }
        else
          throw unsupported_mnemonic_error{mnemonic};
      }
      else if (mnemonic == "END") // END MEASUREMENT/LEARNING MACHINE/FUSION/CIRCUIT
      {
        auto const statement = read_end_statement(columns);

        if (statement == ::bra::end_statement::fusion)
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::end_fusion >());
        else if (statement == ::bra::end_statement::circuit)
        {
          if (not is_in_circuit_)
            throw wrong_mnemonics_error{columns};

          circuit_index_ = 0;
          is_in_circuit_ = false;
        }
        else
          throw unsupported_mnemonic_error{mnemonic};
        /*
        auto const statement = read_end_statement(columns);

        if (statement == ::bra::end_statement::measurement)
        {
#ifndef BRA_NO_MPI
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::measurement >(root_));
#else // BRA_NO_MPI
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::measurement >());
#endif // BRA_NO_MPI
        }
        else if (statement == ::bra::end_statement::learning_machine)
          throw unsupported_mnemonic_error{mnemonic};

        throw unsupported_mnemonic_error{mnemonic};
*/
      }
      else if (mnemonic == "GENERATE") // GENERATE EVENTS
      {
        if (boost::size(columns) != 4u)
          throw wrong_mnemonics_error{columns};

        boost::algorithm::to_upper(columns[1u]);

        auto statement = ::bra::generate_statement{};
        auto num_events = int{};
        auto seed = int{};
        std::tie(statement, num_events, seed) = read_generate_statement(columns);

        if (statement == ::bra::generate_statement::events)
        {
#ifndef BRA_NO_MPI
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::generate_events >(root_, num_events, seed));
#else // BRA_NO_MPI
          circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::generate_events >(num_events, seed));
#endif // BRA_NO_MPI
          break;
        }
      }
      else if (mnemonic == "CLEAR")
        add_clear(columns);
      else if (mnemonic == "SET")
        add_set(columns);
      else if (mnemonic == "DEPOLARIZING")
      {
        if (boost::size(columns) <= 2u)
          throw wrong_mnemonics_error{columns};

        boost::algorithm::to_upper(columns[1u]);

        add_depolarizing(columns, mnemonic);
      }
      else if (mnemonic == "SX")
        add_sx(columns);
      else if (mnemonic == "SX+")
        add_adj_sx(columns);
      else if (mnemonic == "SY")
        add_sy(columns);
      else if (mnemonic == "SY+")
        add_adj_sy(columns);
      else if (mnemonic == "SZ")
        add_sz(columns);
      else if (mnemonic == "SZ+")
        add_adj_sz(columns);
      else if (mnemonic == "SZZ")
        add_szz(columns);
      else if (mnemonic == "SZZ+")
        add_adj_szz(columns);
      else if (mnemonic.size() >= 4u and mnemonic.front() == 'S'
               and std::all_of(std::next(begin(mnemonic)), end(mnemonic), [](char const character) { return character == 'Z'; }))
        add_szs(columns, mnemonic);
      else if (mnemonic.size() >= 5u and mnemonic.front() == 'S'
               and std::all_of(std::next(begin(mnemonic)), std::prev(end(mnemonic)), [](char const character) { return character == 'Z'; })
               and mnemonic.back() == '+')
        add_adj_szs(columns, mnemonic);
      else if (mnemonic.size() >= 4u and mnemonic[0] == 'S' and mnemonic[1] == 'Z' and mnemonic.back() == '+')
        add_adj_szn(columns, mnemonic);
      else if (mnemonic.size() >= 3u and mnemonic[0] == 'S' and mnemonic[1] == 'Z')
        add_szn(columns, mnemonic);
      else if (mnemonic.size() >= 2u and mnemonic.front() == 'C') // controlled gates
        interpret_controlled_gates(columns, mnemonic);
      else
        throw unsupported_mnemonic_error{mnemonic};
    }
  }

  void interpreter::swap(interpreter& other)
    noexcept(
      BRA_is_nothrow_swappable<std::vector<circuit_type>>::value
      and BRA_is_nothrow_swappable< ::bra::bit_integer_type >::value
      and BRA_is_nothrow_swappable< ::bra::state_integer_type >::value
      and BRA_is_nothrow_swappable< ::bra::qubit_type >::value)
  {
    using std::swap;
#ifndef BRA_NO_MPI
    swap(circuits_, other.circuits_);
    swap(label_maps_, other.label_maps_);
    swap(num_qubits_, other.num_qubits_);
    swap(num_lqubits_, other.num_lqubits_);
    swap(initial_state_value_, other.initial_state_value_);
    swap(initial_permutation_, other.initial_permutation_);
    swap(root_, other.root_);
#else // BRA_NO_MPI
    swap(circuits_, other.circuits_);
    swap(label_maps_, other.label_maps_);
    swap(num_qubits_, other.num_qubits_);
    swap(initial_state_value_, other.initial_state_value_);
#endif // BRA_NO_MPI
  }

  void interpreter::apply_circuit(::bra::state& state, int const circuit_index) const
  {
    auto const count = static_cast<int>(circuits_[circuit_index].size());
    for (auto index = 0; index < count; ++index)
    {
      state << *(circuits_[circuit_index][index]);

      if (!state.maybe_label())
        continue;

      index = static_cast<int>(label_maps_[circuit_index].at(*(state.maybe_label())));
      --index;
      state.delete_label();
    }
  }

  ::bra::bit_integer_type interpreter::read_num_qubits(interpreter::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw ::bra::wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    return boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
  }

  ::bra::state_integer_type interpreter::read_initial_state_value(interpreter::columns_type& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    boost::algorithm::to_upper(*++iter);
    if (columns[1] != "STATE")
      throw wrong_mnemonics_error{columns};

    return boost::lexical_cast< ::bra::state_integer_type >(*++iter);
  }

  ::bra::bit_integer_type interpreter::read_num_mpi_processes(interpreter::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw ::bra::wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    return boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
  }

  ::bra::state_integer_type interpreter::read_mpi_buffer_size(interpreter::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw ::bra::wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    return boost::lexical_cast< ::bra::state_integer_type >(*++iter);
  }

  ::bra::state_integer_type interpreter::read_num_circuits(interpreter::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    return boost::lexical_cast< ::bra::state_integer_type >(*++iter);
  }

#ifndef BRA_NO_MPI
  std::vector< ::bra::permutated_qubit_type >
  interpreter::read_initial_permutation(interpreter::columns_type const& columns) const
  {
    auto result = std::vector< ::bra::permutated_qubit_type >{};
    result.reserve(boost::size(columns)-2u);

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    ++iter;

    auto const last = std::end(columns);
    for (; iter != last; ++iter)
      result.push_back(static_cast< ::bra::permutated_qubit_type >(boost::lexical_cast< ::bra::bit_integer_type >(*iter)));

    return result;
  }
#endif

  ::bra::qubit_type interpreter::read_target(interpreter::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);

    return ket::make_qubit< ::bra::state_integer_type >(target);
  }

  ::bra::control_qubit_type interpreter::read_control(interpreter::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);

    return ket::make_control(ket::make_qubit< ::bra::state_integer_type >(target));
  }

  std::tuple< ::bra::qubit_type, ::bra::qubit_type > interpreter::read_2targets(interpreter::columns_type const& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const target1 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const target2 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);

    return std::make_tuple(ket::make_qubit< ::bra::state_integer_type >(target1), ket::make_qubit< ::bra::state_integer_type >(target2));
  }

  std::tuple< ::bra::control_qubit_type, ::bra::control_qubit_type > interpreter::read_2controls(interpreter::columns_type const& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const target1 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const target2 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);

    return std::make_tuple(ket::make_control(ket::make_qubit< ::bra::state_integer_type >(target1)), ket::make_control(ket::make_qubit< ::bra::state_integer_type >(target2)));
  }

  void interpreter::read_multi_targets(interpreter::columns_type const& columns, std::vector< ::bra::qubit_type >& targets) const
  {
    if (boost::size(columns) < 4u or targets.size() != boost::size(columns) - 1u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const targets_last = end(targets);
    for (auto targets_iter = begin(targets); targets_iter != targets_last; ++targets_iter, ++iter)
    {
      auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *targets_iter = ket::make_qubit< ::bra::state_integer_type >(target);
    }
  }

  void interpreter::read_multi_controls(interpreter::columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const
  {
    if (boost::size(columns) < 4u or controls.size() != boost::size(columns) - 1u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }
  }

  ::bra::qubit_type interpreter::read_target_phase(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::real_type, std::string >& phase) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase_string.front())))
      phase = boost::lexical_cast< ::bra::real_type >(phase_string);
    else
      phase = phase_string;

    return ket::make_qubit< ::bra::state_integer_type >(target);
  }

  ::bra::control_qubit_type interpreter::read_control_phase(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::real_type, std::string >& phase) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase_string.front())))
      phase = boost::lexical_cast< ::bra::real_type >(phase_string);
    else
      phase = phase_string;

    return ket::make_control(ket::make_qubit< ::bra::state_integer_type >(target));
  }

  ::bra::qubit_type interpreter::read_target_2phases(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::real_type, std::string >& phase1,
    boost::variant< ::bra::real_type, std::string >& phase2) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase1_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase1_string.front())))
      phase1 = boost::lexical_cast< ::bra::real_type >(phase1_string);
    else
      phase1 = phase1_string;
    auto const phase2_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase2_string.front())))
      phase2 = boost::lexical_cast< ::bra::real_type >(phase2_string);
    else
      phase2 = phase2_string;

    return ket::make_qubit< ::bra::state_integer_type >(target);
  }

  ::bra::qubit_type interpreter::read_target_3phases(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::real_type, std::string >& phase1,
    boost::variant< ::bra::real_type, std::string >& phase2,
    boost::variant< ::bra::real_type, std::string >& phase3) const
  {
    if (boost::size(columns) != 5u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase1_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase1_string.front())))
      phase1 = boost::lexical_cast< ::bra::real_type >(phase1_string);
    else
      phase1 = phase1_string;
    auto const phase2_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase2_string.front())))
      phase2 = boost::lexical_cast< ::bra::real_type >(phase2_string);
    else
      phase2 = phase2_string;
    auto const phase3_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase3_string.front())))
      phase3 = boost::lexical_cast< ::bra::real_type >(phase3_string);
    else
      phase3 = phase3_string;

    return ket::make_qubit< ::bra::state_integer_type >(target);
  }

  ::bra::qubit_type interpreter::read_target_phaseexp(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::int_type, std::string >& phase_exponent) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase_exponent_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase_exponent_string.front())))
      phase_exponent = boost::lexical_cast< ::bra::int_type >(phase_exponent_string);
    else
      phase_exponent = phase_exponent_string;

    return ket::make_qubit< ::bra::state_integer_type >(target);
  }

  ::bra::control_qubit_type interpreter::read_control_phaseexp(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::int_type, std::string >& phase_exponent) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase_exponent_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase_exponent_string.front())))
      phase_exponent = boost::lexical_cast< ::bra::int_type >(phase_exponent_string);
    else
      phase_exponent = phase_exponent_string;

    return ket::make_control(ket::make_qubit< ::bra::state_integer_type >(target));
  }

  std::tuple< ::bra::qubit_type, ::bra::qubit_type >
  interpreter::read_2targets_phase(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::real_type, std::string >& phase) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const target1 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const target2 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase_string.front())))
      phase = boost::lexical_cast< ::bra::real_type >(phase_string);
    else
      phase = phase_string;

    return std::make_tuple(ket::make_qubit< ::bra::state_integer_type >(target1), ket::make_qubit< ::bra::state_integer_type >(target2));
  }

  void interpreter::read_multi_targets_phase(
    interpreter::columns_type const& columns,
    std::vector< ::bra::qubit_type >& targets,
    boost::variant< ::bra::real_type, std::string >& phase) const
  {
    if (boost::size(columns) < 5u or targets.size() == boost::size(columns) - 2u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const targets_last = end(targets);
    for (auto targets_iter = begin(targets); targets_iter != targets_last; ++targets_iter, ++iter)
    {
      auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *targets_iter = ket::make_qubit< ::bra::state_integer_type >(target);
    }
    auto const phase_string = *iter;
    if (std::isdigit(static_cast<unsigned char>(phase_string.front())))
      phase = boost::lexical_cast< ::bra::real_type >(phase_string);
    else
      phase = phase_string;
  }

  std::tuple< ::bra::control_qubit_type, ::bra::qubit_type > interpreter::read_control_target(interpreter::columns_type const& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);

    return std::make_tuple(
      ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control)),
      ket::make_qubit< ::bra::state_integer_type >(target));
  }

  std::tuple< ::bra::control_qubit_type, ::bra::qubit_type >
  interpreter::read_control_target_phaseexp(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::int_type, std::string >& phase_exponent) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase_exponent_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase_exponent_string.front())))
      phase_exponent = boost::lexical_cast< ::bra::int_type >(phase_exponent_string);
    else
      phase_exponent = phase_exponent_string;

    return std::make_tuple(
      ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control)),
      ket::make_qubit< ::bra::state_integer_type >(target));
  }

  std::tuple< ::bra::control_qubit_type, ::bra::control_qubit_type >
  interpreter::read_2controls_phaseexp(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::int_type, std::string >& phase_exponent) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const control1 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const control2 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase_exponent_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase_exponent_string.front())))
      phase_exponent = boost::lexical_cast< ::bra::int_type >(phase_exponent_string);
    else
      phase_exponent = phase_exponent_string;

    return std::make_tuple(
      ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control1)),
      ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control2)));
  }

  std::tuple< ::bra::control_qubit_type, ::bra::control_qubit_type, ::bra::qubit_type >
  interpreter::read_2controls_target(interpreter::columns_type const& columns) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const control1 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const control2 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);

    return std::make_tuple(
      ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control1)),
      ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control2)),
      ket::make_qubit< ::bra::state_integer_type >(target));
  }

  void interpreter::read_multi_controls_phase(
    interpreter::columns_type const& columns,
    std::vector< ::bra::control_qubit_type >& controls,
    boost::variant< ::bra::real_type, std::string >& phase) const
  {
    if (boost::size(columns) < 4u or controls.size() != boost::size(columns) - 2u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }

    auto const phase_string = *iter;
    if (std::isdigit(static_cast<unsigned char>(phase_string.front())))
      phase = boost::lexical_cast< ::bra::real_type >(phase_string);
    else
      phase = phase_string;
  }

  ::bra::qubit_type interpreter::read_multi_controls_target(
    interpreter::columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const
  {
    if (boost::size(columns) < 4u or controls.size() != boost::size(columns) - 2u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }

    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
    return ket::make_qubit< ::bra::state_integer_type >(target);
  }

  std::tuple< ::bra::qubit_type, ::bra::qubit_type >
  interpreter::read_multi_controls_2targets(
    interpreter::columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const
  {
    if (boost::size(columns) < 5u or controls.size() != boost::size(columns) - 3u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }

    auto const target1 = boost::lexical_cast< ::bra::bit_integer_type >(*iter++);
    auto const target2 = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
    return std::make_tuple(
       ket::make_qubit< ::bra::state_integer_type >(target1),
       ket::make_qubit< ::bra::state_integer_type >(target2));
  }

  void interpreter::read_multi_controls_multi_targets(
    interpreter::columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls, std::vector< ::bra::qubit_type >& targets) const
  {
    if (boost::size(columns) < 4u or controls.size() + targets.size() != boost::size(columns) - 1u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }

    auto const targets_last = end(targets);
    for (auto targets_iter = begin(targets); targets_iter != targets_last; ++targets_iter, ++iter)
    {
      auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *targets_iter = ket::make_qubit< ::bra::state_integer_type >(target);
    }
  }

  std::tuple< ::bra::control_qubit_type, ::bra::qubit_type >
  interpreter::read_control_target_phase(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::real_type, std::string >& phase) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase_string.front())))
      phase = boost::lexical_cast< ::bra::real_type >(phase_string);
    else
      phase = phase_string;

    return std::make_tuple(
      ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control)),
      ket::make_qubit< ::bra::state_integer_type >(target));
  }

  std::tuple< ::bra::control_qubit_type, ::bra::control_qubit_type >
  interpreter::read_2controls_phase(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::real_type, std::string >& phase) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const control1 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const control2 = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase_string.front())))
      phase = boost::lexical_cast< ::bra::real_type >(phase_string);
    else
      phase = phase_string;

    return std::make_tuple(
      ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control1)),
      ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control2)));
  }

  ::bra::qubit_type interpreter::read_multi_controls_target_phase(
    interpreter::columns_type const& columns,
    std::vector< ::bra::control_qubit_type >& controls,
    boost::variant< ::bra::real_type, std::string >& phase) const
  {
    if (boost::size(columns) < 5u or controls.size() != boost::size(columns) - 3u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }

    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*iter++);
    auto const phase_string = *iter;
    if (std::isdigit(static_cast<unsigned char>(phase_string.front())))
      phase = boost::lexical_cast< ::bra::real_type >(phase_string);
    else
      phase = phase_string;

    return ket::make_qubit< ::bra::state_integer_type >(target);
  }

  std::tuple< ::bra::control_qubit_type, ::bra::qubit_type >
  interpreter::read_control_target_2phases(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::real_type, std::string >& phase1,
    boost::variant< ::bra::real_type, std::string >& phase2) const
  {
    if (boost::size(columns) != 5u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase1_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase1_string.front())))
      phase1 = boost::lexical_cast< ::bra::real_type >(phase1_string);
    else
      phase1 = phase1_string;
    auto const phase2_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase2_string.front())))
      phase2 = boost::lexical_cast< ::bra::real_type >(phase2_string);
    else
      phase2 = phase2_string;

    return std::make_tuple(
      ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control)),
      ket::make_qubit< ::bra::state_integer_type >(target));
  }

  ::bra::qubit_type interpreter::read_multi_controls_target_2phases(
    interpreter::columns_type const& columns,
    std::vector< ::bra::control_qubit_type >& controls,
    boost::variant< ::bra::real_type, std::string >& phase1,
    boost::variant< ::bra::real_type, std::string >& phase2) const
  {
    if (boost::size(columns) < 6u or controls.size() != boost::size(columns) - 4u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }

    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*iter++);
    auto const phase1_string = *iter++;
    if (std::isdigit(static_cast<unsigned char>(phase1_string.front())))
      phase1 = boost::lexical_cast< ::bra::real_type >(phase1_string);
    else
      phase1 = phase1_string;
    auto const phase2_string = *iter;
    if (std::isdigit(static_cast<unsigned char>(phase2_string.front())))
      phase2 = boost::lexical_cast< ::bra::real_type >(phase2_string);
    else
      phase2 = phase2_string;

    return ket::make_qubit< ::bra::state_integer_type >(target);
  }

  std::tuple< ::bra::control_qubit_type, ::bra::qubit_type >
  interpreter::read_control_target_3phases(
    interpreter::columns_type const& columns,
    boost::variant< ::bra::real_type, std::string >& phase1,
    boost::variant< ::bra::real_type, std::string >& phase2,
    boost::variant< ::bra::real_type, std::string >& phase3) const
  {
    if (boost::size(columns) != 6u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const phase1_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase1_string.front())))
      phase1 = boost::lexical_cast< ::bra::real_type >(phase1_string);
    else
      phase1 = phase1_string;
    auto const phase2_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase2_string.front())))
      phase2 = boost::lexical_cast< ::bra::real_type >(phase2_string);
    else
      phase2 = phase1_string;
    auto const phase3_string = *++iter;
    if (std::isdigit(static_cast<unsigned char>(phase3_string.front())))
      phase3 = boost::lexical_cast< ::bra::real_type >(phase3_string);
    else
      phase3 = phase3_string;

    return std::make_tuple(
      ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control)),
      ket::make_qubit< ::bra::state_integer_type >(target));
  }

  ::bra::qubit_type interpreter::read_multi_controls_target_3phases(
    interpreter::columns_type const& columns,
    std::vector< ::bra::control_qubit_type >& controls,
    boost::variant< ::bra::real_type, std::string >& phase1,
    boost::variant< ::bra::real_type, std::string >& phase2,
    boost::variant< ::bra::real_type, std::string >& phase3) const
  {
    if (boost::size(columns) < 7u or controls.size() != boost::size(columns) - 5u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }

    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*iter++);
    auto const phase1_string = *iter++;
    if (std::isdigit(static_cast<unsigned char>(phase1_string.front())))
      phase1 = boost::lexical_cast< ::bra::real_type >(phase1_string);
    else
      phase1 = phase1_string;
    auto const phase2_string = *iter++;
    if (std::isdigit(static_cast<unsigned char>(phase2_string.front())))
      phase2 = boost::lexical_cast< ::bra::real_type >(phase2_string);
    else
      phase2 = phase1_string;
    auto const phase3_string = *iter;
    if (std::isdigit(static_cast<unsigned char>(phase3_string.front())))
      phase3 = boost::lexical_cast< ::bra::real_type >(phase3_string);
    else
      phase3 = phase3_string;

    return ket::make_qubit< ::bra::state_integer_type >(target);
  }

  void interpreter::read_multi_controls_phaseexp(
    interpreter::columns_type const& columns,
    std::vector< ::bra::control_qubit_type >& controls,
    boost::variant< ::bra::int_type, std::string >& phase_exponent) const
  {
    if (boost::size(columns) < 4u or controls.size() != boost::size(columns) - 2u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }

    auto const phase_exponent_string = *iter;
    if (std::isdigit(static_cast<unsigned char>(phase_exponent_string.front())))
      phase_exponent = boost::lexical_cast< ::bra::real_type >(phase_exponent_string);
    else
      phase_exponent = phase_exponent_string;
  }

  ::bra::qubit_type interpreter::read_multi_controls_target_phaseexp(
    interpreter::columns_type const& columns,
    std::vector< ::bra::control_qubit_type >& controls,
    boost::variant< ::bra::int_type, std::string >& phase_exponent) const
  {
    if (boost::size(columns) < 5u or controls.size() != boost::size(columns) - 3u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }

    auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*iter++);
    auto const phase_exponent_string = *iter;
    if (std::isdigit(static_cast<unsigned char>(phase_exponent_string.front())))
      phase_exponent = boost::lexical_cast< ::bra::real_type >(phase_exponent_string);
    else
      phase_exponent = phase_exponent_string;

    return ket::make_qubit< ::bra::state_integer_type >(target);
  }

  void interpreter::read_multi_controls_multi_targets_phase(
    interpreter::columns_type const& columns,
    std::vector< ::bra::control_qubit_type >& controls,
    std::vector< ::bra::qubit_type >& targets,
    boost::variant< ::bra::real_type, std::string >& phase) const
  {
    if (boost::size(columns) < 5u or controls.size() + targets.size() != boost::size(columns) - 2u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }

    auto const targets_last = end(targets);
    for (auto targets_iter = begin(targets); targets_iter != targets_last; ++targets_iter, ++iter)
    {
      auto const target = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *targets_iter = ket::make_qubit< ::bra::state_integer_type >(target);
    }

    auto const phase_string = *iter;
    if (std::isdigit(static_cast<unsigned char>(phase_string.front())))
      phase = boost::lexical_cast< ::bra::real_type >(phase_string);
    else
      phase = phase_string;
  }

  std::tuple< ::bra::qubit_type, ::bra::qubit_type >
  interpreter::read_multi_controls_2targets_phase(
    interpreter::columns_type const& columns,
    std::vector< ::bra::control_qubit_type >& controls,
    boost::variant< ::bra::real_type, std::string >& phase) const
  {
    if (boost::size(columns) < 6u or controls.size() != boost::size(columns) - 4u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    ++iter;
    using std::end;
    auto const controls_last = end(controls);
    for (auto controls_iter = begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast< ::bra::bit_integer_type >(*iter);
      *controls_iter = ket::make_control(ket::make_qubit< ::bra::state_integer_type >(control));
    }

    auto const target1 = boost::lexical_cast< ::bra::bit_integer_type >(*iter++);
    auto const target2 = boost::lexical_cast< ::bra::bit_integer_type >(*iter++);
    auto const phase_string = *iter;
    if (std::isdigit(static_cast<unsigned char>(phase_string.front())))
      phase = boost::lexical_cast< ::bra::real_type >(phase_string);
    else
      phase = phase_string;

    return std::make_tuple(
       ket::make_qubit< ::bra::state_integer_type >(target1),
       ket::make_qubit< ::bra::state_integer_type >(target2));
  }

  ::bra::begin_statement interpreter::read_begin_statement(interpreter::columns_type const& columns) const
  {
    auto const column_size = boost::size(columns);

    assert(column_size >= 2u);

    using std::begin;
    auto iter = begin(columns);
    ++iter;

    if (*iter == "FUSION"/* and column_size == 2u*/)
      return ::bra::begin_statement::fusion;
    else if (*iter == "CIRCUIT" and column_size == 3u)
      return ::bra::begin_statement::circuit;
    else if (*iter == "LEARNING" and column_size == 3u and *std::next(iter) == "MACHINE")
      return ::bra::begin_statement::learning_machine;
    else if (*iter == "MEASUREMENT" and column_size == 2u)
      return ::bra::begin_statement::measurement;

    throw wrong_mnemonics_error{columns};
  }

  ::bra::end_statement interpreter::read_end_statement(interpreter::columns_type const& columns) const
  {
    auto const column_size = boost::size(columns);

    assert(column_size >= 2u);

    using std::begin;
    auto iter = begin(columns);
    ++iter;

    if (*iter == "FUSION" and column_size == 2u)
      return ::bra::end_statement::fusion;
    if (*iter == "CIRCUIT" and column_size == 2u)
      return ::bra::end_statement::circuit;

    throw wrong_mnemonics_error{columns};
  }

  ::bra::do_statement interpreter::read_do_statement(interpreter::columns_type const& columns) const
  {
    auto const column_size = boost::size(columns);

    assert(column_size >= 2u);

    using std::begin;
    auto iter = begin(columns);
    ++iter;

    if (*iter == "MEASUREMENT" and column_size == 2u)
      return ::bra::do_statement::measurement;
    else if (*iter == "AMPLITUDES" and column_size == 2u)
      return ::bra::do_statement::amplitudes;

    throw wrong_mnemonics_error{columns};
  }

  ::bra::bit_statement interpreter::read_bit_statement(interpreter::columns_type const& columns) const
  {
    assert(boost::size(columns) >= 2u);

    using std::begin;
    auto iter = begin(columns);
    if (*++iter != "ASSIGNMENT")
      throw wrong_mnemonics_error{columns};

    return ::bra::bit_statement::assignment;
  }

  std::tuple< ::bra::bit_integer_type, ::bra::state_integer_type, ::bra::state_integer_type >
  interpreter::read_shor_box(interpreter::columns_type const& columns) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const num_exponent_qubits = boost::lexical_cast< ::bra::bit_integer_type >(*++iter);
    auto const divisor = boost::lexical_cast< ::bra::state_integer_type >(*++iter);
    auto const base = boost::lexical_cast< ::bra::state_integer_type >(*++iter);

    return std::make_tuple(num_exponent_qubits, divisor, base);
  }

  std::tuple< ::bra::generate_statement, int, int > interpreter::read_generate_statement(interpreter::columns_type const& columns) const
  {
    assert(boost::size(columns) == 4u);

    using std::begin;
    auto iter = begin(columns);
    if (*++iter != "EVENTS")
      throw wrong_mnemonics_error{columns};

    auto const num_events = boost::lexical_cast<int>(*++iter);
    auto const seed = boost::lexical_cast<int>(*++iter);
    return std::make_tuple(::bra::generate_statement::events, num_events, seed);
  }

  std::tuple< ::bra::depolarizing_statement, ::bra::real_type, ::bra::real_type, ::bra::real_type, int >
  interpreter::read_depolarizing_statement(interpreter::columns_type const& columns) const
  {
    assert(boost::size(columns) >= 3u);

    auto iter = columns.cbegin();
    if (*++iter != "CHANNEL")
      throw wrong_mnemonics_error{columns};

    auto px = real_type{0};
    auto py = real_type{0};
    auto pz = real_type{0};
    auto seed = -1;
    auto is_px_checked = false;
    auto is_py_checked = false;
    auto is_pz_checked = false;
    auto is_seed_checked = false;

    auto present_string = *++iter; // present_string == "P_*" or "P_*=" or "P_*=0.xxx" or "P_*=0.xxx," or "P_*=0.xxx,..."
    auto probability_string = std::string{"    "};
    auto const last = columns.cend();
    while (iter != last)
    {
      auto string_found = std::find(present_string.cbegin(), present_string.cend(), '=');
      probability_string.assign(present_string.cbegin(), string_found);
      boost::algorithm::to_upper(probability_string);
      if (probability_string == "P_X")
      {
        if (is_px_checked)
          throw wrong_mnemonics_error{columns};

        ::bra::interpreter_detail::read_depolarizing_statement(px, present_string, iter, last, string_found, columns);
        if (px < 0.0 or px > 1.0)
          throw wrong_mnemonics_error{columns};

        is_px_checked = true;
      }
      else if (probability_string == "P_Y")
      {
        if (is_py_checked)
          throw wrong_mnemonics_error{columns};

        ::bra::interpreter_detail::read_depolarizing_statement(py, present_string, iter, last, string_found, columns);
        if (py < 0.0 or py > 1.0)
          throw wrong_mnemonics_error{columns};

        is_py_checked = true;
      }
      else if (probability_string == "P_Z")
      {
        if (is_pz_checked)
          throw wrong_mnemonics_error{columns};

        ::bra::interpreter_detail::read_depolarizing_statement(pz, present_string, iter, last, string_found, columns);
        if (pz < 0.0 or pz > 1.0)
          throw wrong_mnemonics_error{columns};

        is_pz_checked = true;
      }
      else if (probability_string == "SEED")
      {
        if (is_seed_checked)
          throw wrong_mnemonics_error{columns};

        ::bra::interpreter_detail::read_depolarizing_statement(seed, present_string, iter, last, string_found, columns);
        is_seed_checked = true;
      }
      else
        throw wrong_mnemonics_error{columns};
    }

    if (px > real_type{1} or px < real_type{0} or py > real_type{1} or py < real_type{0} or pz > real_type{1} or pz < real_type{0} or px + py + pz > real_type{1})
      throw wrong_mnemonics_error{columns};

    return std::make_tuple(::bra::depolarizing_statement::channel, px, py, pz, seed);
  }

  // VAR X REAL ! declare real variable X
  // VAR NS INT 4 ! declare array with 4 integer elements NS
  void interpreter::add_var(interpreter::columns_type const& columns)
  {
    auto const column_size = boost::size(columns);
    if (not (column_size == 3u or column_size == 4u))
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const variable_name = boost::algorithm::to_upper_copy(*++iter);
    auto const type_name = boost::algorithm::to_upper_copy(*++iter);
    auto const num_elements = column_size == 3u ? 1 : boost::lexical_cast<int>(*++iter);

    auto const type
      = type_name == "REAL"
        ? ::bra::variable_type::real
        : type_name == "INT"
          ? ::bra::variable_type::integer
          : throw 1;

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::var_op >(variable_name, type, num_elements));
  }

  // LET X := 1.0
  // LET NS:1 += N
  void interpreter::add_let(interpreter::columns_type const& columns)
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const lhs_variable_name = boost::algorithm::to_upper_copy(*++iter);
    auto const op_str = *++iter;
    auto const rhs_literal_or_variable_name = boost::algorithm::to_upper_copy(*++iter);

    auto const op
      = op_str == ":="
        ? ::bra::assign_operation_type::assign
        : op_str == "+="
          ? ::bra::assign_operation_type::plus_assign
          : op_str == "-="
            ? ::bra::assign_operation_type::minus_assign
            : op_str == "*="
              ? ::bra::assign_operation_type::multiplies_assign
              : op_str == "/="
                ? ::bra::assign_operation_type::divides_assign
                : throw 1;

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::let_op >(lhs_variable_name, op, rhs_literal_or_variable_name));
  }

  // @LABEL
  void interpreter::add_label(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    if (boost::size(columns) != 1u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    using std::end;
    auto label = std::string{std::next(begin(mnemonic)), end(mnemonic)};
    boost::algorithm::to_upper(label);

    if (label_maps_[circuit_index_].find(label) != end(label_maps_[circuit_index_]))
      throw wrong_mnemonics_error{columns};

    label_maps_[circuit_index_].emplace(label, static_cast<int>(circuits_[circuit_index_].size()));
  }

  // JUMP LABEL
  void interpreter::add_jump(interpreter::columns_type const& columns)
  {
    if (boost::size(columns) != 2u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const label = boost::algorithm::to_upper_copy(*++iter);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::jump_op >(label));
  }

  // JUMPIF LABEL N < 5
  void interpreter::add_jumpif(interpreter::columns_type const& columns)
  {
    if (boost::size(columns) != 5u)
      throw wrong_mnemonics_error{columns};

    using std::begin;
    auto iter = begin(columns);
    auto const label = boost::algorithm::to_upper_copy(*++iter);
    auto const lhs_variable_name = boost::algorithm::to_upper_copy(*++iter);
    auto const op_str = *++iter;
    auto const rhs_literal_or_variable_name = boost::algorithm::to_upper_copy(*++iter);

    auto const op
      = op_str == "=="
        ? ::bra::compare_operation_type::equal_to
        : op_str == "\\="
          ? ::bra::compare_operation_type::not_equal_to
          : op_str == ">"
            ? ::bra::compare_operation_type::greater
            : op_str == "<"
              ? ::bra::compare_operation_type::less
              : op_str == ">="
                ? ::bra::compare_operation_type::greater_equal
                : op_str == "<="
                  ? ::bra::compare_operation_type::less_equal
                  : throw 1;

    circuits_[circuit_index_].push_back(
      std::make_unique< ::bra::gate::jumpif_op >(label, lhs_variable_name, op, rhs_literal_or_variable_name));
  }

  void interpreter::add_i(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::i_gate >(read_target(columns))); }

  void interpreter::add_ic(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::ic_gate >(read_control(columns))); }

  void interpreter::add_ii(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    std::tie(target1, target2) = read_2targets(columns);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::ii_gate >(target1, target2));
  }

  void interpreter::add_is(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size());
    read_multi_targets(columns, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::in_gate >(std::move(targets)));
  }

  void interpreter::add_in(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = end(mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::i_gate >(read_target(columns)));
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      std::tie(target1, target2) = read_2targets(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::ii_gate >(target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      read_multi_targets(columns, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::in_gate >(std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_h(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::hadamard >(read_target(columns))); }

  void interpreter::add_not(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::not_ >(read_target(columns))); }

  void interpreter::add_x(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_x >(read_target(columns))); }

  void interpreter::add_xx(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    std::tie(target1, target2) = read_2targets(columns);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_xx >(target1, target2));
  }

  void interpreter::add_xs(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size());
    read_multi_targets(columns, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_xn >(std::move(targets)));
  }

  void interpreter::add_xn(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = end(mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_x >(read_target(columns)));
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      std::tie(target1, target2) = read_2targets(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_xx >(target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      read_multi_targets(columns, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_xn >(std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_y(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_y >(read_target(columns))); }

  void interpreter::add_yy(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    std::tie(target1, target2) = read_2targets(columns);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_yy >(target1, target2));
  }

  void interpreter::add_ys(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size());
    read_multi_targets(columns, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_yn >(std::move(targets)));
  }

  void interpreter::add_yn(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = end(mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_y >(read_target(columns)));
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      std::tie(target1, target2) = read_2targets(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_yy >(target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      read_multi_targets(columns, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_yn >(std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_z(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_z >(read_control(columns))); }

  void interpreter::add_zz(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    std::tie(target1, target2) = read_2targets(columns);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_zz >(target1, target2));
  }

  void interpreter::add_zs(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size());
    read_multi_targets(columns, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_zn >(std::move(targets)));
  }

  void interpreter::add_zn(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = end(mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_z >(read_control(columns)));
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      std::tie(target1, target2) = read_2targets(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_zz >(target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      read_multi_targets(columns, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::pauli_zn >(std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_swap(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    std::tie(target1, target2) = read_2targets(columns);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::swap >(target1, target2));
  }

  void interpreter::add_sx(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::sqrt_pauli_x >(read_target(columns))); }

  void interpreter::add_adj_sx(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_sqrt_pauli_x >(read_target(columns))); }

  void interpreter::add_sy(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::sqrt_pauli_y >(read_target(columns))); }

  void interpreter::add_adj_sy(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_sqrt_pauli_y >(read_target(columns))); }

  void interpreter::add_sz(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::sqrt_pauli_z >(read_control(columns))); }

  void interpreter::add_adj_sz(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_sqrt_pauli_z >(read_control(columns))); }

  void interpreter::add_szz(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    std::tie(target1, target2) = read_2targets(columns);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::sqrt_pauli_zz >(target1, target2));
  }

  void interpreter::add_adj_szz(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    std::tie(target1, target2) = read_2targets(columns);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_sqrt_pauli_zz >(target1, target2));
  }

  void interpreter::add_szs(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size() - 1u);
    read_multi_targets(columns, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::sqrt_pauli_zn >(std::move(targets)));
  }

  void interpreter::add_adj_szs(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size() - 2u);
    read_multi_targets(columns, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_sqrt_pauli_zn >(std::move(targets)));
  }

  void interpreter::add_szn(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = end(mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::sqrt_pauli_z >(read_control(columns)));
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      std::tie(target1, target2) = read_2targets(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::sqrt_pauli_zz >(target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      read_multi_targets(columns, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::sqrt_pauli_zn >(std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_adj_szn(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = std::prev(end(mnemonic));
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_sqrt_pauli_z >(read_control(columns)));
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      std::tie(target1, target2) = read_2targets(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_sqrt_pauli_zz >(target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      read_multi_targets(columns, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_sqrt_pauli_zn >(std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_s(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::s_gate >(read_control(columns))); }

  void interpreter::add_adj_s(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_s_gate >(read_control(columns))); }

  void interpreter::add_t(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::t_gate >(read_control(columns))); }

  void interpreter::add_adj_t(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_t_gate >(read_control(columns))); }

  void interpreter::add_u1(interpreter::columns_type const& columns)
  {
    auto phase = boost::variant<real_type, std::string>{};
    auto const control = read_control_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::u1 >(phase, control));
  }

  void interpreter::add_adj_u1(interpreter::columns_type const& columns)
  {
    auto phase = boost::variant<real_type, std::string>{};
    auto const control = read_control_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_u1 >(phase, control));
  }

  void interpreter::add_u2(interpreter::columns_type const& columns)
  {
    auto phase1 = boost::variant<real_type, std::string>{};
    auto phase2 = boost::variant<real_type, std::string>{};
    auto const target = read_target_2phases(columns, phase1, phase2);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::u2 >(phase1, phase2, target));
  }

  void interpreter::add_adj_u2(interpreter::columns_type const& columns)
  {
    auto phase1 = boost::variant<real_type, std::string>{};
    auto phase2 = boost::variant<real_type, std::string>{};
    auto const target = read_target_2phases(columns, phase1, phase2);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_u2 >(phase1, phase2, target));
  }

  void interpreter::add_u3(interpreter::columns_type const& columns)
  {
    auto phase1 = boost::variant<real_type, std::string>{};
    auto phase2 = boost::variant<real_type, std::string>{};
    auto phase3 = boost::variant<real_type, std::string>{};
    auto const target = read_target_3phases(columns, phase1, phase2, phase3);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::u3 >(phase1, phase2, phase3, target));
  }

  void interpreter::add_adj_u3(interpreter::columns_type const& columns)
  {
    auto phase1 = boost::variant<real_type, std::string>{};
    auto phase2 = boost::variant<real_type, std::string>{};
    auto phase3 = boost::variant<real_type, std::string>{};
    auto const target = read_target_3phases(columns, phase1, phase2, phase3);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_u3 >(phase1, phase2, phase3, target));
  }

  void interpreter::add_r(interpreter::columns_type const& columns)
  {
    auto phase_exponent = boost::variant<int_type, std::string>{};
    auto const control = read_control_phaseexp(columns, phase_exponent);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::phase_shift >(phase_exponent, control));
  }

  void interpreter::add_adj_r(interpreter::columns_type const& columns)
  {
    auto phase_exponent = boost::variant<int_type, std::string>{};
    auto const control = read_control_phaseexp(columns, phase_exponent);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_phase_shift >(phase_exponent, control));
  }

  void interpreter::add_rotx(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::x_rotation_half_pi >(read_target(columns))); }

  void interpreter::add_adj_rotx(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_x_rotation_half_pi >(read_target(columns))); }

  void interpreter::add_roty(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::y_rotation_half_pi >(read_target(columns))); }

  void interpreter::add_adj_roty(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_y_rotation_half_pi >(read_target(columns))); }

  void interpreter::add_u(interpreter::columns_type const& columns)
  {
    auto control1 = ::bra::control_qubit_type{};
    auto control2 = ::bra::control_qubit_type{};
    auto phase_exponent = boost::variant<int_type, std::string>{};
    std::tie(control1, control2) = read_2controls_phaseexp(columns, phase_exponent);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_phase_shift >(phase_exponent, control1, control2));
  }

  void interpreter::add_adj_u(interpreter::columns_type const& columns)
  {
    auto control1 = ::bra::control_qubit_type{};
    auto control2 = ::bra::control_qubit_type{};
    auto phase_exponent = boost::variant<int_type, std::string>{};
    std::tie(control1, control2) = read_2controls_phaseexp(columns, phase_exponent);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_phase_shift >(phase_exponent, control1, control2));
  }

  void interpreter::add_ex(interpreter::columns_type const& columns)
  {
    auto phase = boost::variant<real_type, std::string>{};
    auto const target = read_target_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_x >(phase, target));
  }

  void interpreter::add_adj_ex(interpreter::columns_type const& columns)
  {
    auto phase = boost::variant<real_type, std::string>{};
    auto const target = read_target_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_x >(phase, target));
  }

  void interpreter::add_exx(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    auto phase = boost::variant<real_type, std::string>{};
    std::tie(target1, target2) = read_2targets_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_xx >(phase, target1, target2));
  }

  void interpreter::add_adj_exx(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    auto phase = boost::variant<real_type, std::string>{};
    std::tie(target1, target2) = read_2targets_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_xx >(phase, target1, target2));
  }

  void interpreter::add_exs(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size() - 1u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_targets_phase(columns, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_xn >(phase, std::move(targets)));
  }

  void interpreter::add_adj_exs(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size() - 2u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_targets_phase(columns, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_xn >(phase, std::move(targets)));
  }

  void interpreter::add_exn(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = end(mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
    {
      auto phase = boost::variant<real_type, std::string>{};
      auto const target = read_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_x >(phase, target));
    }
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(target1, target2) = read_2targets_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_xx >(phase, target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_targets_phase(columns, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_xn >(phase, std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_adj_exn(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = std::prev(end(mnemonic));
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
    {
      auto phase = boost::variant<real_type, std::string>{};
      auto const target = read_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_x >(phase, target));
    }
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(target1, target2) = read_2targets_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_xx >(phase, target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_targets_phase(columns, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_xn >(phase, std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_ey(interpreter::columns_type const& columns)
  {
    auto phase = boost::variant<real_type, std::string>{};
    auto const target = read_target_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_y >(phase, target));
  }

  void interpreter::add_adj_ey(interpreter::columns_type const& columns)
  {
    auto phase = boost::variant<real_type, std::string>{};
    auto const target = read_target_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_y >(phase, target));
  }

  void interpreter::add_eyy(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    auto phase = boost::variant<real_type, std::string>{};
    std::tie(target1, target2) = read_2targets_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_yy >(phase, target1, target2));
  }

  void interpreter::add_adj_eyy(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    auto phase = boost::variant<real_type, std::string>{};
    std::tie(target1, target2) = read_2targets_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_yy >(phase, target1, target2));
  }

  void interpreter::add_eys(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size() - 1u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_targets_phase(columns, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_yn >(phase, std::move(targets)));
  }

  void interpreter::add_adj_eys(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size() - 1u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_targets_phase(columns, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_yn >(phase, std::move(targets)));
  }

  void interpreter::add_eyn(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = end(mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
    {
      auto phase = boost::variant<real_type, std::string>{};
      auto const target = read_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_y >(phase, target));
    }
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(target1, target2) = read_2targets_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_yy >(phase, target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_targets_phase(columns, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_yn >(phase, std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_adj_eyn(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = std::prev(end(mnemonic));
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
    {
      auto phase = boost::variant<real_type, std::string>{};
      auto const target = read_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_y >(phase, target));
    }
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(target1, target2) = read_2targets_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_yy >(phase, target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_targets_phase(columns, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_yn >(phase, std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_ez(interpreter::columns_type const& columns)
  {
    auto phase = boost::variant<real_type, std::string>{};
    auto const target = read_target_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_z >(phase, target));
  }

  void interpreter::add_adj_ez(interpreter::columns_type const& columns)
  {
    auto phase = boost::variant<real_type, std::string>{};
    auto const target = read_target_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_z >(phase, target));
  }

  void interpreter::add_ezz(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    auto phase = boost::variant<real_type, std::string>{};
    std::tie(target1, target2) = read_2targets_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_zz >(phase, target1, target2));
  }

  void interpreter::add_adj_ezz(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    auto phase = boost::variant<real_type, std::string>{};
    std::tie(target1, target2) = read_2targets_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_zz >(phase, target1, target2));
  }

  void interpreter::add_ezs(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size() - 1u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_targets_phase(columns, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_zn >(phase, std::move(targets)));
  }

  void interpreter::add_adj_ezs(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector< ::bra::qubit_type >(mnemonic.size() - 1u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_targets_phase(columns, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_zn >(phase, std::move(targets)));
  }

  void interpreter::add_ezn(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = end(mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
    {
      auto phase = boost::variant<real_type, std::string>{};
      auto const target = read_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_z >(phase, target));
    }
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(target1, target2) = read_2targets_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_zz >(phase, target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_targets_phase(columns, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_pauli_zn >(phase, std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_adj_ezn(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last = std::prev(end(mnemonic));
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_qubits == 1)
    {
      auto phase = boost::variant<real_type, std::string>{};
      auto const target = read_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_z >(phase, target));
    }
    else if (num_qubits == 2)
    {
      auto target1 = ::bra::qubit_type{};
      auto target2 = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(target1, target2) = read_2targets_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_zz >(phase, target1, target2));
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector< ::bra::qubit_type >(num_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_targets_phase(columns, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_pauli_zn >(phase, std::move(targets)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_eswap(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    auto phase = boost::variant<real_type, std::string>{};
    std::tie(target1, target2) = read_2targets_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::exponential_swap >(phase, target1, target2));
  }

  void interpreter::add_adj_eswap(interpreter::columns_type const& columns)
  {
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    auto phase = boost::variant<real_type, std::string>{};
    std::tie(target1, target2) = read_2targets_phase(columns, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_exponential_swap >(phase, target1, target2));
  }

  void interpreter::add_toffoli(interpreter::columns_type const& columns)
  {
    auto control1 = ::bra::control_qubit_type{};
    auto control2 = ::bra::control_qubit_type{};
    auto target = ::bra::qubit_type{};
    std::tie(control1, control2, target) = read_2controls_target(columns);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::toffoli >(target, control1, control2));
  }

  void interpreter::add_m(interpreter::columns_type const& columns)
  {
#ifndef BRA_NO_MPI
    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::projective_measurement >(read_target(columns), root_));
#else // BRA_NO_MPI
    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::projective_measurement >(read_target(columns)));
#endif // BRA_NO_MPI
  }

  void interpreter::add_shor_box(interpreter::columns_type const& columns)
  {
    auto num_exponent_qubits = ::bra::bit_integer_type{};
    auto divisor = ::bra::state_integer_type{};
    auto base = ::bra::state_integer_type{};
    std::tie(num_exponent_qubits, divisor, base) = read_shor_box(columns);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::shor_box >(num_exponent_qubits, divisor, base));
  }

  void interpreter::add_clear(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::clear >(read_target(columns))); }

  void interpreter::add_set(interpreter::columns_type const& columns)
  { circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::set >(read_target(columns))); }

  void interpreter::add_depolarizing(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    auto statement = ::bra::depolarizing_statement{};
    auto px = real_type{};
    auto py = real_type{};
    auto pz = real_type{};
    auto seed = int{};
    std::tie(statement, px, py, pz, seed) = read_depolarizing_statement(columns);

    if (statement == ::bra::depolarizing_statement::channel)
      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::depolarizing_channel >(px, py, pz, seed));
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::interpret_controlled_gates(interpreter::columns_type const& columns, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const cs_first = begin(mnemonic);
    auto const cs_last
      = std::find_if_not(
          cs_first, end(mnemonic),
          [](char const character) { return character == 'C'; });
    auto const possible_digits_first = std::next(begin(mnemonic));
    auto const possible_digits_last
      = std::find_if_not(
          possible_digits_first, end(mnemonic),
          [](unsigned char const character) { return std::isdigit(character); });

    auto const num_control_qubits
      = cs_last - cs_first > 1
        ? static_cast<int>(cs_last - cs_first)
        : possible_digits_first == possible_digits_last
          ? 1
          : ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_control_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    auto const noncontrol_mnemonic
      = cs_last - cs_first > 1
        ? std::string{cs_last, end(mnemonic)}
        : std::string{possible_digits_last, end(mnemonic)};

    if (noncontrol_mnemonic == "I")
      add_ci(columns, num_control_qubits);
    if (noncontrol_mnemonic == "IC")
      add_cic(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 2u
             and std::all_of(begin(noncontrol_mnemonic), end(noncontrol_mnemonic), [](char const character) { return character == 'I'; }))
      add_cis(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 2u and noncontrol_mnemonic.front() == 'I')
      add_cin(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "H")
      add_ch(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "NOT")
      add_cnot(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "X")
      add_cx(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 2u
             and std::all_of(begin(noncontrol_mnemonic), end(noncontrol_mnemonic), [](char const character) { return character == 'X'; }))
      add_cxs(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 2u and noncontrol_mnemonic.front() == 'X')
      add_cxn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "Y")
      add_cy(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 2u
             and std::all_of(begin(noncontrol_mnemonic), end(noncontrol_mnemonic), [](char const character) { return character == 'Y'; }))
      add_cys(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 2u and noncontrol_mnemonic.front() == 'Y')
      add_cyn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "Z")
      add_cz(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 2u
             and std::all_of(begin(noncontrol_mnemonic), end(noncontrol_mnemonic), [](char const character) { return character == 'Z'; }))
      add_czs(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 2u and noncontrol_mnemonic.front() == 'Z')
      add_czn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "SWAP")
      add_cswap(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "S")
      add_cs(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "S+")
      add_adj_cs(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "T")
      add_ct(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "T+")
      add_adj_ct(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "U1")
      add_cu1(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "U1+")
      add_adj_cu1(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "U2")
      add_cu2(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "U2+")
      add_adj_cu2(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "U3")
      add_cu3(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "U3+")
      add_adj_cu3(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "R")
      add_cr(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "R+")
      add_adj_cr(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "+X")
      add_crotx(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "-X")
      add_adj_crotx(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "+Y")
      add_croty(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "-Y")
      add_adj_croty(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "EX")
      add_cex(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "EX+")
      add_adj_cex(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 3u
             and noncontrol_mnemonic.front() == 'E'
             and std::all_of(std::next(begin(noncontrol_mnemonic)), end(noncontrol_mnemonic), [](char const character) { return character == 'X'; }))
      add_cexs(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 4u
             and noncontrol_mnemonic.front() == 'E'
             and std::all_of(std::next(begin(noncontrol_mnemonic)), std::prev(end(noncontrol_mnemonic)), [](char const character) { return character == 'X'; })
             and noncontrol_mnemonic.back() == '+')
      add_adj_cexs(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 4u and noncontrol_mnemonic[0u] == 'E' and noncontrol_mnemonic[1u] == 'X' and noncontrol_mnemonic.back() == '+')
      add_adj_cexn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic.size() >= 3u and noncontrol_mnemonic[0u] == 'E' and noncontrol_mnemonic[1u] == 'X')
      add_cexn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "EY")
      add_cey(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "EY+")
      add_adj_cey(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 3u
             and noncontrol_mnemonic.front() == 'E'
             and std::all_of(std::next(begin(noncontrol_mnemonic)), end(noncontrol_mnemonic), [](char const character) { return character == 'Y'; }))
      add_ceys(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 4u
             and noncontrol_mnemonic.front() == 'E'
             and std::all_of(std::next(begin(noncontrol_mnemonic)), std::prev(end(noncontrol_mnemonic)), [](char const character) { return character == 'Y'; })
             and noncontrol_mnemonic.back() == '+')
      add_adj_ceys(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 4u and noncontrol_mnemonic[0u] == 'E' and noncontrol_mnemonic[1u] == 'Y' and noncontrol_mnemonic.back() == '+')
      add_adj_ceyn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic.size() >= 3u and noncontrol_mnemonic[0u] == 'E' and noncontrol_mnemonic[1u] == 'Y')
      add_ceyn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "EZ")
      add_cez(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "EZ+")
      add_adj_cez(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 3u
             and noncontrol_mnemonic.front() == 'E'
             and std::all_of(std::next(begin(noncontrol_mnemonic)), end(noncontrol_mnemonic), [](char const character) { return character == 'Z'; }))
      add_cezs(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 4u
             and noncontrol_mnemonic.front() == 'E'
             and std::all_of(std::next(begin(noncontrol_mnemonic)), std::prev(end(noncontrol_mnemonic)), [](char const character) { return character == 'Z'; })
             and noncontrol_mnemonic.back() == '+')
      add_adj_cezs(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 4u and noncontrol_mnemonic[0u] == 'E' and noncontrol_mnemonic[1u] == 'Z' and noncontrol_mnemonic.back() == '+')
      add_adj_cezn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic.size() >= 3u and noncontrol_mnemonic[0u] == 'E' and noncontrol_mnemonic[1u] == 'Z')
      add_cezn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "ESWAP")
      add_ceswap(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "ESWAP+")
      add_adj_ceswap(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "SX")
      add_csx(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "SX+")
      add_adj_csx(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "SY")
      add_csy(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "SY+")
      add_adj_csy(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "SZ")
      add_csz(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "SZ+")
      add_adj_csz(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 3u
             and noncontrol_mnemonic.front() == 'S'
             and std::all_of(std::next(begin(noncontrol_mnemonic)), end(noncontrol_mnemonic), [](char const character) { return character == 'Z'; }))
      add_cszs(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 4u
             and noncontrol_mnemonic.front() == 'S'
             and std::all_of(std::next(begin(noncontrol_mnemonic)), std::prev(end(noncontrol_mnemonic)), [](char const character) { return character == 'Z'; })
             and noncontrol_mnemonic.back() == '+')
      add_adj_cszs(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 4u and noncontrol_mnemonic[0u] == 'S' and noncontrol_mnemonic[1u] == 'Z' and noncontrol_mnemonic.back() == '+')
      add_adj_cszn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic.size() >= 3u and noncontrol_mnemonic[0u] == 'S' and noncontrol_mnemonic[1u] == 'Z')
      add_cszn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_ci(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_i_gate >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(1u);
      read_multi_controls_multi_targets(columns, controls, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_in_gate >(std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_cic(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      std::tie(control1, control2) = read_2controls(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_ic_gate >(control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
      read_multi_controls(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_ic_gate >(std::move(controls)));
    }
  }

  void interpreter::add_cis(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size());
    read_multi_controls_multi_targets(columns, controls, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_in_gate >(std::move(targets), std::move(controls)));
  }

  void interpreter::add_cin(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = end(noncontrol_mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_control_qubits == 1 and num_target_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_i_gate >(target, control));
    }
    else if (num_target_qubits >= 1)
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      read_multi_controls_multi_targets(columns, controls, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_in_gate >(std::move(targets), std::move(controls)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_ch(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_hadamard >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_hadamard >(target, std::move(controls)));
    }
  }

  void interpreter::add_cnot(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_not >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto target = read_multi_controls_target(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_not >(target, std::move(controls)));
    }
  }

  void interpreter::add_cx(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_pauli_x >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(1u);
      read_multi_controls_multi_targets(columns, controls, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_pauli_xn >(std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_cxs(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size());
    read_multi_controls_multi_targets(columns, controls, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_pauli_xn >(std::move(targets), std::move(controls)));
  }

  void interpreter::add_cxn(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = end(noncontrol_mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_control_qubits == 1 and num_target_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_pauli_x >(target, control));
    }
    else if (num_target_qubits >= 1)
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      read_multi_controls_multi_targets(columns, controls, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_pauli_xn >(std::move(targets), std::move(controls)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_cy(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_pauli_y >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(1u);
      read_multi_controls_multi_targets(columns, controls, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_pauli_yn >(std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_cys(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size());
    read_multi_controls_multi_targets(columns, controls, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_pauli_yn >(std::move(targets), std::move(controls)));
  }

  void interpreter::add_cyn(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = end(noncontrol_mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_control_qubits == 1 and num_target_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_pauli_y >(target, control));
    }
    else if (num_target_qubits >= 1)
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      read_multi_controls_multi_targets(columns, controls, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_pauli_yn >(std::move(targets), std::move(controls)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_cz(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      std::tie(control1, control2) = read_2controls(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_pauli_z >(control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
      read_multi_controls(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_pauli_z >(std::move(controls)));
    }
  }

  void interpreter::add_czs(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size());
    read_multi_controls_multi_targets(columns, controls, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_pauli_zn >(std::move(targets), std::move(controls)));
  }

  void interpreter::add_czn(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = end(noncontrol_mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_target_qubits == 1)
    {
      if (num_control_qubits == 1)
      {
        auto control1 = ::bra::control_qubit_type{};
        auto control2 = ::bra::control_qubit_type{};
        std::tie(control1, control2) = read_2controls(columns);

        circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_pauli_z >(control1, control2));
      }
      else
      {
        auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
        read_multi_controls(columns, controls);

        circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_pauli_z >(std::move(controls)));
      }
    }
    else if (num_target_qubits >= 2)
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      read_multi_controls_multi_targets(columns, controls, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_pauli_zn >(std::move(targets), std::move(controls)));
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void interpreter::add_cswap(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    std::tie(target1, target2) = read_multi_controls_2targets(columns, controls);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_swap >(target1, target2, std::move(controls)));
  }

  void interpreter::add_cs(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      std::tie(control1, control2) = read_2controls(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_s_gate >(control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
      read_multi_controls(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_s_gate >(std::move(controls)));
    }
  }

  void interpreter::add_adj_cs(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      std::tie(control1, control2) = read_2controls(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_s_gate >(control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
      read_multi_controls(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_s_gate >(std::move(controls)));
    }
  }

  void interpreter::add_ct(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      std::tie(control1, control2) = read_2controls(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_t_gate >(control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
      read_multi_controls(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_t_gate >(std::move(controls)));
    }
  }

  void interpreter::add_adj_ct(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      std::tie(control1, control2) = read_2controls(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_t_gate >(control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
      read_multi_controls(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_t_gate >(std::move(controls)));
    }
  }

  void interpreter::add_cu1(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control1, control2) = read_2controls_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_u1 >(phase, control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_phase(columns, controls, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_u1 >(phase, std::move(controls)));
    }
  }

  void interpreter::add_adj_cu1(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control1, control2) = read_2controls_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_u1 >(phase, control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_phase(columns, controls, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_u1 >(phase, std::move(controls)));
    }
  }

  void interpreter::add_cu2(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase1 = boost::variant<real_type, std::string>{};
      auto phase2 = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_2phases(columns, phase1, phase2);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_u2 >(phase1, phase2, target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto phase1 = boost::variant<real_type, std::string>{};
      auto phase2 = boost::variant<real_type, std::string>{};
      auto const target = read_multi_controls_target_2phases(columns, controls, phase1, phase2);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_u2 >(phase1, phase2, target, std::move(controls)));
    }
  }

  void interpreter::add_adj_cu2(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase1 = boost::variant<real_type, std::string>{};
      auto phase2 = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_2phases(columns, phase1, phase2);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_u2 >(phase1, phase2, target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto phase1 = boost::variant<real_type, std::string>{};
      auto phase2 = boost::variant<real_type, std::string>{};
      auto const target = read_multi_controls_target_2phases(columns, controls, phase1, phase2);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_u2 >(phase1, phase2, target, std::move(controls)));
    }
  }

  void interpreter::add_cu3(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase1 = boost::variant<real_type, std::string>{};
      auto phase2 = boost::variant<real_type, std::string>{};
      auto phase3 = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_3phases(columns, phase1, phase2, phase3);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_u3 >(phase1, phase2, phase3, target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto phase1 = boost::variant<real_type, std::string>{};
      auto phase2 = boost::variant<real_type, std::string>{};
      auto phase3 = boost::variant<real_type, std::string>{};
      auto const target = read_multi_controls_target_3phases(columns, controls, phase1, phase2, phase3);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_u3 >(phase1, phase2, phase3, target, std::move(controls)));
    }
  }

  void interpreter::add_adj_cu3(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase1 = boost::variant<real_type, std::string>{};
      auto phase2 = boost::variant<real_type, std::string>{};
      auto phase3 = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_3phases(columns, phase1, phase2, phase3);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_u3 >(phase1, phase2, phase3, target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto phase1 = boost::variant<real_type, std::string>{};
      auto phase2 = boost::variant<real_type, std::string>{};
      auto phase3 = boost::variant<real_type, std::string>{};
      auto const target = read_multi_controls_target_3phases(columns, controls, phase1, phase2, phase3);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_u3 >(phase1, phase2, phase3, target, std::move(controls)));
    }
  }

  void interpreter::add_cr(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      auto phase_exponent = boost::variant<int_type, std::string>{};
      std::tie(control1, control2) = read_2controls_phaseexp(columns, phase_exponent);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_phase_shift_ >(phase_exponent, control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
      auto phase_exponent = boost::variant<int_type, std::string>{};
      read_multi_controls_phaseexp(columns, controls, phase_exponent);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_phase_shift >(phase_exponent, std::move(controls)));
    }
  }

  void interpreter::add_adj_cr(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      auto phase_exponent = boost::variant<int_type, std::string>{};
      std::tie(control1, control2) = read_2controls_phaseexp(columns, phase_exponent);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_phase_shift >(phase_exponent, control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto phase_exponent = boost::variant<int_type, std::string>{};
      read_multi_controls_phaseexp(columns, controls, phase_exponent);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_phase_shift >(phase_exponent, std::move(controls)));
    }
  }

  void interpreter::add_crotx(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_x_rotation_half_pi >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_x_rotation_half_pi >(target, std::move(controls)));
    }
  }

  void interpreter::add_adj_crotx(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_x_rotation_half_pi >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_x_rotation_half_pi >(target, std::move(controls)));
    }
  }

  void interpreter::add_croty(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_y_rotation_half_pi >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_y_rotation_half_pi >(target, std::move(controls)));
    }
  }

  void interpreter::add_adj_croty(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_y_rotation_half_pi >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_y_rotation_half_pi >(target, std::move(controls)));
    }
  }

  void interpreter::add_cex(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_exponential_pauli_x >(phase, target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(1u);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_exponential_pauli_xn >(phase, std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_adj_cex(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_exponential_pauli_x >(phase, target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(1u);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_exponential_pauli_xn >(phase, std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_cexs(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size() - 1u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_exponential_pauli_xn >(phase, std::move(targets), std::move(controls)));
  }

  void interpreter::add_adj_cexs(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size() - 1u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_exponential_pauli_xn >(phase, std::move(targets), std::move(controls)));
  }

  void interpreter::add_cexn(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = end(noncontrol_mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_target_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    if (num_control_qubits + num_target_qubits == 2)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_exponential_pauli_x >(phase, target, control));
    }
    else // num_control_qubits + num_target_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_exponential_pauli_xn >(phase, std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_adj_cexn(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = std::prev(end(noncontrol_mnemonic));
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_target_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    if (num_control_qubits + num_target_qubits == 2)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_exponential_pauli_x >(phase, target, control));
    }
    else // num_control_qubits + num_target_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_exponential_pauli_xn >(phase, std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_cey(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_exponential_pauli_y >(phase, target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(1u);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_exponential_pauli_yn >(phase, std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_adj_cey(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_exponential_pauli_y >(phase, target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(1u);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_exponential_pauli_yn >(phase, std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_ceys(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size() - 1u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_exponential_pauli_yn >(phase, std::move(targets), std::move(controls)));
  }

  void interpreter::add_adj_ceys(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size() - 1u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_exponential_pauli_yn >(phase, std::move(targets), std::move(controls)));
  }

  void interpreter::add_ceyn(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = end(noncontrol_mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_target_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    if (num_control_qubits + num_target_qubits == 2)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_exponential_pauli_y >(phase, target, control));
    }
    else // num_control_qubits + num_target_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_exponential_pauli_yn >(phase, std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_adj_ceyn(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = std::prev(end(noncontrol_mnemonic));
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_target_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    if (num_control_qubits + num_target_qubits == 2)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_exponential_pauli_y >(phase, target, control));
    }
    else // num_control_qubits + num_target_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_exponential_pauli_yn >(phase, std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_cez(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_exponential_pauli_z >(phase, target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      auto const target = read_multi_controls_target_phase(columns, controls, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_exponential_pauli_z >(phase, target, std::move(controls)));
    }
  }

  void interpreter::add_adj_cez(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_exponential_pauli_z >(phase, target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      auto const target = read_multi_controls_target_phase(columns, controls, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_exponential_pauli_z >(phase, target, std::move(controls)));
    }
  }

  void interpreter::add_cezs(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size() - 1u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_exponential_pauli_zn >(phase, std::move(targets), std::move(controls)));
  }

  void interpreter::add_adj_cezs(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size() - 1u);
    auto phase = boost::variant<real_type, std::string>{};
    read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_exponential_pauli_zn >(phase, std::move(targets), std::move(controls)));
  }

  void interpreter::add_cezn(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = end(noncontrol_mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_target_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    if (num_control_qubits + num_target_qubits == 2)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_exponential_pauli_z >(phase, target, control));
    }
    else if (num_target_qubits == 1)
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      auto const target = read_multi_controls_target_phase(columns, controls, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_exponential_pauli_z >(phase, target, std::move(controls)));
    }
    else // num_control_qubits + num_target_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_exponential_pauli_zn >(phase, std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_adj_cezn(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = std::prev(end(noncontrol_mnemonic));
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_target_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    if (num_control_qubits + num_target_qubits == 2)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      auto phase = boost::variant<real_type, std::string>{};
      std::tie(control, target) = read_control_target_phase(columns, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_exponential_pauli_z >(phase, target, control));
    }
    else if (num_target_qubits == 1)
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      auto const target = read_multi_controls_target_phase(columns, controls, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_exponential_pauli_z >(phase, target, std::move(controls)));
    }
    else // num_control_qubits + num_target_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      auto phase = boost::variant<real_type, std::string>{};
      read_multi_controls_multi_targets_phase(columns, controls, targets, phase);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_exponential_pauli_zn >(phase, std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_ceswap(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    auto phase = boost::variant<real_type, std::string>{};
    std::tie(target1, target2) = read_multi_controls_2targets_phase(columns, controls, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_exponential_swap >(phase, target1, target2, std::move(controls)));
  }

  void interpreter::add_adj_ceswap(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto target1 = ::bra::qubit_type{};
    auto target2 = ::bra::qubit_type{};
    auto phase = boost::variant<real_type, std::string>{};
    std::tie(target1, target2) = read_multi_controls_2targets_phase(columns, controls, phase);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_exponential_swap >(phase, target1, target2, std::move(controls)));
  }

  void interpreter::add_csx(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_sqrt_pauli_x >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_sqrt_pauli_x >(target, std::move(controls)));
    }
  }

  void interpreter::add_adj_csx(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_sqrt_pauli_x >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_sqrt_pauli_x >(target, std::move(controls)));
    }
  }

  void interpreter::add_csy(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_sqrt_pauli_y >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_sqrt_pauli_y >(target, std::move(controls)));
    }
  }

  void interpreter::add_adj_csy(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = ::bra::control_qubit_type{};
      auto target = ::bra::qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_sqrt_pauli_y >(target, control));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_sqrt_pauli_y >(target, std::move(controls)));
    }
  }

  void interpreter::add_csz(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      std::tie(control1, control2) = read_2controls(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_sqrt_pauli_z >(control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
      read_multi_controls(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_sqrt_pauli_z >(std::move(controls)));
    }
  }

  void interpreter::add_adj_csz(interpreter::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control1 = ::bra::control_qubit_type{};
      auto control2 = ::bra::control_qubit_type{};
      std::tie(control1, control2) = read_2controls(columns);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_sqrt_pauli_z >(control1, control2));
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
      read_multi_controls(columns, controls);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_sqrt_pauli_z >(std::move(controls)));
    }
  }

  void interpreter::add_cszs(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size() - 1u);
    read_multi_controls_multi_targets(columns, controls, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_sqrt_pauli_zn >(std::move(targets), std::move(controls)));
  }

  void interpreter::add_adj_cszs(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic)
  {
    auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
    auto targets = std::vector< ::bra::qubit_type >(noncontrol_mnemonic.size() - 1u);
    read_multi_controls_multi_targets(columns, controls, targets);

    circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_sqrt_pauli_zn >(std::move(targets), std::move(controls)));
  }

  void interpreter::add_cszn(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = end(noncontrol_mnemonic);
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_target_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    if (num_target_qubits == 1)
    {
      if (num_control_qubits == 1)
      {
        auto control1 = ::bra::control_qubit_type{};
        auto control2 = ::bra::control_qubit_type{};
        std::tie(control1, control2) = read_2controls(columns);

        circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::controlled_sqrt_pauli_z >(control1, control2));
      }
      else // num_control_qubits >= 2
      {
        auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
        read_multi_controls(columns, controls);

        circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_sqrt_pauli_z >(std::move(controls)));
      }
    }
    else
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      read_multi_controls_multi_targets(columns, controls, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::multi_controlled_sqrt_pauli_zn >(std::move(targets), std::move(controls)));
    }
  }

  void interpreter::add_adj_cszn(
    interpreter::columns_type const& columns, int const num_control_qubits,
    std::string const& noncontrol_mnemonic, std::string const& mnemonic)
  {
    using std::begin;
    using std::end;
    auto const possible_digits_first = std::next(begin(noncontrol_mnemonic));
    auto const possible_digits_last = std::prev(end(noncontrol_mnemonic));
    if (not std::all_of(
              possible_digits_first, possible_digits_last,
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_first, possible_digits_last);
    if (num_target_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    if (num_target_qubits == 1)
    {
      if (num_control_qubits == 1)
      {
        auto control1 = ::bra::control_qubit_type{};
        auto control2 = ::bra::control_qubit_type{};
        std::tie(control1, control2) = read_2controls(columns);

        circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_controlled_sqrt_pauli_z >(control1, control2));
      }
      else // num_control_qubits >= 2
      {
        auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits + 1u);
        read_multi_controls(columns, controls);

        circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_sqrt_pauli_z >(std::move(controls)));
      }
    }
    else
    {
      auto controls = std::vector< ::bra::control_qubit_type >(num_control_qubits);
      auto targets = std::vector< ::bra::qubit_type >(num_target_qubits);
      read_multi_controls_multi_targets(columns, controls, targets);

      circuits_[circuit_index_].push_back(std::make_unique< ::bra::gate::adj_multi_controlled_sqrt_pauli_zn >(std::move(targets), std::move(controls)));
    }
  }
} // namespace bra


# undef BRA_is_nothrow_swappable
