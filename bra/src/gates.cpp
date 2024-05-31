#include <istream>
#include <string>
#include <vector>
#include <tuple>
#include <utility>
#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
# if __cplusplus >= 201703L
#   include <type_traits>
# else
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

#include <boost/lexical_cast.hpp>

#include <boost/range/empty.hpp>
#include <boost/range/size.hpp>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <boost/utility/string_view.hpp>
#include <boost/core/pointer_traits.hpp>

#ifndef BRA_NO_MPI
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
#endif // BRA_NO_MPI

#include <ket/qubit.hpp>
#include <ket/control.hpp>
#include <ket/utility/integer_log2.hpp>
#include <ket/utility/integer_exp2.hpp>
#include <ket/utility/generate_phase_coefficients.hpp>

#include <bra/gates.hpp>
#include <bra/state.hpp>
#include <bra/utility/to_integer.hpp>
#include <bra/gate/gate.hpp>
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
#include <bra/gate/s_gate.hpp>
#include <bra/gate/adj_s_gate.hpp>
#include <bra/gate/t_gate.hpp>
#include <bra/gate/adj_t_gate.hpp>
#include <bra/gate/u1.hpp>
#include <bra/gate/u2.hpp>
#include <bra/gate/u3.hpp>
#include <bra/gate/phase_shift.hpp>
#include <bra/gate/adj_phase_shift.hpp>
#include <bra/gate/x_rotation_half_pi.hpp>
#include <bra/gate/adj_x_rotation_half_pi.hpp>
#include <bra/gate/y_rotation_half_pi.hpp>
#include <bra/gate/adj_y_rotation_half_pi.hpp>
#include <bra/gate/controlled_phase_shift.hpp>
#include <bra/gate/adj_controlled_phase_shift.hpp>
#include <bra/gate/controlled_v.hpp>
#include <bra/gate/adj_controlled_v.hpp>
#include <bra/gate/exponential_pauli_x.hpp>
#include <bra/gate/exponential_pauli_xx.hpp>
#include <bra/gate/exponential_pauli_xn.hpp>
#include <bra/gate/exponential_pauli_y.hpp>
#include <bra/gate/exponential_pauli_yy.hpp>
#include <bra/gate/exponential_pauli_yn.hpp>
#include <bra/gate/exponential_pauli_z.hpp>
#include <bra/gate/exponential_pauli_zz.hpp>
#include <bra/gate/exponential_pauli_zn.hpp>
#include <bra/gate/exponential_swap.hpp>
#include <bra/gate/toffoli.hpp>
#include <bra/gate/projective_measurement.hpp>
#include <bra/gate/measurement.hpp>
#include <bra/gate/generate_events.hpp>
#include <bra/gate/shor_box.hpp>
#include <bra/gate/clear.hpp>
#include <bra/gate/set.hpp>
#include <bra/gate/depolarizing_channel.hpp>
#include <bra/gate/exit.hpp>
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
#include <bra/gate/multi_controlled_swap.hpp>
#include <bra/gate/controlled_s_gate.hpp>
#include <bra/gate/multi_controlled_s_gate.hpp>
#include <bra/gate/adj_controlled_s_gate.hpp>
#include <bra/gate/adj_multi_controlled_s_gate.hpp>
#include <bra/gate/controlled_t_gate.hpp>
#include <bra/gate/multi_controlled_t_gate.hpp>
#include <bra/gate/adj_controlled_t_gate.hpp>
#include <bra/gate/adj_multi_controlled_t_gate.hpp>
#include <bra/gate/controlled_u1.hpp>
#include <bra/gate/multi_controlled_u1.hpp>
#include <bra/gate/controlled_u2.hpp>
#include <bra/gate/multi_controlled_u2.hpp>
#include <bra/gate/controlled_u3.hpp>
#include <bra/gate/multi_controlled_u3.hpp>
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
#include <bra/gate/controlled_phase_shift_cu.hpp>
#include <bra/gate/adj_controlled_phase_shift_cu.hpp>
#include <bra/gate/controlled_v_.hpp>
#include <bra/gate/adj_controlled_v_.hpp>
#include <bra/gate/multi_controlled_v.hpp>
#include <bra/gate/adj_multi_controlled_v.hpp>
#include <bra/gate/controlled_exponential_pauli_x.hpp>
#include <bra/gate/multi_controlled_exponential_pauli_xn.hpp>
#include <bra/gate/controlled_exponential_pauli_y.hpp>
#include <bra/gate/multi_controlled_exponential_pauli_yn.hpp>
#include <bra/gate/controlled_exponential_pauli_z.hpp>
#include <bra/gate/multi_controlled_exponential_pauli_zn.hpp>
#include <bra/gate/multi_controlled_exponential_swap.hpp>

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

  wrong_mnemonics_error::wrong_mnemonics_error(::bra::gates::columns_type const& columns)
    : std::runtime_error{generate_what_string(columns).c_str()}
  { }

  std::string wrong_mnemonics_error::generate_what_string(::bra::gates::columns_type const& columns)
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
  gates::gates()
    : data_{}, num_qubits_{}, num_lqubits_{}, num_uqubits_{}, num_processes_per_unit_{1u},
      initial_state_value_{}, initial_permutation_{}, phase_coefficients_{}, root_{}
  { }

  gates::gates(gates::allocator_type const& allocator)
    : data_{allocator}, num_qubits_{}, num_lqubits_{}, num_uqubits_{}, num_processes_per_unit_{1u},
      initial_state_value_{}, initial_permutation_{}, phase_coefficients_{}, root_{}
  { }

  gates::gates(gates&& other, gates::allocator_type const& allocator)
      : data_{std::move(other.data_), allocator},
        num_qubits_{std::move(other.num_qubits_)},
        num_lqubits_{std::move(other.num_lqubits_)},
        num_uqubits_{std::move(other.num_uqubits_)},
        num_processes_per_unit_{std::move(other.num_processes_per_unit_)},
        initial_state_value_{std::move(other.initial_state_value_)},
        initial_permutation_{std::move(other.initial_permutation_)},
        phase_coefficients_{std::move(other.phase_coefficients_)},
        root_{std::move(other.root_)}
  { }
#else // BRA_NO_MPI
  gates::gates()
    : data_{}, num_qubits_{},
      initial_state_value_{}, phase_coefficients_{}
  { }

  gates::gates(gates::allocator_type const& allocator)
    : data_{allocator}, num_qubits_{},
      initial_state_value_{}, phase_coefficients_{}
  { }

  gates::gates(gates&& other, gates::allocator_type const& allocator)
      : data_{std::move(other.data_), allocator},
        num_qubits_{std::move(other.num_qubits_)},
        initial_state_value_{std::move(other.initial_state_value_)},
        phase_coefficients_{std::move(other.phase_coefficients_)}
  { }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  gates::gates(
    std::istream& input_stream,
    bit_integer_type num_uqubits, unsigned int num_processes_per_unit,
    yampi::environment const& environment,
    yampi::rank const root, yampi::communicator const& communicator,
    size_type const num_reserved_gates)
    : data_{}, num_qubits_{}, num_lqubits_{},
      num_uqubits_{num_uqubits}, num_processes_per_unit_{num_processes_per_unit},
      initial_state_value_{}, initial_permutation_{}, phase_coefficients_{}, root_{root}
  {
    assert(num_processes_per_unit >= 1u);
    assign(input_stream, environment, communicator, num_reserved_gates);
  }
#else // BRA_NO_MPI
  gates::gates(std::istream& input_stream)
    : data_{}, num_qubits_{},
      initial_state_value_{}, phase_coefficients_{}
  { assign(input_stream, size_type{0u}); }

  gates::gates(std::istream& input_stream, size_type const num_reserved_gates)
    : data_{}, num_qubits_{},
      initial_state_value_{}, phase_coefficients_{}
  { assign(input_stream, num_reserved_gates); }
#endif // BRA_NO_MPI

  bool gates::operator==(gates const& other) const
  {
#ifndef BRA_NO_MPI
    return data_ == other.data_
      and num_qubits_ == other.num_qubits_
      and num_lqubits_ == other.num_lqubits_
      and num_uqubits_ == other.num_uqubits_
      and num_processes_per_unit_ == other.num_processes_per_unit_
      and initial_state_value_ == other.initial_state_value_
      and initial_permutation_ == other.initial_permutation_
      and phase_coefficients_ == other.phase_coefficients_
      and root_ == other.root_;
#else // BRA_NO_MPI
    return data_ == other.data_
      and num_qubits_ == other.num_qubits_
      and initial_state_value_ == other.initial_state_value_
      and phase_coefficients_ == other.phase_coefficients_;
#endif // BRA_NO_MPI
  }

#ifndef BRA_NO_MPI
  void gates::num_qubits(
    bit_integer_type const new_num_qubits,
    yampi::communicator const& communicator, yampi::environment const& environment)
  {
    auto const num_gqubits
      = ket::utility::integer_log2<bit_integer_type>(
          communicator.size(environment) / num_processes_per_unit_);
    set_num_qubits_params(new_num_qubits - num_gqubits - num_uqubits_, num_gqubits, communicator, environment);
  }

  void gates::num_lqubits(
    bit_integer_type const new_num_lqubits,
    yampi::communicator const& communicator, yampi::environment const& environment)
  {
    set_num_qubits_params(
      new_num_lqubits,
      ket::utility::integer_log2<bit_integer_type>(
        communicator.size(environment) / num_processes_per_unit_),
      communicator, environment);
  }
#else // BRA_NO_MPI
  void gates::num_qubits(bit_integer_type const new_num_qubits)
  { set_num_qubits_params(new_num_qubits); }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  void gates::set_num_qubits_params(
    bit_integer_type const new_num_lqubits, bit_integer_type const num_gqubits,
    yampi::communicator const& communicator, yampi::environment const& environment)
  {
    if (ket::utility::integer_exp2<bit_integer_type>(num_gqubits) * num_processes_per_unit_
        != static_cast<bit_integer_type>(communicator.size(environment)))
      throw wrong_mpi_communicator_size_error{};

    num_lqubits_ = new_num_lqubits;
    num_qubits_ = new_num_lqubits + num_uqubits_ + num_gqubits;
    ket::utility::generate_phase_coefficients(phase_coefficients_, num_qubits_);

    initial_permutation_.clear();
    initial_permutation_.reserve(num_qubits_);
    for (auto bit = bit_integer_type{0u}; bit < num_qubits_; ++bit)
      initial_permutation_.push_back(permutated_qubit_type{bit});
  }
#else // BRA_NO_MPI
  void gates::set_num_qubits_params(bit_integer_type const new_num_qubits)
  {
    num_qubits_ = new_num_qubits;
    ket::utility::generate_phase_coefficients(phase_coefficients_, num_qubits_);
  }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  void gates::assign(
    std::istream& input_stream, yampi::environment const& environment,
    yampi::communicator const& communicator, size_type const num_reserved_gates)
#else // BRA_NO_MPI
  void gates::assign(std::istream& input_stream, size_type const num_reserved_gates)
#endif // BRA_NO_MPI
  {
    data_.clear();
    data_.reserve(num_reserved_gates);

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
      if (mnemonic == "QUBITS")
      {
#ifndef BRA_NO_MPI
        num_qubits(
          static_cast< ::bra::state::bit_integer_type >(read_num_qubits(columns)),
          communicator, environment);
#else // BRA_NO_MPI
        num_qubits(
          static_cast< ::bra::state::bit_integer_type >(read_num_qubits(columns)));
#endif // BRA_NO_MPI
      }
      else if (mnemonic == "INITIAL") // INITIAL STATE
        initial_state_value_
          = static_cast< ::bra::state::state_integer_type >(read_initial_state_value(columns));
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
      else if (mnemonic == "I")
        continue;
      else if (mnemonic == "H")
        add_h(columns);
      else if (mnemonic == "NOT")
        add_not(columns);
      else if (mnemonic == "X")
        add_x(columns);
      else if (mnemonic == "XX")
        add_xx(columns);
      else if (mnemonic.size() >= 3u
               and std::all_of(
                     std::begin(mnemonic), std::end(mnemonic),
                     [](char const character) { return character == 'X'; }))
        add_xs(columns, mnemonic);
      else if (mnemonic.size() >= 2u and mnemonic.front() == 'X')
        add_xn(columns, mnemonic);
      else if (mnemonic == "Y")
        add_y(columns);
      else if (mnemonic == "YY")
        add_yy(columns);
      else if (mnemonic.size() >= 3u
               and std::all_of(
                     std::begin(mnemonic), std::end(mnemonic),
                     [](char const character) { return character == 'Y'; }))
        add_ys(columns, mnemonic);
      else if (mnemonic.size() >= 2u and mnemonic.front() == 'Y')
        add_yn(columns, mnemonic);
      else if (mnemonic == "Z")
        add_z(columns);
      else if (mnemonic == "ZZ")
        add_zz(columns);
      else if (mnemonic.size() >= 3u
               and std::all_of(
                     std::begin(mnemonic), std::end(mnemonic),
                     [](char const character) { return character == 'Z'; }))
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
      else if (mnemonic == "U2")
        add_u2(columns);
      else if (mnemonic == "U3")
        add_u3(columns);
      else if (mnemonic == "R" or mnemonic == "+R")
        add_r(columns);
      else if (mnemonic == "-R")
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
      else if (mnemonic == "V")
        add_v(columns);
      else if (mnemonic == "EX")
        add_ex(columns);
      else if (mnemonic == "EXX")
        add_exx(columns);
      else if (mnemonic.size() >= 4u and mnemonic.front() == 'E'
               and std::all_of(
                     std::next(std::begin(mnemonic)), std::end(mnemonic),
                     [](char const character) { return character == 'X'; }))
        add_exs(columns, mnemonic);
      else if (mnemonic.size() >= 3u and mnemonic[0] == 'E' and mnemonic[1] == 'X')
        add_exn(columns, mnemonic);
      else if (mnemonic == "EY")
        add_ey(columns);
      else if (mnemonic == "EYY")
        add_eyy(columns);
      else if (mnemonic.size() >= 4u and mnemonic.front() == 'E'
               and std::all_of(
                     std::next(std::begin(mnemonic)), std::end(mnemonic),
                     [](char const character) { return character == 'Y'; }))
        add_eys(columns, mnemonic);
      else if (mnemonic.size() >= 3u and mnemonic[0] == 'E' and mnemonic[1] == 'Y')
        add_eyn(columns, mnemonic);
      else if (mnemonic == "EZ")
        add_ez(columns);
      else if (mnemonic == "EZZ")
        add_ezz(columns);
      else if (mnemonic.size() >= 4u and mnemonic.front() == 'E'
               and std::all_of(
                     std::next(std::begin(mnemonic)), std::end(mnemonic),
                     [](char const character) { return character == 'Z'; }))
        add_ezs(columns, mnemonic);
      else if (mnemonic.size() >= 3u and mnemonic[0] == 'E' and mnemonic[1] == 'Z')
        add_ezn(columns, mnemonic);
      else if (mnemonic == "ESWAP")
        add_eswap(columns);
      else if (mnemonic == "TOFFOLI")
        add_toffoli(columns);
      else if (mnemonic == "M")
        add_m(columns);
      else if (mnemonic == "SHORBOX")
        add_shor_box(columns);
      else if (mnemonic == "BEGIN") // BEGIN MEASUREMENT/LEARNING MACHINE
      {
        if (columns.size() <= 1u)
          throw wrong_mnemonics_error{columns};

        std::for_each(
          std::next(std::begin(columns)), std::end(columns),
          [](std::string& str) { boost::algorithm::to_upper(str); });

        auto const statement = read_begin_statement(columns);

        if (statement == ::bra::begin_statement::measurement)
        {
#ifndef BRA_NO_MPI
          data_.push_back(
            std::unique_ptr< ::bra::gate::gate >{new ::bra::gate::measurement{root_}});
#else // BRA_NO_MPI
          data_.push_back(
            std::unique_ptr< ::bra::gate::gate >{new ::bra::gate::measurement{}});
#endif // BRA_NO_MPI
        }
        else if (statement == ::bra::begin_statement::learning_machine)
          throw unsupported_mnemonic_error{mnemonic};
      }
      else if (mnemonic == "DO") // DO MEASUREMENT
      {
        /*
        auto const statement = read_do_statement(columns);

        if (statement == do_statement::error)
          throw wrong_mnemonics_error{columns};
        else if (statement == do_statement::measurement)
          throw unsupported_mnemonic_error{mnemonic};
          */
        throw unsupported_mnemonic_error{mnemonic};
      }
      else if (mnemonic == "END") // END MEASUREMENT/LEARNING MACHINE
      {
        /*
        auto const statement = read_end_statement(columns);

        if (statement == ::bra::end_statement::measurement)
        {
#ifndef BRA_NO_MPI
          data_.push_back(std::make_unique< ::bra::gate::measurement >(root_));
#else // BRA_NO_MPI
          data_.push_back(std::make_unique< ::bra::gate::measurement >());
#endif // BRA_NO_MPI
        }
        else if (statement == ::bra::end_statement::learning_machine)
          throw unsupported_mnemonic_error{mnemonic};
*/
        throw unsupported_mnemonic_error{mnemonic};
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
          data_.push_back(
            std::unique_ptr< ::bra::gate::gate >{
              new ::bra::gate::generate_events{root_, num_events, seed}});
#else // BRA_NO_MPI
          data_.push_back(
            std::unique_ptr< ::bra::gate::gate >{
              new ::bra::gate::generate_events{num_events, seed}});
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
      else if (mnemonic == "EXIT")
      {
        if (boost::size(columns) != 1u)
          throw wrong_mnemonics_error{columns};

#ifndef BRA_NO_MPI
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{new ::bra::gate::exit{root_}});
#else // BRA_NO_MPI
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{new ::bra::gate::exit{}});
#endif // BRA_NO_MPI
        break;
      }
      else if (mnemonic.size() >= 2u and mnemonic.front() == 'C') // controlled gates
        interpret_controlled_gates(columns, mnemonic);
      else
        throw unsupported_mnemonic_error{mnemonic};
    }
  }

  void gates::swap(gates& other)
    noexcept(
      BRA_is_nothrow_swappable<data_type>::value
      and BRA_is_nothrow_swappable<bit_integer_type>::value
      and BRA_is_nothrow_swappable<state_integer_type>::value
      and BRA_is_nothrow_swappable<qubit_type>::value)
  {
    using std::swap;
#ifndef BRA_NO_MPI
    swap(data_, other.data_);
    swap(num_qubits_, other.num_qubits_);
    swap(num_lqubits_, other.num_lqubits_);
    swap(initial_state_value_, other.initial_state_value_);
    swap(initial_permutation_, other.initial_permutation_);
    swap(phase_coefficients_, other.phase_coefficients_);
    swap(root_, other.root_);
#else // BRA_NO_MPI
    swap(data_, other.data_);
    swap(num_qubits_, other.num_qubits_);
    swap(initial_state_value_, other.initial_state_value_);
    swap(phase_coefficients_, other.phase_coefficients_);
#endif // BRA_NO_MPI
  }

  gates::bit_integer_type gates::read_num_qubits(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw ::bra::wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    return boost::lexical_cast<bit_integer_type>(*++iter);
  }

  gates::state_integer_type gates::read_initial_state_value(gates::columns_type& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    boost::algorithm::to_upper(*++iter);
    if (columns[1] != "STATE")
      throw wrong_mnemonics_error{columns};

    return boost::lexical_cast<state_integer_type>(*++iter);
  }

  gates::bit_integer_type gates::read_num_mpi_processes(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw ::bra::wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    return boost::lexical_cast<bit_integer_type>(*++iter);
  }

  gates::state_integer_type gates::read_mpi_buffer_size(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw ::bra::wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    return boost::lexical_cast<state_integer_type>(*++iter);
  }

#ifndef BRA_NO_MPI
  std::vector<gates::permutated_qubit_type>
  gates::read_initial_permutation(gates::columns_type const& columns) const
  {
    auto result = std::vector<permutated_qubit_type>{};
    result.reserve(boost::size(columns)-2u);

    auto iter = std::begin(columns);
    ++iter;
    ++iter;

    auto const last = std::end(columns);
    for (; iter != last; ++iter)
      result.push_back(static_cast<permutated_qubit_type>(boost::lexical_cast<bit_integer_type>(*iter)));

    return result;
  }
#endif

  gates::qubit_type gates::read_target(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const target = boost::lexical_cast<bit_integer_type>(*++iter);

    return ket::make_qubit<state_integer_type>(target);
  }

  std::tuple<gates::qubit_type, gates::qubit_type> gates::read_2targets(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const target1 = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const target2 = boost::lexical_cast<bit_integer_type>(*++iter);

    return std::make_tuple(ket::make_qubit<state_integer_type>(target1), ket::make_qubit<state_integer_type>(target2));
  }

  void gates::read_multi_targets(gates::columns_type const& columns, std::vector<gates::qubit_type>& targets) const
  {
    if (boost::size(columns) < 4u or targets.size() != boost::size(columns) - 1u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    ++iter;
    auto const targets_last = std::end(targets);
    for (auto targets_iter = std::begin(targets); targets_iter != targets_last; ++targets_iter, ++iter)
    {
      auto const target = boost::lexical_cast<bit_integer_type>(*iter);
      *targets_iter = ket::make_qubit<state_integer_type>(target);
    }
  }

  std::tuple<gates::qubit_type, gates::real_type>
  gates::read_target_phase(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const target = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const phase = boost::lexical_cast<real_type>(*++iter);

    return std::make_tuple(ket::make_qubit<state_integer_type>(target), phase);
  }

  std::tuple<gates::qubit_type, gates::real_type, gates::real_type>
  gates::read_target_2phases(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const target = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const phase1 = boost::lexical_cast<real_type>(*++iter);
    auto const phase2 = boost::lexical_cast<real_type>(*++iter);

    return std::make_tuple(ket::make_qubit<state_integer_type>(target), phase1, phase2);
  }

  std::tuple<gates::qubit_type, gates::real_type, gates::real_type, gates::real_type>
  gates::read_target_3phases(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 5u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const target = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const phase1 = boost::lexical_cast<real_type>(*++iter);
    auto const phase2 = boost::lexical_cast<real_type>(*++iter);
    auto const phase3 = boost::lexical_cast<real_type>(*++iter);

    return std::make_tuple(ket::make_qubit<state_integer_type>(target), phase1, phase2, phase3);
  }

  std::tuple<gates::qubit_type, int> gates::read_target_phaseexp(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const target = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const phase_exponent = boost::lexical_cast<int>(*++iter);

    return std::make_tuple(ket::make_qubit<state_integer_type>(target), phase_exponent);
  }

  std::tuple<gates::qubit_type, gates::qubit_type, gates::real_type>
  gates::read_2targets_phase(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const target1 = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const target2 = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const phase = boost::lexical_cast<real_type>(*++iter);

    return std::make_tuple(ket::make_qubit<state_integer_type>(target1), ket::make_qubit<state_integer_type>(target2), phase);
  }

  gates::real_type gates::read_multi_targets_phase(gates::columns_type const& columns, std::vector<gates::qubit_type>& targets) const
  {
    if (boost::size(columns) < 5u or targets.size() == boost::size(columns) - 2u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    ++iter;
    auto const targets_last = std::end(targets);
    for (auto targets_iter = std::begin(targets); targets_iter != targets_last; ++targets_iter, ++iter)
    {
      auto const target = boost::lexical_cast<bit_integer_type>(*iter);
      *targets_iter = ket::make_qubit<state_integer_type>(target);
    }
    auto const phase = boost::lexical_cast<real_type>(*iter);

    return phase;
  }

  std::tuple<gates::control_qubit_type, gates::qubit_type> gates::read_control_target(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const control = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const target = boost::lexical_cast<bit_integer_type>(*++iter);

    return std::make_tuple(
      ket::make_control(ket::make_qubit<state_integer_type>(control)),
      ket::make_qubit<state_integer_type>(target));
  }

  std::tuple<gates::control_qubit_type, gates::qubit_type, int>
  gates::read_control_target_phaseexp(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const control = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const target = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const phase_exponent = boost::lexical_cast<int>(*++iter);

    return std::make_tuple(
      ket::make_control(ket::make_qubit<state_integer_type>(control)),
      ket::make_qubit<state_integer_type>(target),
      phase_exponent);
  }

  std::tuple<gates::control_qubit_type, gates::control_qubit_type, gates::qubit_type>
  gates::read_2controls_target(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const control1 = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const control2 = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const target = boost::lexical_cast<bit_integer_type>(*++iter);

    return std::make_tuple(
      ket::make_control(ket::make_qubit<state_integer_type>(control1)),
      ket::make_control(ket::make_qubit<state_integer_type>(control2)),
      ket::make_qubit<state_integer_type>(target));
  }

  gates::qubit_type gates::read_multi_controls_target(
    gates::columns_type const& columns, std::vector<gates::control_qubit_type>& controls) const
  {
    if (boost::size(columns) < 4u or controls.size() != boost::size(columns) - 2u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    ++iter;
    auto const controls_last = std::end(controls);
    for (auto controls_iter = std::begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast<bit_integer_type>(*iter);
      *controls_iter = ket::make_control(ket::make_qubit<state_integer_type>(control));
    }

    auto const target = boost::lexical_cast<bit_integer_type>(*iter);
    return ket::make_qubit<state_integer_type>(target);
  }

  std::tuple<gates::qubit_type, gates::qubit_type>
  gates::read_multi_controls_2targets(
    gates::columns_type const& columns, std::vector<gates::control_qubit_type>& controls) const
  {
    if (boost::size(columns) < 5u or controls.size() != boost::size(columns) - 3u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    ++iter;
    auto const controls_last = std::end(controls);
    for (auto controls_iter = std::begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast<bit_integer_type>(*iter);
      *controls_iter = ket::make_control(ket::make_qubit<state_integer_type>(control));
    }

    auto const target1 = boost::lexical_cast<bit_integer_type>(*iter++);
    auto const target2 = boost::lexical_cast<bit_integer_type>(*iter);
    return std::make_tuple(
       ket::make_qubit<state_integer_type>(target1),
       ket::make_qubit<state_integer_type>(target2));
  }

  void gates::read_multi_controls_multi_targets(
    gates::columns_type const& columns, std::vector<gates::control_qubit_type>& controls, std::vector<gates::qubit_type>& targets) const
  {
    if (boost::size(columns) < 4u or controls.size() + targets.size() != boost::size(columns) - 1u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    ++iter;
    auto const controls_last = std::end(controls);
    for (auto controls_iter = std::begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast<bit_integer_type>(*iter);
      *controls_iter = ket::make_control(ket::make_qubit<state_integer_type>(control));
    }

    auto const targets_last = std::end(targets);
    for (auto targets_iter = std::begin(targets); targets_iter != targets_last; ++targets_iter, ++iter)
    {
      auto const target = boost::lexical_cast<bit_integer_type>(*iter);
      *targets_iter = ket::make_qubit<state_integer_type>(target);
    }
  }

  std::tuple<gates::control_qubit_type, gates::qubit_type, gates::real_type>
  gates::read_control_target_phase(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const control = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const target = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const phase = boost::lexical_cast<real_type>(*++iter);

    return std::make_tuple(
      ket::make_control(ket::make_qubit<state_integer_type>(control)),
      ket::make_qubit<state_integer_type>(target),
      phase);
  }

  std::tuple<gates::qubit_type, gates::real_type>
  gates::read_multi_controls_target_phase(
    gates::columns_type const& columns, std::vector<gates::control_qubit_type>& controls) const
  {
    if (boost::size(columns) < 5u or controls.size() != boost::size(columns) - 3u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    ++iter;
    auto const controls_last = std::end(controls);
    for (auto controls_iter = std::begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast<bit_integer_type>(*iter);
      *controls_iter = ket::make_control(ket::make_qubit<state_integer_type>(control));
    }

    auto const target = boost::lexical_cast<bit_integer_type>(*iter++);
    auto const phase = boost::lexical_cast<real_type>(*iter);
    return std::make_tuple(ket::make_qubit<state_integer_type>(target), phase);
  }

  std::tuple<gates::control_qubit_type, gates::qubit_type, gates::real_type, gates::real_type>
  gates::read_control_target_2phases(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 5u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const control = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const target = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const phase1 = boost::lexical_cast<real_type>(*++iter);
    auto const phase2 = boost::lexical_cast<real_type>(*++iter);

    return std::make_tuple(
      ket::make_control(ket::make_qubit<state_integer_type>(control)),
      ket::make_qubit<state_integer_type>(target),
      phase1, phase2);
  }

  std::tuple<gates::qubit_type, gates::real_type, gates::real_type>
  gates::read_multi_controls_target_2phases(
    gates::columns_type const& columns, std::vector<gates::control_qubit_type>& controls) const
  {
    if (boost::size(columns) < 6u or controls.size() != boost::size(columns) - 4u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    ++iter;
    auto const controls_last = std::end(controls);
    for (auto controls_iter = std::begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast<bit_integer_type>(*iter);
      *controls_iter = ket::make_control(ket::make_qubit<state_integer_type>(control));
    }

    auto const target = boost::lexical_cast<bit_integer_type>(*iter++);
    auto const phase1 = boost::lexical_cast<real_type>(*iter++);
    auto const phase2 = boost::lexical_cast<real_type>(*iter);
    return std::make_tuple(ket::make_qubit<state_integer_type>(target), phase1, phase2);
  }

  std::tuple<gates::control_qubit_type, gates::qubit_type, gates::real_type, gates::real_type, gates::real_type>
  gates::read_control_target_3phases(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 6u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const control = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const target = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const phase1 = boost::lexical_cast<real_type>(*++iter);
    auto const phase2 = boost::lexical_cast<real_type>(*++iter);
    auto const phase3 = boost::lexical_cast<real_type>(*++iter);

    return std::make_tuple(
      ket::make_control(ket::make_qubit<state_integer_type>(control)),
      ket::make_qubit<state_integer_type>(target),
      phase1, phase2, phase3);
  }

  std::tuple<gates::qubit_type, gates::real_type, gates::real_type, gates::real_type>
  gates::read_multi_controls_target_3phases(
    gates::columns_type const& columns, std::vector<gates::control_qubit_type>& controls) const
  {
    if (boost::size(columns) < 7u or controls.size() != boost::size(columns) - 5u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    ++iter;
    auto const controls_last = std::end(controls);
    for (auto controls_iter = std::begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast<bit_integer_type>(*iter);
      *controls_iter = ket::make_control(ket::make_qubit<state_integer_type>(control));
    }

    auto const target = boost::lexical_cast<bit_integer_type>(*iter++);
    auto const phase1 = boost::lexical_cast<real_type>(*iter++);
    auto const phase2 = boost::lexical_cast<real_type>(*iter++);
    auto const phase3 = boost::lexical_cast<real_type>(*iter);
    return std::make_tuple(ket::make_qubit<state_integer_type>(target), phase1, phase2, phase3);
  }

  std::tuple<gates::qubit_type, int>
  gates::read_multi_controls_target_phaseexp(
    gates::columns_type const& columns, std::vector<gates::control_qubit_type>& controls) const
  {
    if (boost::size(columns) < 5u or controls.size() != boost::size(columns) - 3u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    ++iter;
    auto const controls_last = std::end(controls);
    for (auto controls_iter = std::begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast<bit_integer_type>(*iter);
      *controls_iter = ket::make_control(ket::make_qubit<state_integer_type>(control));
    }

    auto const target = boost::lexical_cast<bit_integer_type>(*iter++);
    auto const phase_exponent = boost::lexical_cast<int>(*iter);
    return std::make_tuple(ket::make_qubit<state_integer_type>(target), phase_exponent);
  }

  gates::real_type gates::read_multi_controls_multi_targets_phase(
    gates::columns_type const& columns,
    std::vector<gates::control_qubit_type>& controls, std::vector<gates::qubit_type>& targets) const
  {
    if (boost::size(columns) < 5u or controls.size() + targets.size() != boost::size(columns) - 2u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    ++iter;
    auto const controls_last = std::end(controls);
    for (auto controls_iter = std::begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast<bit_integer_type>(*iter);
      *controls_iter = ket::make_control(ket::make_qubit<state_integer_type>(control));
    }

    auto const targets_last = std::end(targets);
    for (auto targets_iter = std::begin(targets); targets_iter != targets_last; ++targets_iter, ++iter)
    {
      auto const target = boost::lexical_cast<bit_integer_type>(*iter);
      *targets_iter = ket::make_qubit<state_integer_type>(target);
    }

    auto const phase = boost::lexical_cast<real_type>(*iter);
    return phase;
  }

  std::tuple<gates::qubit_type, gates::qubit_type, gates::real_type>
  gates::read_multi_controls_2targets_phase(
    gates::columns_type const& columns, std::vector<gates::control_qubit_type>& controls) const
  {
    if (boost::size(columns) < 6u or controls.size() != boost::size(columns) - 4u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    ++iter;
    auto const controls_last = std::end(controls);
    for (auto controls_iter = std::begin(controls); controls_iter != controls_last; ++controls_iter, ++iter)
    {
      auto const control = boost::lexical_cast<bit_integer_type>(*iter);
      *controls_iter = ket::make_control(ket::make_qubit<state_integer_type>(control));
    }

    auto const target1 = boost::lexical_cast<bit_integer_type>(*iter++);
    auto const target2 = boost::lexical_cast<bit_integer_type>(*iter++);
    auto const phase = boost::lexical_cast<real_type>(*iter);
    return std::make_tuple(
       ket::make_qubit<state_integer_type>(target1),
       ket::make_qubit<state_integer_type>(target2),
       phase);
  }

  ::bra::begin_statement gates::read_begin_statement(gates::columns_type const& columns) const
  {
    auto const column_size = boost::size(columns);

    assert(column_size >= 2u);
    if (column_size >= 4u)
      throw wrong_mnemonics_error{columns};

    if (column_size == 3u)
    {
      auto iter = std::begin(columns);
      if (*++iter == "LEARNING")
        if (*++iter == "MACHINE")
          return ::bra::begin_statement::learning_machine;

      throw wrong_mnemonics_error{columns};
    }

    // if (column_size == 2u)
    auto iter = std::begin(columns);
    if (*++iter != "MEASUREMENT")
      throw wrong_mnemonics_error{columns};

    return ::bra::begin_statement::measurement;
  }

  ::bra::bit_statement gates::read_bit_statement(gates::columns_type const& columns) const
  {
    assert(boost::size(columns) >= 2u);

    auto iter = std::begin(columns);
    if (*++iter != "ASSIGNMENT")
      throw wrong_mnemonics_error{columns};

    return ::bra::bit_statement::assignment;
  }

  std::tuple<gates::bit_integer_type, gates::state_integer_type, gates::state_integer_type>
  gates::read_shor_box(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    auto const num_exponent_qubits = boost::lexical_cast<bit_integer_type>(*++iter);
    auto const divisor = boost::lexical_cast<state_integer_type>(*++iter);
    auto const base = boost::lexical_cast<state_integer_type>(*++iter);

    return std::make_tuple(num_exponent_qubits, divisor, base);
  }

  std::tuple< ::bra::generate_statement, int, int > gates::read_generate_statement(gates::columns_type const& columns) const
  {
    assert(boost::size(columns) == 4u);

    auto iter = std::begin(columns);
    if (*++iter != "EVENTS")
      throw wrong_mnemonics_error{columns};

    auto const num_events = boost::lexical_cast<int>(*++iter);
    auto const seed = boost::lexical_cast<int>(*++iter);
    return std::make_tuple(::bra::generate_statement::events, num_events, seed);
  }

  std::tuple< ::bra::depolarizing_statement, gates::real_type, gates::real_type, gates::real_type, int >
  gates::read_depolarizing_statement(gates::columns_type const& columns) const
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

        ::bra::gates_detail::read_depolarizing_statement(px, present_string, iter, last, string_found, columns);
        if (px < 0.0 or px > 1.0)
          throw wrong_mnemonics_error{columns};

        is_px_checked = true;
      }
      else if (probability_string == "P_Y")
      {
        if (is_py_checked)
          throw wrong_mnemonics_error{columns};

        ::bra::gates_detail::read_depolarizing_statement(py, present_string, iter, last, string_found, columns);
        if (py < 0.0 or py > 1.0)
          throw wrong_mnemonics_error{columns};

        is_py_checked = true;
      }
      else if (probability_string == "P_Z")
      {
        if (is_pz_checked)
          throw wrong_mnemonics_error{columns};

        ::bra::gates_detail::read_depolarizing_statement(pz, present_string, iter, last, string_found, columns);
        if (pz < 0.0 or pz > 1.0)
          throw wrong_mnemonics_error{columns};

        is_pz_checked = true;
      }
      else if (probability_string == "SEED")
      {
        if (is_seed_checked)
          throw wrong_mnemonics_error{columns};

        ::bra::gates_detail::read_depolarizing_statement(seed, present_string, iter, last, string_found, columns);
        is_seed_checked = true;
      }
      else
        throw wrong_mnemonics_error{columns};
    }

    if (px > real_type{1} or px < real_type{0} or py > real_type{1} or py < real_type{0} or pz > real_type{1} or pz < real_type{0} or px + py + pz > real_type{1})
      throw wrong_mnemonics_error{columns};

    return std::make_tuple(::bra::depolarizing_statement::channel, px, py, pz, seed);
  }

  void gates::add_h(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::hadamard{read_target(columns)}});
  }

  void gates::add_not(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::not_{read_target(columns)}});
  }

  void gates::add_x(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::pauli_x{read_target(columns)}});
  }

  void gates::add_xx(gates::columns_type const& columns)
  {
    auto target1 = qubit_type{};
    auto target2 = qubit_type{};
    std::tie(target1, target2) = read_2targets(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::pauli_xx{target1, target2}});
  }

  void gates::add_xs(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector<qubit_type>(mnemonic.size());
    read_multi_targets(columns, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::pauli_xn{std::move(targets)}});
  }

  void gates::add_xn(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto const possible_digits_view = boost::string_view{std::next(mnemonic.data()), std::size(mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_qubits == 1)
      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::pauli_x{read_target(columns)}});
    else if (num_qubits == 2)
    {
      auto target1 = qubit_type{};
      auto target2 = qubit_type{};
      std::tie(target1, target2) = read_2targets(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::pauli_xx{target1, target2}});
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector<qubit_type>(num_qubits);
      read_multi_targets(columns, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::pauli_xn{std::move(targets)}});
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void gates::add_y(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::pauli_y{read_target(columns)}});
  }

  void gates::add_yy(gates::columns_type const& columns)
  {
    auto target1 = qubit_type{};
    auto target2 = qubit_type{};
    std::tie(target1, target2) = read_2targets(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::pauli_yy{target1, target2}});
  }

  void gates::add_ys(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector<qubit_type>(mnemonic.size());
    read_multi_targets(columns, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::pauli_yn{std::move(targets)}});
  }

  void gates::add_yn(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto const possible_digits_view = boost::string_view{std::next(mnemonic.data()), std::size(mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_qubits == 1)
      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::pauli_y{read_target(columns)}});
    else if (num_qubits == 2)
    {
      auto target1 = qubit_type{};
      auto target2 = qubit_type{};
      std::tie(target1, target2) = read_2targets(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::pauli_yy{target1, target2}});
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector<qubit_type>(num_qubits);
      read_multi_targets(columns, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::pauli_yn{std::move(targets)}});
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void gates::add_z(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::pauli_z{read_target(columns)}});
  }

  void gates::add_zz(gates::columns_type const& columns)
  {
    auto target1 = qubit_type{};
    auto target2 = qubit_type{};
    std::tie(target1, target2) = read_2targets(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::pauli_zz{target1, target2}});
  }

  void gates::add_zs(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector<qubit_type>(mnemonic.size());
    read_multi_targets(columns, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::pauli_zn{std::move(targets)}});
  }

  void gates::add_zn(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto const possible_digits_view = boost::string_view{std::next(mnemonic.data()), std::size(mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_qubits == 1)
      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::pauli_z{read_target(columns)}});
    else if (num_qubits == 2)
    {
      auto target1 = qubit_type{};
      auto target2 = qubit_type{};
      std::tie(target1, target2) = read_2targets(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::pauli_zz{target1, target2}});
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector<qubit_type>(num_qubits);
      read_multi_targets(columns, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::pauli_zn{std::move(targets)}});
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void gates::add_swap(gates::columns_type const& columns)
  {
    auto target1 = qubit_type{};
    auto target2 = qubit_type{};
    std::tie(target1, target2) = read_2targets(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::swap{target1, target2}});
  }

  void gates::add_s(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::s_gate{phase_coefficients_[2u], read_target(columns)}});
  }

  void gates::add_adj_s(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::adj_s_gate{phase_coefficients_[2u], read_target(columns)}});
  }

  void gates::add_t(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::t_gate{phase_coefficients_[3u], read_target(columns)}});
  }

  void gates::add_adj_t(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::adj_t_gate{phase_coefficients_[3u], read_target(columns)}});
  }

  void gates::add_u1(gates::columns_type const& columns)
  {
    auto target = qubit_type{};
    auto phase = real_type{};
    std::tie(target, phase) = read_target_phase(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{new ::bra::gate::u1{phase, target}});
  }

  void gates::add_u2(gates::columns_type const& columns)
  {
    auto target = qubit_type{};
    auto phase1 = real_type{};
    auto phase2 = real_type{};
    std::tie(target, phase1, phase2) = read_target_2phases(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{new ::bra::gate::u2{phase1, phase2, target}});
  }

  void gates::add_u3(gates::columns_type const& columns)
  {
    auto target = qubit_type{};
    auto phase1 = real_type{};
    auto phase2 = real_type{};
    auto phase3 = real_type{};
    std::tie(target, phase1, phase2, phase3) = read_target_3phases(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{new ::bra::gate::u3{phase1, phase2, phase3, target}});
  }

  void gates::add_r(gates::columns_type const& columns)
  {
    auto target = qubit_type{};
    auto phase_exponent = int{};
    std::tie(target, phase_exponent) = read_target_phaseexp(columns);

    if (phase_exponent >= 0)
      data_.push_back(std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::phase_shift{
          phase_exponent, phase_coefficients_[phase_exponent], target}});
    else
    {
      phase_exponent *= -1;
      data_.push_back(std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::adj_phase_shift{
          phase_exponent, phase_coefficients_[phase_exponent], target}});
    }
  }

  void gates::add_adj_r(gates::columns_type const& columns)
  {
    auto target = qubit_type{};
    auto phase_exponent = int{};
    std::tie(target, phase_exponent) = read_target_phaseexp(columns);

    if (phase_exponent >= 0)
      data_.push_back(std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::adj_phase_shift{
          phase_exponent, phase_coefficients_[phase_exponent], target}});
    else
    {
      phase_exponent *= -1;
      data_.push_back(std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::phase_shift{
          phase_exponent, phase_coefficients_[phase_exponent], target}});
    }
  }

  void gates::add_rotx(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::x_rotation_half_pi{read_target(columns)}});
  }

  void gates::add_adj_rotx(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::adj_x_rotation_half_pi{read_target(columns)}});
  }

  void gates::add_roty(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::y_rotation_half_pi{read_target(columns)}});
  }

  void gates::add_adj_roty(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::adj_y_rotation_half_pi{read_target(columns)}});
  }

  void gates::add_u(gates::columns_type const& columns)
  {
    auto control = control_qubit_type{};
    auto target = qubit_type{};
    auto phase_exponent = int{};
    std::tie(control, target, phase_exponent) = read_control_target_phaseexp(columns);

    if (phase_exponent >= 0)
      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_phase_shift{
            phase_exponent, phase_coefficients_[phase_exponent], target, control}});
    else
    {
      phase_exponent *= -1;
      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::adj_controlled_phase_shift{
            phase_exponent, phase_coefficients_[phase_exponent], target, control}});
    }
  }

  void gates::add_v(gates::columns_type const& columns)
  {
    auto control = control_qubit_type{};
    auto target = qubit_type{};
    auto phase_exponent = int{};
    std::tie(control, target, phase_exponent) = read_control_target_phaseexp(columns);

    if (phase_exponent >= 0)
      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_v{
            phase_exponent, phase_coefficients_[phase_exponent], target, control}});
    else
    {
      phase_exponent *= -1;
      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::adj_controlled_v{
            phase_exponent, phase_coefficients_[phase_exponent], target, control}});
    }
  }

  void gates::add_ex(gates::columns_type const& columns)
  {
    auto target = qubit_type{};
    auto phase = real_type{};
    std::tie(target, phase) = read_target_phase(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::exponential_pauli_x{phase, target}});
  }

  void gates::add_exx(gates::columns_type const& columns)
  {
    auto target1 = qubit_type{};
    auto target2 = qubit_type{};
    auto phase = real_type{};
    std::tie(target1, target2, phase) = read_2targets_phase(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::exponential_pauli_xx{phase, target1, target2}});
  }

  void gates::add_exs(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector<qubit_type>(mnemonic.size() - 1u);
    auto const phase = read_multi_targets_phase(columns, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::exponential_pauli_xn{phase, std::move(targets)}});
  }

  void gates::add_exn(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto const possible_digits_view = boost::string_view{std::next(mnemonic.data()), std::size(mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_qubits == 1)
    {
      auto target = qubit_type{};
      auto phase = real_type{};
      std::tie(target, phase) = read_target_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::exponential_pauli_x{phase, target}});
    }
    else if (num_qubits == 2)
    {
      auto target1 = qubit_type{};
      auto target2 = qubit_type{};
      auto phase = real_type{};
      std::tie(target1, target2, phase) = read_2targets_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::exponential_pauli_xx{phase, target1, target2}});
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector<qubit_type>(num_qubits);
      auto const phase = read_multi_targets_phase(columns, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::exponential_pauli_xn{phase, std::move(targets)}});
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void gates::add_ey(gates::columns_type const& columns)
  {
    auto target = qubit_type{};
    auto phase = real_type{};
    std::tie(target, phase) = read_target_phase(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::exponential_pauli_y{phase, target}});
  }

  void gates::add_eyy(gates::columns_type const& columns)
  {
    auto target1 = qubit_type{};
    auto target2 = qubit_type{};
    auto phase = real_type{};
    std::tie(target1, target2, phase) = read_2targets_phase(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::exponential_pauli_yy{phase, target1, target2}});
  }

  void gates::add_eys(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector<qubit_type>(mnemonic.size() - 1u);
    auto const phase = read_multi_targets_phase(columns, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::exponential_pauli_yn{phase, std::move(targets)}});
  }

  void gates::add_eyn(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto const possible_digits_view = boost::string_view{std::next(mnemonic.data()), std::size(mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_qubits == 1)
    {
      auto target = qubit_type{};
      auto phase = real_type{};
      std::tie(target, phase) = read_target_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::exponential_pauli_y{phase, target}});
    }
    else if (num_qubits == 2)
    {
      auto target1 = qubit_type{};
      auto target2 = qubit_type{};
      auto phase = real_type{};
      std::tie(target1, target2, phase) = read_2targets_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::exponential_pauli_yy{phase, target1, target2}});
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector<qubit_type>(num_qubits);
      auto const phase = read_multi_targets_phase(columns, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::exponential_pauli_yn{phase, std::move(targets)}});
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void gates::add_ez(gates::columns_type const& columns)
  {
    auto target = qubit_type{};
    auto phase = real_type{};
    std::tie(target, phase) = read_target_phase(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::exponential_pauli_z{phase, target}});
  }

  void gates::add_ezz(gates::columns_type const& columns)
  {
    auto target1 = qubit_type{};
    auto target2 = qubit_type{};
    auto phase = real_type{};
    std::tie(target1, target2, phase) = read_2targets_phase(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::exponential_pauli_zz{phase, target1, target2}});
  }

  void gates::add_ezs(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto targets = std::vector<qubit_type>(mnemonic.size() - 1u);
    auto const phase = read_multi_targets_phase(columns, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::exponential_pauli_zn{phase, std::move(targets)}});
  }

  void gates::add_ezn(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto const possible_digits_view = boost::string_view{std::next(mnemonic.data()), std::size(mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_qubits == 1)
    {
      auto target = qubit_type{};
      auto phase = real_type{};
      std::tie(target, phase) = read_target_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::exponential_pauli_z{phase, target}});
    }
    else if (num_qubits == 2)
    {
      auto target1 = qubit_type{};
      auto target2 = qubit_type{};
      auto phase = real_type{};
      std::tie(target1, target2, phase) = read_2targets_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::exponential_pauli_zz{phase, target1, target2}});
    }
    else if (num_qubits >= 3)
    {
      auto targets = std::vector<qubit_type>(num_qubits);
      auto const phase = read_multi_targets_phase(columns, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::exponential_pauli_zn{phase, std::move(targets)}});
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void gates::add_eswap(gates::columns_type const& columns)
  {
    auto target1 = qubit_type{};
    auto target2 = qubit_type{};
    auto phase = real_type{};
    std::tie(target1, target2, phase) = read_2targets_phase(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::exponential_swap{phase, target1, target2}});
  }

  void gates::add_toffoli(gates::columns_type const& columns)
  {
    auto control1 = control_qubit_type{};
    auto control2 = control_qubit_type{};
    auto target = qubit_type{};
    std::tie(control1, control2, target) = read_2controls_target(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::toffoli{target, control1, control2}});
  }

  void gates::add_m(gates::columns_type const& columns)
  {
#ifndef BRA_NO_MPI
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::projective_measurement{read_target(columns), root_}});
#else // BRA_NO_MPI
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::projective_measurement{read_target(columns)}});
#endif // BRA_NO_MPI
  }

  void gates::add_shor_box(gates::columns_type const& columns)
  {
    auto num_exponent_qubits = bit_integer_type{};
    auto divisor = state_integer_type{};
    auto base = state_integer_type{};
    std::tie(num_exponent_qubits, divisor, base) = read_shor_box(columns);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::shor_box{num_exponent_qubits, divisor, base}});
  }

  void gates::add_clear(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::clear{read_target(columns)}});
  }

  void gates::add_set(gates::columns_type const& columns)
  {
    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::set{read_target(columns)}});
  }

  void gates::add_depolarizing(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto statement = ::bra::depolarizing_statement{};
    auto px = real_type{};
    auto py = real_type{};
    auto pz = real_type{};
    auto seed = int{};
    std::tie(statement, px, py, pz, seed) = read_depolarizing_statement(columns);

    if (statement == ::bra::depolarizing_statement::channel)
      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::depolarizing_channel{px, py, pz, seed}});
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void gates::interpret_controlled_gates(gates::columns_type const& columns, std::string const& mnemonic)
  {
    auto const cs_first = std::begin(mnemonic);
    auto const cs_last
      = std::find_if_not(
          cs_first, std::end(mnemonic),
          [](char const character) { return character == 'C'; });
    auto const possible_digits_first = std::next(std::begin(mnemonic));
    auto const possible_digits_last
      = std::find_if_not(
          possible_digits_first, std::end(mnemonic),
          [](unsigned char const character) { return std::isdigit(character); });

    auto const num_control_qubits
      = cs_last - cs_first > 1
        ? static_cast<int>(cs_last - cs_first)
        : possible_digits_first == possible_digits_last
          ? 1
          : ::bra::utility::to_integer<int>(
              boost::string_view{
                boost::to_address(possible_digits_first),
                static_cast<boost::string_view::size_type>(possible_digits_last - possible_digits_first)});
    if (num_control_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    auto const noncontrol_mnemonic
      = cs_last - cs_first > 1
        ? boost::string_view{
            boost::to_address(cs_last),
            static_cast<boost::string_view::size_type>(std::end(mnemonic) - cs_last)}
        : boost::string_view{
            boost::to_address(possible_digits_last),
            static_cast<boost::string_view::size_type>(std::end(mnemonic) - possible_digits_last)};

    if (noncontrol_mnemonic == "H")
      add_ch(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "NOT")
      add_cnot(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "X")
      add_cx(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 2u
             and std::all_of(
                   std::begin(noncontrol_mnemonic), std::end(noncontrol_mnemonic),
                   [](char const character) { return character == 'X'; }))
      add_cxs(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 2u and noncontrol_mnemonic.front() == 'X')
      add_cxn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "Y")
      add_cy(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 2u
             and std::all_of(
                   std::begin(noncontrol_mnemonic), std::end(noncontrol_mnemonic),
                   [](char const character) { return character == 'Y'; }))
      add_cys(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 2u and noncontrol_mnemonic.front() == 'Y')
      add_cyn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "Z")
      add_cz(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 2u
             and std::all_of(
                   std::begin(noncontrol_mnemonic), std::end(noncontrol_mnemonic),
                   [](char const character) { return character == 'Z'; }))
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
    else if (noncontrol_mnemonic == "U2")
      add_cu2(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "U3")
      add_cu3(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "R" or noncontrol_mnemonic == "+R")
      add_cr(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "-R")
      add_adj_cr(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "+X")
      add_crotx(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "-X")
      add_adj_crotx(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "+Y")
      add_croty(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "-Y")
      add_adj_croty(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "U")
      add_cu(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "V")
      add_cv(columns, num_control_qubits);
    else if (noncontrol_mnemonic == "EX")
      add_cex(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 3u
             and noncontrol_mnemonic.front() == 'E'
             and std::all_of(
                   std::next(std::begin(noncontrol_mnemonic)), std::end(noncontrol_mnemonic),
                   [](char const character) { return character == 'X'; }))
      add_cexs(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 3u and noncontrol_mnemonic[0u] == 'E' and noncontrol_mnemonic[1u] == 'X')
      add_cexn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "EY")
      add_cey(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 3u
             and noncontrol_mnemonic.front() == 'E'
             and std::all_of(
                   std::next(std::begin(noncontrol_mnemonic)), std::end(noncontrol_mnemonic),
                   [](char const character) { return character == 'Y'; }))
      add_ceys(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 3u and noncontrol_mnemonic[0u] == 'E' and noncontrol_mnemonic[1u] == 'Y')
      add_ceyn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "EZ")
      add_cez(columns, num_control_qubits);
    else if (noncontrol_mnemonic.size() >= 3u
             and noncontrol_mnemonic.front() == 'E'
             and std::all_of(
                   std::next(std::begin(noncontrol_mnemonic)), std::end(noncontrol_mnemonic),
                   [](char const character) { return character == 'Z'; }))
      add_cezs(columns, num_control_qubits, noncontrol_mnemonic);
    else if (noncontrol_mnemonic.size() >= 3u and noncontrol_mnemonic[0u] == 'E' and noncontrol_mnemonic[1u] == 'Z')
      add_cezn(columns, num_control_qubits, noncontrol_mnemonic, mnemonic);
    else if (noncontrol_mnemonic == "ESWAP")
      add_ceswap(columns, num_control_qubits);
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void gates::add_ch(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_hadamard{target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_hadamard{target, std::move(controls)}});
    }
  }

  void gates::add_cnot(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_not{target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto target = read_multi_controls_target(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_not{target, std::move(controls)}});
    }
  }

  void gates::add_cx(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_pauli_x{target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(1u);
      read_multi_controls_multi_targets(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_pauli_xn{std::move(targets), std::move(controls)}});
    }
  }

  void gates::add_cxs(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic)
  {
    auto controls = std::vector<control_qubit_type>(num_control_qubits);
    auto targets = std::vector<qubit_type>(noncontrol_mnemonic.size());
    read_multi_controls_multi_targets(columns, controls, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::multi_controlled_pauli_xn{std::move(targets), std::move(controls)}});
  }

  void gates::add_cxn(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic, std::string const& mnemonic)
  {
    auto const possible_digits_view
      = boost::string_view{std::next(noncontrol_mnemonic.data()), std::size(noncontrol_mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_control_qubits == 1 and num_target_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_pauli_x{target, control}});
    }
    else if (num_target_qubits >= 1)
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(num_target_qubits);
      read_multi_controls_multi_targets(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_pauli_xn{std::move(targets), std::move(controls)}});
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void gates::add_cy(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_pauli_y{target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(1u);
      read_multi_controls_multi_targets(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_pauli_yn{std::move(targets), std::move(controls)}});
    }
  }

  void gates::add_cys(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic)
  {
    auto controls = std::vector<control_qubit_type>(num_control_qubits);
    auto targets = std::vector<qubit_type>(noncontrol_mnemonic.size());
    read_multi_controls_multi_targets(columns, controls, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::multi_controlled_pauli_yn{std::move(targets), std::move(controls)}});
  }

  void gates::add_cyn(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic, std::string const& mnemonic)
  {
    auto const possible_digits_view
      = boost::string_view{std::next(noncontrol_mnemonic.data()), std::size(noncontrol_mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_control_qubits == 1 and num_target_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_pauli_y{target, control}});
    }
    else if (num_target_qubits >= 1)
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(num_target_qubits);
      read_multi_controls_multi_targets(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_pauli_yn{std::move(targets), std::move(controls)}});
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void gates::add_cz(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_pauli_z{target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(1u);
      read_multi_controls_multi_targets(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_pauli_zn{std::move(targets), std::move(controls)}});
    }
  }

  void gates::add_czs(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic)
  {
    auto controls = std::vector<control_qubit_type>(num_control_qubits);
    auto targets = std::vector<qubit_type>(noncontrol_mnemonic.size());
    read_multi_controls_multi_targets(columns, controls, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::multi_controlled_pauli_zn{std::move(targets), std::move(controls)}});
  }

  void gates::add_czn(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic, std::string const& mnemonic)
  {
    auto const possible_digits_view
      = boost::string_view{std::next(noncontrol_mnemonic.data()), std::size(noncontrol_mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_control_qubits == 1 and num_target_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_pauli_z{target, control}});
    }
    else if (num_target_qubits >= 1)
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(num_target_qubits);
      read_multi_controls_multi_targets(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_pauli_zn{std::move(targets), std::move(controls)}});
    }
    else
      throw unsupported_mnemonic_error{mnemonic};
  }

  void gates::add_cswap(gates::columns_type const& columns, int const num_control_qubits)
  {
    auto controls = std::vector<control_qubit_type>(num_control_qubits);
    auto target1 = qubit_type{};
    auto target2 = qubit_type{};
    std::tie(target1, target2) = read_multi_controls_2targets(columns, controls);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::multi_controlled_swap{target1, target2, std::move(controls)}});
  }

  void gates::add_cs(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_s_gate{phase_coefficients_[2u], target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_s_gate{phase_coefficients_[2u], target, std::move(controls)}});
    }
  }

  void gates::add_adj_cs(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::adj_controlled_s_gate{phase_coefficients_[2u], target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::adj_multi_controlled_s_gate{phase_coefficients_[2u], target, std::move(controls)}});
    }
  }

  void gates::add_ct(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_t_gate{phase_coefficients_[3u], target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_t_gate{phase_coefficients_[3u], target, std::move(controls)}});
    }
  }

  void gates::add_adj_ct(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::adj_controlled_t_gate{phase_coefficients_[3u], target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::adj_multi_controlled_t_gate{phase_coefficients_[3u], target, std::move(controls)}});
    }
  }

  void gates::add_cu1(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase = real_type{};
      std::tie(control, target, phase) = read_control_target_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_u1{phase, target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto target = qubit_type{};
      auto phase = real_type{};
      std::tie(target, phase) = read_multi_controls_target_phase(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_u1{phase, target, std::move(controls)}});
    }
  }

  void gates::add_cu2(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase1 = real_type{};
      auto phase2 = real_type{};
      std::tie(control, target, phase1, phase2) = read_control_target_2phases(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_u2{phase1, phase2, target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto target = qubit_type{};
      auto phase1 = real_type{};
      auto phase2 = real_type{};
      std::tie(target, phase1, phase2) = read_multi_controls_target_2phases(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_u2{phase1, phase2, target, std::move(controls)}});
    }
  }

  void gates::add_cu3(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase1 = real_type{};
      auto phase2 = real_type{};
      auto phase3 = real_type{};
      std::tie(control, target, phase1, phase2, phase3) = read_control_target_3phases(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_u3{phase1, phase2, phase3, target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto target = qubit_type{};
      auto phase1 = real_type{};
      auto phase2 = real_type{};
      auto phase3 = real_type{};
      std::tie(target, phase1, phase2, phase3) = read_multi_controls_target_3phases(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_u3{phase1, phase2, phase3, target, std::move(controls)}});
    }
  }

  void gates::add_cr(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase_exponent = int{};
      std::tie(control, target, phase_exponent) = read_control_target_phaseexp(columns);

      if (phase_exponent >= 0)
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::controlled_phase_shift_{
              phase_exponent, phase_coefficients_[phase_exponent], target, control}});
      else
      {
        phase_exponent *= -1;
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_controlled_phase_shift_{
              phase_exponent, phase_coefficients_[phase_exponent], target, control}});
      }
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto target = qubit_type{};
      auto phase_exponent = int{};
      std::tie(target, phase_exponent) = read_multi_controls_target_phaseexp(columns, controls);

      if (phase_exponent >= 0)
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::multi_controlled_phase_shift{
              phase_exponent, phase_coefficients_[phase_exponent], target, std::move(controls)}});
      else
      {
        phase_exponent *= -1;
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_multi_controlled_phase_shift{
              phase_exponent, phase_coefficients_[phase_exponent], target, std::move(controls)}});
      }
    }
  }

  void gates::add_adj_cr(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase_exponent = int{};
      std::tie(control, target, phase_exponent) = read_control_target_phaseexp(columns);

      if (phase_exponent >= 0)
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_controlled_phase_shift{
              phase_exponent, phase_coefficients_[phase_exponent], target, control}});
      else
      {
        phase_exponent *= -1;
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::controlled_phase_shift{
              phase_exponent, phase_coefficients_[phase_exponent], target, control}});
      }
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto target = qubit_type{};
      auto phase_exponent = int{};
      std::tie(target, phase_exponent) = read_multi_controls_target_phaseexp(columns, controls);

      if (phase_exponent >= 0)
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_multi_controlled_phase_shift{
              phase_exponent, phase_coefficients_[phase_exponent], target, std::move(controls)}});
      else
      {
        phase_exponent *= -1;
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::multi_controlled_phase_shift{
              phase_exponent, phase_coefficients_[phase_exponent], target, std::move(controls)}});
      }
    }
  }

  void gates::add_crotx(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_x_rotation_half_pi{target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_x_rotation_half_pi{target, std::move(controls)}});
    }
  }

  void gates::add_adj_crotx(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::adj_controlled_x_rotation_half_pi{target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::adj_multi_controlled_x_rotation_half_pi{target, std::move(controls)}});
    }
  }

  void gates::add_croty(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_y_rotation_half_pi{target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_y_rotation_half_pi{target, std::move(controls)}});
    }
  }

  void gates::add_adj_croty(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      std::tie(control, target) = read_control_target(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::adj_controlled_y_rotation_half_pi{target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto const target = read_multi_controls_target(columns, controls);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::adj_multi_controlled_y_rotation_half_pi{target, std::move(controls)}});
    }
  }

  void gates::add_cu(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase_exponent = int{};
      std::tie(control, target, phase_exponent) = read_control_target_phaseexp(columns);

      if (phase_exponent >= 0)
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::controlled_phase_shift_cu{
              phase_exponent, phase_coefficients_[phase_exponent], target, control}});
      else
      {
        phase_exponent *= -1;
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_controlled_phase_shift_cu{
              phase_exponent, phase_coefficients_[phase_exponent], target, control}});
      }
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto target = qubit_type{};
      auto phase_exponent = int{};
      std::tie(target, phase_exponent) = read_multi_controls_target_phaseexp(columns, controls);

      if (phase_exponent >= 0)
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::multi_controlled_phase_shift{
              phase_exponent, phase_coefficients_[phase_exponent], target, std::move(controls)}});
      else
      {
        phase_exponent *= -1;
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_multi_controlled_phase_shift{
              phase_exponent, phase_coefficients_[phase_exponent], target, std::move(controls)}});
      }
    }
  }

  void gates::add_cv(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase_exponent = int{};
      std::tie(control, target, phase_exponent) = read_control_target_phaseexp(columns);

      if (phase_exponent >= 0)
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::controlled_v_{
              phase_exponent, phase_coefficients_[phase_exponent], target, control}});
      else
      {
        phase_exponent *= -1;
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_controlled_v_{
              phase_exponent, phase_coefficients_[phase_exponent], target, control}});
      }
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto target = qubit_type{};
      auto phase_exponent = int{};
      std::tie(target, phase_exponent) = read_multi_controls_target_phaseexp(columns, controls);

      if (phase_exponent >= 0)
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::multi_controlled_v{
              phase_exponent, phase_coefficients_[phase_exponent], target, std::move(controls)}});
      else
      {
        phase_exponent *= -1;
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_multi_controlled_v{
              phase_exponent, phase_coefficients_[phase_exponent], target, std::move(controls)}});
      }
    }
  }

  void gates::add_cex(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase = real_type{};
      std::tie(control, target, phase) = read_control_target_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_exponential_pauli_x{phase, target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(1u);
      auto const phase = read_multi_controls_multi_targets_phase(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_exponential_pauli_xn{phase, std::move(targets), std::move(controls)}});
    }
  }

  void gates::add_cexs(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic)
  {
    auto controls = std::vector<control_qubit_type>(num_control_qubits);
    auto targets = std::vector<qubit_type>(noncontrol_mnemonic.size() - 1u);
    auto const phase = read_multi_controls_multi_targets_phase(columns, controls, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::multi_controlled_exponential_pauli_xn{phase, std::move(targets), std::move(controls)}});
  }

  void gates::add_cexn(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic, std::string const& mnemonic)
  {
    auto const possible_digits_view
      = boost::string_view{std::next(noncontrol_mnemonic.data()), std::size(noncontrol_mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_target_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    if (num_control_qubits + num_target_qubits == 2)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase = real_type{};
      std::tie(control, target, phase) = read_control_target_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_exponential_pauli_x{phase, target, control}});
    }
    else // num_control_qubits + num_target_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(num_target_qubits);
      auto const phase = read_multi_controls_multi_targets_phase(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_exponential_pauli_xn{phase, std::move(targets), std::move(controls)}});
    }
  }

  void gates::add_cey(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase = real_type{};
      std::tie(control, target, phase) = read_control_target_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_exponential_pauli_y{phase, target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(1u);
      auto const phase = read_multi_controls_multi_targets_phase(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_exponential_pauli_yn{phase, std::move(targets), std::move(controls)}});
    }
  }

  void gates::add_ceys(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic)
  {
    auto controls = std::vector<control_qubit_type>(num_control_qubits);
    auto targets = std::vector<qubit_type>(noncontrol_mnemonic.size() - 1u);
    auto const phase = read_multi_controls_multi_targets_phase(columns, controls, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::multi_controlled_exponential_pauli_yn{phase, std::move(targets), std::move(controls)}});
  }

  void gates::add_ceyn(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic, std::string const& mnemonic)
  {
    auto const possible_digits_view
      = boost::string_view{std::next(noncontrol_mnemonic.data()), std::size(noncontrol_mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_target_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    if (num_control_qubits + num_target_qubits == 2)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase = real_type{};
      std::tie(control, target, phase) = read_control_target_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_exponential_pauli_y{phase, target, control}});
    }
    else // num_control_qubits + num_target_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(num_target_qubits);
      auto const phase = read_multi_controls_multi_targets_phase(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_exponential_pauli_yn{phase, std::move(targets), std::move(controls)}});
    }
  }

  void gates::add_cez(gates::columns_type const& columns, int const num_control_qubits)
  {
    if (num_control_qubits == 1)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase = real_type{};
      std::tie(control, target, phase) = read_control_target_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_exponential_pauli_z{phase, target, control}});
    }
    else // num_control_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(1u);
      auto const phase = read_multi_controls_multi_targets_phase(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_exponential_pauli_zn{phase, std::move(targets), std::move(controls)}});
    }
  }

  void gates::add_cezs(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic)
  {
    auto controls = std::vector<control_qubit_type>(num_control_qubits);
    auto targets = std::vector<qubit_type>(noncontrol_mnemonic.size() - 1u);
    auto const phase = read_multi_controls_multi_targets_phase(columns, controls, targets);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::multi_controlled_exponential_pauli_zn{phase, std::move(targets), std::move(controls)}});
  }

  void gates::add_cezn(
    gates::columns_type const& columns, int const num_control_qubits,
    boost::string_view const noncontrol_mnemonic, std::string const& mnemonic)
  {
    auto const possible_digits_view
      = boost::string_view{std::next(noncontrol_mnemonic.data()), std::size(noncontrol_mnemonic) - 1u};
    if (not std::all_of(
              std::begin(possible_digits_view), std::end(possible_digits_view),
              [](unsigned char const character) { return std::isdigit(character); }))
      throw unsupported_mnemonic_error{mnemonic};

    auto const num_target_qubits = ::bra::utility::to_integer<int>(possible_digits_view);
    if (num_target_qubits <= 0)
      throw unsupported_mnemonic_error{mnemonic};

    if (num_control_qubits + num_target_qubits == 2)
    {
      auto control = control_qubit_type{};
      auto target = qubit_type{};
      auto phase = real_type{};
      std::tie(control, target, phase) = read_control_target_phase(columns);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::controlled_exponential_pauli_z{phase, target, control}});
    }
    else // num_control_qubits + num_target_qubits >= 2
    {
      auto controls = std::vector<control_qubit_type>(num_control_qubits);
      auto targets = std::vector<qubit_type>(num_target_qubits);
      auto const phase = read_multi_controls_multi_targets_phase(columns, controls, targets);

      data_.push_back(
        std::unique_ptr< ::bra::gate::gate >{
          new ::bra::gate::multi_controlled_exponential_pauli_zn{phase, std::move(targets), std::move(controls)}});
    }
  }

  void gates::add_ceswap(gates::columns_type const& columns, int const num_control_qubits)
  {
    auto controls = std::vector<control_qubit_type>(num_control_qubits);
    auto target1 = qubit_type{};
    auto target2 = qubit_type{};
    auto phase = real_type{};
    std::tie(target1, target2, phase) = read_multi_controls_2targets_phase(columns, controls);

    data_.push_back(
      std::unique_ptr< ::bra::gate::gate >{
        new ::bra::gate::multi_controlled_exponential_swap{phase, target1, target2, std::move(controls)}});
  }
} // namespace bra


# undef BRA_is_nothrow_swappable
