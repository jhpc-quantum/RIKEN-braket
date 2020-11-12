#include <istream>
#include <string>
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
#include <bra/gate/gate.hpp>
#include <bra/gate/hadamard.hpp>
#include <bra/gate/pauli_x.hpp>
#include <bra/gate/pauli_y.hpp>
#include <bra/gate/pauli_z.hpp>
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
#include <bra/gate/controlled_not.hpp>
#include <bra/gate/controlled_phase_shift.hpp>
#include <bra/gate/adj_controlled_phase_shift.hpp>
#include <bra/gate/controlled_v.hpp>
#include <bra/gate/adj_controlled_v.hpp>
#include <bra/gate/toffoli.hpp>
#include <bra/gate/projective_measurement.hpp>
#include <bra/gate/measurement.hpp>
#include <bra/gate/generate_events.hpp>
#include <bra/gate/shor_box.hpp>
#include <bra/gate/clear.hpp>
#include <bra/gate/set.hpp>
#include <bra/gate/depolarizing_channel.hpp>
#include <bra/gate/exit.hpp>

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
    : data_{}, num_qubits_{}, num_lqubits_{},
      initial_state_value_{}, initial_permutation_{}, phase_coefficients_{}, root_{}
  { }

  gates::gates(gates::allocator_type const& allocator)
    : data_{allocator}, num_qubits_{}, num_lqubits_{},
      initial_state_value_{}, initial_permutation_{}, phase_coefficients_{}, root_{}
  { }

  gates::gates(gates&& other, gates::allocator_type const& allocator)
      : data_{std::move(other.data_), allocator},
        num_qubits_{std::move(other.num_qubits_)},
        num_lqubits_{std::move(other.num_lqubits_)},
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
    std::istream& input_stream, yampi::environment const& environment,
    yampi::rank const root, yampi::communicator const& communicator,
    size_type const num_reserved_gates)
    : data_{}, num_qubits_{}, num_lqubits_{},
      initial_state_value_{}, initial_permutation_{}, phase_coefficients_{}, root_{root}
  { assign(input_stream, environment, communicator, num_reserved_gates); }
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
      = ket::utility::integer_log2<bit_integer_type>(communicator.size(environment));
    set_num_qubits_params(new_num_qubits - num_gqubits, num_gqubits, communicator, environment);
  }

  void gates::num_lqubits(
    bit_integer_type const new_num_lqubits,
    yampi::communicator const& communicator, yampi::environment const& environment)
  {
    set_num_qubits_params(
      new_num_lqubits,
      ket::utility::integer_log2<bit_integer_type>(communicator.size(environment)),
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
    if (ket::utility::integer_exp2<bit_integer_type>(num_gqubits)
        != static_cast<bit_integer_type>(communicator.size(environment)))
      throw wrong_mpi_communicator_size_error{};

    num_lqubits_ = new_num_lqubits;
    num_qubits_ = new_num_lqubits + num_gqubits;
    ket::utility::generate_phase_coefficients(phase_coefficients_, num_qubits_);

    initial_permutation_.clear();
    initial_permutation_.reserve(num_qubits_);
    for (auto bit = bit_integer_type{0u}; bit < num_qubits_; ++bit)
      initial_permutation_.push_back(qubit_type{bit});
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
#ifndef BRA_NO_MPI
    auto const num_gqubits
      = ket::utility::integer_log2<bit_integer_type>(communicator.size(environment));
    if (ket::utility::integer_exp2<bit_integer_type>(num_gqubits)
        != static_cast<bit_integer_type>(communicator.size(environment)))
      throw wrong_mpi_communicator_size_error{};

#endif // BRA_NO_MPI
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
      auto const& first_mnemonic = columns.front();
      if (first_mnemonic == "QUBITS")
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
      else if (first_mnemonic == "INITIAL") // INITIAL STATE
        initial_state_value_
          = static_cast< ::bra::state::state_integer_type >(read_initial_state_value(columns));
      else if (first_mnemonic == "MPIPROCESSES")
      {
        read_num_mpi_processes(columns);
        // ignore this statement
      }
      else if (first_mnemonic == "MPISWAPBUFFER")
      {
        read_mpi_buffer_size(columns);
        // ignore this statement
      }
      else if (first_mnemonic == "BIT") // BIT ASSIGNMENT
      {
        auto const statement = read_bit_statement(columns);

        if (statement == ::bra::bit_statement::assignment)
        {
#ifndef BRA_NO_MPI
          initial_permutation_ = read_initial_permutation(columns);
#endif
        }
      }
      else if (first_mnemonic == "PERMUTATION")
        throw unsupported_mnemonic_error{first_mnemonic};
      else if (first_mnemonic == "RANDOM") // RANDOM PERMUTATION
        throw unsupported_mnemonic_error{first_mnemonic};
      else if (first_mnemonic == "H")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::hadamard{read_hadamard(columns)}});
      else if (first_mnemonic == "X")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::pauli_x{read_pauli_x(columns)}});
      else if (first_mnemonic == "Y")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::pauli_y{read_pauli_y(columns)}});
      else if (first_mnemonic == "Z")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::pauli_z{read_pauli_z(columns)}});
      else if (first_mnemonic == "S")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::s_gate{
              phase_coefficients_[2u], read_s_gate(columns)}});
      else if (first_mnemonic == "S+")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_s_gate{
              phase_coefficients_[2u], read_adj_s_gate(columns)}});
      else if (first_mnemonic == "T")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::t_gate{
              phase_coefficients_[3u], read_t_gate(columns)}});
      else if (first_mnemonic == "T+")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_t_gate{
              phase_coefficients_[3u], read_adj_t_gate(columns)}});
      else if (first_mnemonic == "U1")
      {
        auto target = qubit_type{};
        auto phase = real_type{};
        std::tie(target, phase) = read_u1(columns);

        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::u1{phase, target}});
      }
      else if (first_mnemonic == "U2")
      {
        auto target = qubit_type{};
        auto phase1 = real_type{};
        auto phase2 = real_type{};
        std::tie(target, phase1, phase2) = read_u2(columns);

        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::u2{phase1, phase2, target}});
      }
      else if (first_mnemonic == "U3")
      {
        auto target = qubit_type{};
        auto phase1 = real_type{};
        auto phase2 = real_type{};
        auto phase3 = real_type{};
        std::tie(target, phase1, phase2, phase3) = read_u3(columns);

        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::u3{phase1, phase2, phase3, target}});
      }
      else if (first_mnemonic == "R" or first_mnemonic == "+R")
      {
        auto target = qubit_type{};
        auto phase_exponent = int{};
        std::tie(target, phase_exponent) = read_phase_shift(columns);

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
      else if (first_mnemonic == "-R")
      {
        auto target = qubit_type{};
        auto phase_exponent = int{};
        std::tie(target, phase_exponent) = read_phase_shift(columns);

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
      else if (first_mnemonic == "+X")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::x_rotation_half_pi{read_x_rotation_half_pi(columns)}});
      else if (first_mnemonic == "-X")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_x_rotation_half_pi{read_adj_x_rotation_half_pi(columns)}});
      else if (first_mnemonic == "+Y")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::y_rotation_half_pi{read_y_rotation_half_pi(columns)}});
      else if (first_mnemonic == "-Y")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::adj_y_rotation_half_pi{read_adj_y_rotation_half_pi(columns)}});
      else if (first_mnemonic == "CNOT")
      {
        auto control = control_qubit_type{};
        auto target = qubit_type{};
        std::tie(control, target) = read_controlled_not(columns);

        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::controlled_not{target, control}});
      }
      else if (first_mnemonic == "U")
      {
        auto control = control_qubit_type{};
        auto target = qubit_type{};
        auto phase_exponent = int{};
        std::tie(control, target, phase_exponent) = read_controlled_phase_shift(columns);

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
      else if (first_mnemonic == "V")
      {
        auto control = control_qubit_type{};
        auto target = qubit_type{};
        auto phase_exponent = int{};
        std::tie(control, target, phase_exponent) = read_controlled_v(columns);

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
      else if (first_mnemonic == "TOFFOLI")
      {
        auto control1 = control_qubit_type{};
        auto control2 = control_qubit_type{};
        auto target = qubit_type{};
        std::tie(control1, control2, target) = read_toffoli(columns);

        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::toffoli{target, control1, control2}});
      }
      else if (first_mnemonic == "M")
      {
#ifndef BRA_NO_MPI
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::projective_measurement{read_projective_measurement(columns), root_}});
#else // BRA_NO_MPI
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::projective_measurement{read_projective_measurement(columns)}});
#endif // BRA_NO_MPI
      }
      else if (first_mnemonic == "SHORBOX")
      {
        auto num_exponent_qubits = bit_integer_type{};
        auto divisor = state_integer_type{};
        auto base = state_integer_type{};
        std::tie(num_exponent_qubits, divisor, base) = read_shor_box(columns);

        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::shor_box{num_exponent_qubits, divisor, base}});
      }
      else if (first_mnemonic == "BEGIN") // BEGIN MEASUREMENT/LEARNING MACHINE
      {
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
          throw unsupported_mnemonic_error{first_mnemonic};
      }
      else if (first_mnemonic == "DO") // DO MEASUREMENT
      {
        /*
        auto const statement = read_do_statement(columns);

        if (statement == do_statement::error)
          throw wrong_mnemonics_error{columns};
        else if (statement == do_statement::measurement)
          throw unsupported_mnemonic_error{first_mnemonic};
          */
        throw unsupported_mnemonic_error{first_mnemonic};
      }
      else if (first_mnemonic == "END") // END MEASUREMENT/LEARNING MACHINE
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
          throw unsupported_mnemonic_error{first_mnemonic};
*/
        throw unsupported_mnemonic_error{first_mnemonic};
      }
      else if (first_mnemonic == "GENERATE") // GENERATE EVENTS
      {
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
      else if (first_mnemonic == "CLEAR")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::clear{read_clear(columns)}});
      else if (first_mnemonic == "SET")
        data_.push_back(
          std::unique_ptr< ::bra::gate::gate >{
            new ::bra::gate::set{read_set(columns)}});
      else if (first_mnemonic == "DEPOLARIZING")
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
          throw unsupported_mnemonic_error{first_mnemonic};
      }
      else if (first_mnemonic == "EXIT")
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
      else
        throw unsupported_mnemonic_error{first_mnemonic};
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
  std::vector<gates::qubit_type>
  gates::read_initial_permutation(gates::columns_type const& columns) const
  {
    auto result = std::vector<qubit_type>{};
    result.reserve(boost::size(columns)-2u);

    auto iter = std::begin(columns);
    ++iter;
    ++iter;

    auto const last = std::end(columns);
    for (; iter != last; ++iter)
      result.push_back(static_cast<qubit_type>(boost::lexical_cast<bit_integer_type>(*iter)));

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

  ::bra::begin_statement gates::read_begin_statement(gates::columns_type& columns) const
  {
    auto const column_size = boost::size(columns);

    if (column_size <= 1u or column_size >= 4u)
      throw wrong_mnemonics_error{columns};

    if (column_size == 3u)
    {
      auto iter = std::begin(columns);
      boost::algorithm::to_upper(*++iter);
      if (*iter == "LEARNING")
      {
        boost::algorithm::to_upper(*++iter);

        if (*iter == "MACHINE")
          return ::bra::begin_statement::learning_machine;
        else
          throw wrong_mnemonics_error{columns};
      }
      else
        throw wrong_mnemonics_error{columns};
    }

    // if (column_size == 2u)
    auto iter = std::begin(columns);
    boost::algorithm::to_upper(*++iter);

    if (*iter != "MEASUREMENT")
      throw wrong_mnemonics_error{columns};

    return ::bra::begin_statement::measurement;
  }

  ::bra::bit_statement gates::read_bit_statement(gates::columns_type& columns) const
  {
    if (boost::size(columns) <= 1u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    boost::algorithm::to_upper(*++iter);

    if (*iter != "ASSIGNMENT")
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

  std::tuple< ::bra::generate_statement, int, int > gates::read_generate_statement(gates::columns_type& columns) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error{columns};

    auto iter = std::begin(columns);
    boost::algorithm::to_upper(*++iter);

    if (*iter != "EVENTS")
      throw wrong_mnemonics_error{columns};

    auto const num_events = boost::lexical_cast<int>(*++iter);
    auto const seed = boost::lexical_cast<int>(*++iter);
    return std::make_tuple(::bra::generate_statement::events, num_events, seed);
  }

  std::tuple< ::bra::depolarizing_statement, gates::real_type, gates::real_type, gates::real_type, int >
  gates::read_depolarizing_statement(gates::columns_type& columns) const
  {
    if (boost::size(columns) <= 2u)
      throw wrong_mnemonics_error{columns};

    auto iter = columns.cbegin();
    auto present_string = std::string{*++iter}; // present_string == "CHANNEL"
    boost::algorithm::to_upper(present_string);

    if (present_string != "CHANNEL")
      throw wrong_mnemonics_error{columns};

    auto px = real_type{};
    auto py = real_type{};
    auto pz = real_type{};
    auto seed = -1;
    auto is_px_checked = false;
    auto is_py_checked = false;
    auto is_pz_checked = false;
    auto is_seed_checked = false;

    present_string = *++iter; // present_string == "P_*" or "P_*=" or "P_*=0.xxx" or "P_*=0.xxx," or "P_*=0.xxx,..."
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

    if (is_px_checked and is_py_checked and is_pz_checked and px + py + pz > 1.0)
      throw wrong_mnemonics_error{columns};
    else if (is_px_checked and is_py_checked and not is_pz_checked and px + py < 1.0)
      pz = 1.0 - px - py;
    else if (is_px_checked and not is_py_checked and is_pz_checked and px + pz < 1.0)
      py = 1.0 - px - pz;
    else if (not is_px_checked and is_py_checked and is_pz_checked and py + pz < 1.0)
      px = 1.0 - py - pz;
    else
      throw wrong_mnemonics_error{columns};

    return std::make_tuple(::bra::depolarizing_statement::channel, px, py, pz, seed);
  }
} // namespace bra


# undef BRA_is_nothrow_swappable
