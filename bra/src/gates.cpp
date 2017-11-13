#include <boost/config.hpp>

#include <istream>
#include <string>
#include <utility>
#include <algorithm>
#include <stdexcept>

#include <boost/lexical_cast.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/move/unique_ptr.hpp>

#include <boost/range/iterator.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/size.hpp>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <yampi/communicator.hpp>
#include <yampi/environment.hpp>

#include <ket/qubit.hpp>
#include <ket/control.hpp>
#include <ket/utility/integer_log2.hpp>
#include <ket/utility/integer_exp2.hpp>
#include <ket/utility/is_nothrow_swappable.hpp>
#include <ket/utility/generate_phase_coefficients.hpp>

#include <bra/gates.hpp>
#include <bra/state.hpp>
#include <bra/gate/gate.hpp>
#include <bra/gate/hadamard.hpp>
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
//#include <bra/gate/toffoli.hpp>
#include <bra/gate/measurement.hpp>


namespace bra
{
  unsupported_mnemonic_error::unsupported_mnemonic_error(std::string const& mnemonic)
    : std::runtime_error((mnemonic + " is not supported").c_str())
  { }

  wrong_mnemonics_error::wrong_mnemonics_error(::bra::gates::columns_type const& columns)
    : std::runtime_error(generate_what_string(columns).c_str())
  { }

  std::string wrong_mnemonics_error::generate_what_string(::bra::gates::columns_type const& columns)
  {
    std::string result;

    typedef ::bra::gates::columns_type::const_iterator const_iterator;
    const_iterator const last = columns.end();
    for (const_iterator iter = columns.begin(); iter != last; ++iter)
    {
      result += *iter;
      result += " ";
    }

    return result;
  }

  wrong_mpi_communicator_size_error::wrong_mpi_communicator_size_error()
    : std::runtime_error("communicator size is wrong")
  { }


  gates::gates()
    : data_(), num_qubits_(), num_lqubits_(),
      initial_state_value_(), phase_coefficients_(), root_()
  { }

  gates::gates(gates::allocator_type const& allocator)
    : data_(allocator), num_qubits_(), num_lqubits_(),
      initial_state_value_(), phase_coefficients_(), root_()
  { }

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
  gates::gates(gates&& other, gates::allocator_type const& allocator)
      : data_(std::move(other.data_), allocator),
        num_qubits_(std::move(other.num_qubits_)),
        num_lqubits_(std::move(other.num_lqubits_)),
        initial_state_value_(std::move(other.initial_state_value_)),
        phase_coefficients_(std::move(other.phase_coefficients_)),
        root_(std::move(other.root_))
    { }
# endif


  gates::gates(
    std::istream& input_stream, yampi::environment const& environment,
    yampi::rank const root, yampi::communicator const communicator,
    size_type const num_reserved_gates)
    : data_(), num_qubits_(), num_lqubits_(),
      initial_state_value_(), phase_coefficients_(), root_(root)
  { assign(input_stream, environment, communicator, num_reserved_gates); }

  bool gates::operator==(gates const& other) const
  {
    return data_ == other.data_
      and num_qubits_ == other.num_qubits_
      and num_lqubits_ == other.num_lqubits_
      and initial_state_value_ == other.initial_state_value_
      and phase_coefficients_ == other.phase_coefficients_;
  }

  void gates::num_qubits(
    bit_integer_type const new_num_qubits,
    yampi::communicator const communicator, yampi::environment const& environment)
  {
    bit_integer_type const num_gqubits
      = ket::utility::integer_log2<bit_integer_type>(communicator.size(environment));
    set_num_qubits_params(new_num_qubits-num_gqubits, num_gqubits, communicator, environment);
  }

  void gates::num_lqubits(
    bit_integer_type const new_num_lqubits,
    yampi::communicator const communicator, yampi::environment const& environment)
  {
    set_num_qubits_params(
      new_num_lqubits,
      ket::utility::integer_log2<bit_integer_type>(communicator.size(environment)),
      communicator, environment);
  }

  void gates::set_num_qubits_params(
    bit_integer_type const new_num_lqubits, bit_integer_type const num_gqubits,
    yampi::communicator const communicator, yampi::environment const& environment)
  {
    if (ket::utility::integer_exp2<bit_integer_type>(num_gqubits)
        != static_cast<bit_integer_type>(communicator.size(environment)))
      throw wrong_mpi_communicator_size_error();

    num_lqubits_ = new_num_lqubits;
    num_qubits_ = new_num_lqubits + num_gqubits;
    ket::utility::generate_phase_coefficients(phase_coefficients_, num_qubits_);

    initial_permutation_.clear();
    initial_permutation_.reserve(num_qubits_);
    for (bit_integer_type bit = 0u; bit < num_qubits_; ++bit)
      initial_permutation_.push_back(static_cast<qubit_type>(bit));
  }

  void gates::assign(
    std::istream& input_stream, yampi::environment const& environment,
    yampi::communicator const communicator, size_type const num_reserved_gates)
  {
    bit_integer_type const num_gqubits
      = ket::utility::integer_log2<bit_integer_type>(communicator.size(environment));
    if (ket::utility::integer_exp2<bit_integer_type>(num_gqubits)
        != static_cast<bit_integer_type>(communicator.size(environment)))
      throw wrong_mpi_communicator_size_error();

    data_.clear();
    data_.reserve(num_reserved_gates);

    std::string line;
    columns_type columns;
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
      std::string const& first_mnemonic = columns.front();
      if (first_mnemonic == "QUBITS")
        num_qubits(
          static_cast< ::bra::state::bit_integer_type >(read_num_qubits(columns)),
          communicator, environment);
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
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
        bit_statement const statement = read_bit_statement(columns);

        if (statement == bit_statement::error)
          throw wrong_mnemonics_error(columns);
        else if (statement == bit_statement::assignment)
          initial_permutation_ = read_initial_permutation(columns);
# else // BOOST_NO_CXX11_SCOPED_ENUMS
        bit_statement_::bit_statement const statement
          = read_bit_statement(columns);

        if (statement == bit_statement_::error)
          throw wrong_mnemonics_error(columns);
        else if (statement == bit_statement_::assignment)
          initial_permutation_ = read_initial_permutation(columns);
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
      }
      else if (first_mnemonic == "PERMUTATION")
        throw unsupported_mnemonic_error(first_mnemonic);
      else if (first_mnemonic == "RANDOM") // RANDOM PERMUTATION
        throw unsupported_mnemonic_error(first_mnemonic);
      else if (first_mnemonic == "H")
        data_.push_back(
          boost::movelib::unique_ptr< ::bra::gate::gate >(
            new ::bra::gate::hadamard(read_hadamard(columns))));
      else if (first_mnemonic == "R" or first_mnemonic == "+R")
      {
        qubit_type target;
        int phase_exponent;
        boost::tie(target, phase_exponent) = read_phase_shift(columns);

        if (phase_exponent >= 0)
          data_.push_back(boost::movelib::unique_ptr< ::bra::gate::gate >(
            new ::bra::gate::phase_shift(
              phase_exponent, phase_coefficients_[phase_exponent], target)));
        else
        {
          phase_exponent *= -1;
          data_.push_back(boost::movelib::unique_ptr< ::bra::gate::gate >(
            new ::bra::gate::adj_phase_shift(
              phase_exponent, phase_coefficients_[phase_exponent], target)));
        }
      }
      else if (first_mnemonic == "-R")
      {
        qubit_type target;
        int phase_exponent;
        boost::tie(target, phase_exponent) = read_phase_shift(columns);

        if (phase_exponent >= 0)
          data_.push_back(boost::movelib::unique_ptr< ::bra::gate::gate >(
            new ::bra::gate::adj_phase_shift(
              phase_exponent, phase_coefficients_[phase_exponent], target)));
        else
        {
          phase_exponent *= -1;
          data_.push_back(boost::movelib::unique_ptr< ::bra::gate::gate >(
            new ::bra::gate::phase_shift(
              phase_exponent, phase_coefficients_[phase_exponent], target)));
        }
      }
      else if (first_mnemonic == "+X")
        data_.push_back(
          boost::movelib::unique_ptr< ::bra::gate::gate >(
            new ::bra::gate::x_rotation_half_pi(read_x_rotation_half_pi(columns))));
      else if (first_mnemonic == "-X")
        data_.push_back(
          boost::movelib::unique_ptr< ::bra::gate::gate >(
            new ::bra::gate::adj_x_rotation_half_pi(read_adj_x_rotation_half_pi(columns))));
      else if (first_mnemonic == "+Y")
        data_.push_back(
          boost::movelib::unique_ptr< ::bra::gate::gate >(
            new ::bra::gate::y_rotation_half_pi(read_y_rotation_half_pi(columns))));
      else if (first_mnemonic == "-Y")
        data_.push_back(
          boost::movelib::unique_ptr< ::bra::gate::gate >(
            new ::bra::gate::adj_y_rotation_half_pi(read_adj_y_rotation_half_pi(columns))));
      else if (first_mnemonic == "CNOT")
      {
        control_qubit_type control;
        qubit_type target;
        boost::tie(control, target) = read_controlled_not(columns);

        data_.push_back(
          boost::movelib::unique_ptr< ::bra::gate::gate >(
            new ::bra::gate::controlled_not(target, control)));
      }
      else if (first_mnemonic == "U")
      {
        control_qubit_type control;
        qubit_type target;
        int phase_exponent;
        boost::tie(control, target, phase_exponent) = read_controlled_phase_shift(columns);

        if (phase_exponent >= 0)
          data_.push_back(
            boost::movelib::unique_ptr< ::bra::gate::gate >(
              new ::bra::gate::controlled_phase_shift(
                phase_exponent, phase_coefficients_[phase_exponent], target, control)));
        else
        {
          phase_exponent *= -1;
          data_.push_back(
            boost::movelib::unique_ptr< ::bra::gate::gate >(
              new ::bra::gate::adj_controlled_phase_shift(
                phase_exponent, phase_coefficients_[phase_exponent], target, control)));
        }
      }
      else if (first_mnemonic == "V")
      {
        control_qubit_type control;
        qubit_type target;
        int phase_exponent;
        boost::tie(control, target, phase_exponent) = read_controlled_v(columns);

        if (phase_exponent >= 0)
          data_.push_back(
            boost::movelib::unique_ptr< ::bra::gate::gate >(
              new ::bra::gate::controlled_v(
                phase_exponent, phase_coefficients_[phase_exponent], target, control)));
        else
        {
          phase_exponent *= -1;
          data_.push_back(
            boost::movelib::unique_ptr< ::bra::gate::gate >(
              new ::bra::gate::adj_controlled_v(
                phase_exponent, phase_coefficients_[phase_exponent], target, control)));
        }
      }
      else if (first_mnemonic == "TOFFOLI")
      {
        control_qubit_type control1;
        control_qubit_type control2;
        qubit_type target;
        boost::tie(control1, control2, target) = read_toffoli(columns);

        throw unsupported_mnemonic_error(first_mnemonic);
        /*
        data_.push_back(
          boost::movelib::unique_ptr< ::bra::gate::gate >(
            new ::bra::gate::toffoli(
              phase_exponent, phase_coefficients_[phase_exponent], target, control)));
              */
      }
      else if (first_mnemonic == "X")
      {
        throw unsupported_mnemonic_error(first_mnemonic);
        //data_.push_back(boost::movelib::make_unique< ::bra::gate::x >(read_target(columns)));
      }
      else if (first_mnemonic == "Y")
      {
        throw unsupported_mnemonic_error(first_mnemonic);
        //data_.push_back(boost::movelib::make_unique< ::bra::gate::y >(read_target(columns)));
      }
      else if (first_mnemonic == "Z")
      {
        throw unsupported_mnemonic_error(first_mnemonic);
        //data_.push_back(boost::movelib::make_unique< ::bra::gate::z >(read_target(columns)));
      }
      else if (first_mnemonic == "S")
      {
        throw unsupported_mnemonic_error(first_mnemonic);
        //data_.push_back(boost::movelib::make_unique< ::bra::gate::s >(read_target(columns)));
      }
      else if (first_mnemonic == "S+")
      {
        throw unsupported_mnemonic_error(first_mnemonic);
        //data_.push_back(boost::movelib::make_unique< ::bra::gate::spl >(read_target(columns)));
      }
      else if (first_mnemonic == "T")
      {
        throw unsupported_mnemonic_error(first_mnemonic);
        //data_.push_back(boost::movelib::make_unique< ::bra::gate::t >(read_target(columns)));
      }
      else if (first_mnemonic == "T+")
      {
        throw unsupported_mnemonic_error(first_mnemonic);
        //data_.push_back(boost::movelib::make_unique< ::bra::gate::tpl >(read_target(columns)));
      }
      else if (first_mnemonic == "U1")
      {
        //qubit_type target;
        //int r0;
        //boost::tie(target, r0) = read_target_0(columns);

        throw unsupported_mnemonic_error(first_mnemonic);
        //data_.push_back(boost::movelib::make_unique< ::bra::gate::u1 >(target, r0));
      }
      else if (first_mnemonic == "U2")
      {
        //qubit_type target;
        //int r0, r1;
        //boost::tie(target, r0, r1) = read_target_0_1(columns);

        throw unsupported_mnemonic_error(first_mnemonic);
        //data_.push_back(boost::movelib::make_unique< ::bra::gate::u2 >(target, r0, r1));
      }
      else if (first_mnemonic == "U3")
      {
        //qubit_type target;
        //int r0, r1, r2;
        //boost::tie(target, r0, r1, r2) = read_target_0_1_2(columns);

        throw unsupported_mnemonic_error(first_mnemonic);
        //data_.push_back(boost::movelib::make_unique< ::bra::gate::u3 >(target, r0, r1, r2));
      }
      else if (first_mnemonic == "SHORBOX")
        throw unsupported_mnemonic_error(first_mnemonic);
      else if (first_mnemonic == "BEGIN") // BEGIN MEASUREMENT/LEARNING MACHINE
      {
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
        begin_statement const statement = read_begin_statement(columns);

        if (statement == begin_statement::error)
          throw wrong_mnemonics_error(columns);
        else if (statement == begin_statement::measurement)
          data_.push_back(
            boost::movelib::unique_ptr< ::bra::gate::gate >(new ::bra::gate::measurement(root_)));
        else if (statement == begin_statement::learning_machine)
          throw unsupported_mnemonic_error(first_mnemonic);
# else // BOOST_NO_CXX11_SCOPED_ENUMS
        begin_statement_::begin_statement const statement
          = read_begin_statement(columns);

        if (statement == begin_statement_::error)
          throw wrong_mnemonics_error(columns);
        else if (statement == begin_statement_::measurement)
          data_.push_back(
            boost::movelib::unique_ptr< ::bra::gate::gate >(new ::bra::gate::measurement(root_)));
        else if (statement == begin_statement_::learning_machine)
          throw unsupported_mnemonic_error(first_mnemonic);
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
      }
      else if (first_mnemonic == "DO") // DO MEASUREMENT
      {
        /*
        do_statement const statement = read_do_statement(columns);

        if (statement == do_statement::error)
          throw wrong_mnemonics_error(columns);
        else if (statement == do_statement::measurement)
          throw unsupported_mnemonic_error(first_mnemonic);
          */
        throw unsupported_mnemonic_error(first_mnemonic);
      }
      else if (first_mnemonic == "END") // END MEASUREMENT/LEARNING MACHINE
      {
        /*
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
        end_statement const statement = read_end_statement(columns);

        if (statement == end_statement::error)
          throw wrong_mnemonics_error(columns);
        else if (statement == end_statement::measurement)
          data_.push_back(boost::movelib::make_unique< ::bra::gate::measurement >(root_));
        else if (statement == end_statement::learning_machine)
          throw unsupported_mnemonic_error(first_mnemonic);
# else // BOOST_NO_CXX11_SCOPED_ENUMS
        end_statement_::end_statement const statement
          = read_end_statement(columns);

        if (statement == end_statement_::error)
          throw wrong_mnemonics_error(columns);
        else if (statement == end_statement_::measurement)
          data_.push_back(boost::movelib::make_unique< ::bra::gate::measurement >(root_));
        else if (statement == end_statement_::learning_machine)
          throw unsupported_mnemonic_error(first_mnemonic);
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
*/
        throw unsupported_mnemonic_error(first_mnemonic);
      }
      else if (first_mnemonic == "GENERATE") // GENERATE EVENTS
      {
        //throw unsupported_mnemonic_error(first_mnemonic);
      }
      else
        throw unsupported_mnemonic_error(first_mnemonic);
    }
  }

  void gates::swap(gates& other)
    BOOST_NOEXCEPT_IF(
      ket::utility::is_nothrow_swappable<data_type>::value
      and ket::utility::is_nothrow_swappable<bit_integer_type>::value
      and ket::utility::is_nothrow_swappable<state_integer_type>::value)
  {
    using std::swap;
    swap(data_, other.data_);
    swap(num_qubits_, other.num_qubits_);
    swap(num_lqubits_, other.num_lqubits_);
    swap(initial_state_value_, other.initial_state_value_);
    swap(phase_coefficients_, other.phase_coefficients_);
  }

  gates::bit_integer_type gates::read_num_qubits(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw ::bra::wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type const>::type iter = boost::begin(columns);
    ++iter;
    return boost::lexical_cast<bit_integer_type>(*iter);
  }

  gates::state_integer_type gates::read_initial_state_value(gates::columns_type& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type>::type iter = boost::begin(columns);
    ++iter;
    boost::algorithm::to_upper(*iter);
    if (columns[1] != "STATE")
      throw wrong_mnemonics_error(columns);

    ++iter;
    return boost::lexical_cast<state_integer_type>(*iter);
  }

  gates::bit_integer_type gates::read_num_mpi_processes(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw ::bra::wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type const>::type iter = boost::begin(columns);
    ++iter;
    return boost::lexical_cast<bit_integer_type>(*iter);
  }

  gates::state_integer_type gates::read_mpi_buffer_size(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw ::bra::wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type const>::type iter = boost::begin(columns);
    ++iter;
    return boost::lexical_cast<state_integer_type>(*iter);
  }

  std::vector<gates::qubit_type>
  gates::read_initial_permutation(gates::columns_type const& columns) const
  {
    std::vector<qubit_type> result;
    result.reserve(boost::size(columns)-2u);

    boost::range_iterator<columns_type const>::type iter = boost::begin(columns);
    ++iter;
    ++iter;

    boost::range_iterator<columns_type const>::type last = boost::end(columns);
    for (; iter != last; ++iter)
      result.push_back(static_cast<qubit_type>(boost::lexical_cast<bit_integer_type>(*iter)));

    return result;
  }

  gates::qubit_type gates::read_target(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 2u)
      throw wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type const>::type iter = boost::begin(columns);
    ++iter;
    bit_integer_type const target = boost::lexical_cast<bit_integer_type>(*iter);

    return ket::make_qubit<state_integer_type>(target);
  }

  boost::tuple<gates::qubit_type, int> gates::read_target_phaseexp(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type const>::type iter = boost::begin(columns);
    ++iter;
    bit_integer_type const target = boost::lexical_cast<bit_integer_type>(*iter);
    ++iter;
    int const phase_exponent = boost::lexical_cast<int>(*iter);

    return boost::make_tuple(ket::make_qubit<state_integer_type>(target), phase_exponent);
  }

  boost::tuple<gates::control_qubit_type, gates::qubit_type> gates::read_control_target(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 3u)
      throw wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type const>::type iter = boost::begin(columns);
    ++iter;
    bit_integer_type const control = boost::lexical_cast<bit_integer_type>(*iter);
    ++iter;
    bit_integer_type const target = boost::lexical_cast<bit_integer_type>(*iter);

    return boost::make_tuple(
      ket::make_control(ket::make_qubit<state_integer_type>(control)),
      ket::make_qubit<state_integer_type>(target));
  }

  boost::tuple<gates::control_qubit_type, gates::qubit_type, int> gates::read_control_target_phaseexp(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type const>::type iter = boost::begin(columns);
    ++iter;
    bit_integer_type const control = boost::lexical_cast<bit_integer_type>(*iter);
    ++iter;
    bit_integer_type const target = boost::lexical_cast<bit_integer_type>(*iter);
    ++iter;
    int const phase_exponent = boost::lexical_cast<int>(*iter);

    return boost::make_tuple(
      ket::make_control(ket::make_qubit<state_integer_type>(control)),
      ket::make_qubit<state_integer_type>(target),
      phase_exponent);
  }

  boost::tuple<gates::control_qubit_type, gates::control_qubit_type, gates::qubit_type> gates::read_2controls_target(gates::columns_type const& columns) const
  {
    if (boost::size(columns) != 4u)
      throw wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type const>::type iter = boost::begin(columns);
    ++iter;
    bit_integer_type const control1 = boost::lexical_cast<bit_integer_type>(*iter);
    ++iter;
    bit_integer_type const control2 = boost::lexical_cast<bit_integer_type>(*iter);
    ++iter;
    bit_integer_type const target = boost::lexical_cast<bit_integer_type>(*iter);

    return boost::make_tuple(
      ket::make_control(ket::make_qubit<state_integer_type>(control1)),
      ket::make_control(ket::make_qubit<state_integer_type>(control2)),
      ket::make_qubit<state_integer_type>(target));
  }

# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
  begin_statement gates::read_begin_statement(gates::columns_type& columns) const
  {
    if (boost::size(columns) != 2u)
      throw wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type>::type iter = boost::begin(columns);
    boost::algorithm::to_upper(*++iter);
    if (*iter == "MEASUREMENT")
      return begin_statement::measurement;
    else if (*iter == "LEARNING")
    {
      boost::algorithm::to_upper(*++iter);
      if (*iter == "MACHINE")
        return begin_statement::learning_machine;
    }

    return begin_statement::error;
  }
# else // BOOST_NO_CXX11_SCOPED_ENUMS
  begin_statement_::begin_statement gates::read_begin_statement(gates::columns_type& columns) const
  {
    if (boost::size(columns) != 2u)
      throw wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type>::type iter = boost::begin(columns);
    boost::algorithm::to_upper(*++iter);
    if (*iter == "MEASUREMENT")
      return begin_statement_::measurement;
    else if (*iter == "LEARNING")
    {
      boost::algorithm::to_upper(*++iter);
      if (*iter == "MACHINE")
        return begin_statement_::learning_machine;
    }

    return begin_statement_::error;
  }
# endif // BOOST_NO_CXX11_SCOPED_ENUMS

# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
  bit_statement gates::read_bit_statement(gates::columns_type& columns) const
  {
    if (boost::size(columns) <= 1u)
      throw wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type>::type iter = boost::begin(columns);
    boost::algorithm::to_upper(*++iter);
    if (*iter == "ASSIGNMENT")
      return bit_statement::assignment;

    return bit_statement::error;
  }
# else // BOOST_NO_CXX11_SCOPED_ENUMS
  bit_statement_::bit_statement gates::read_bit_statement(gates::columns_type& columns) const
  {
    if (boost::size(columns) <= 1u)
      throw wrong_mnemonics_error(columns);

    boost::range_iterator<columns_type>::type iter = boost::begin(columns);
    boost::algorithm::to_upper(*++iter);
    if (*iter == "ASSIGNMENT")
      return bit_statement_::assignment;

    return bit_statement_::error;
  }
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
}
