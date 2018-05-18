#ifndef BRA_GATES_HPP
# define BRA_GATES_HPP

# include <boost/config.hpp>

# include <cassert>
# include <iosfwd>
# include <vector>
# include <string>
# include <utility>
# include <stdexcept>
# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
#   include <initializer_list>
# endif

# include <boost/tuple/tuple.hpp>
# include <boost/container/vector.hpp>
# include <boost/move/unique_ptr.hpp>

# include <yampi/allocator.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/utility/is_nothrow_swappable.hpp>

# include <bra/state.hpp>
# include <bra/gate/gate.hpp>


namespace bra
{
  class unsupported_mnemonic_error
    : public std::runtime_error
  {
   public:
    unsupported_mnemonic_error(std::string const& mnemonic);
  };

  class wrong_mnemonics_error
    : public std::runtime_error
  {
   public:
    typedef std::vector<std::string> columns_type;
    wrong_mnemonics_error(columns_type const& columns);

   private:
    std::string generate_what_string(columns_type const& columns);
  };

  class wrong_mpi_communicator_size_error
    : public std::runtime_error
  {
   public:
    wrong_mpi_communicator_size_error();
  };


# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
  enum class begin_statement : int { error, measurement, learning_machine };
  enum class bit_statement : int { error, assignment };
# else // BOOST_NO_CXX11_SCOPED_ENUMS
  namespace begin_statement_ { enum begin_statement { error, measurement, learning_machine }; }
  namespace bit_statement_ { enum bit_statement { error, assignment }; }
# endif // BOOST_NO_CXX11_SCOPED_ENUMS


  class gates
  {
    typedef boost::movelib::unique_ptr< ::bra::gate::gate > value_type_;
    //typedef std::vector<value_type_, yampi::allocator<value_type_> > data_type;
    typedef boost::container::vector<value_type_, yampi::allocator<value_type_> > data_type;
    data_type data_;

   public:
    typedef ::bra::state::bit_integer_type bit_integer_type;
    typedef ::bra::state::state_integer_type state_integer_type;
    typedef ::bra::state::qubit_type qubit_type;
    typedef ::bra::state::control_qubit_type control_qubit_type;

    typedef ::bra::state::real_type real_type;

    typedef wrong_mnemonics_error::columns_type columns_type;

   private:
    bit_integer_type num_qubits_;
    bit_integer_type num_lqubits_;
    state_integer_type initial_state_value_;
    std::vector<qubit_type> initial_permutation_;

    typedef ::bra::state::complex_type complex_type;
    typedef std::vector<complex_type, yampi::allocator<complex_type> > phase_coefficients_type;
    phase_coefficients_type phase_coefficients_;

    yampi::rank root_;

   public:
    typedef data_type::value_type value_type;
    typedef data_type::allocator_type allocator_type;
    typedef data_type::size_type size_type;
    typedef data_type::difference_type difference_type;
    typedef data_type::reference reference;
    typedef data_type::const_reference const_reference;
    typedef data_type::pointer pointer;
    typedef data_type::const_pointer const_pointer;
    typedef data_type::iterator iterator;
    typedef data_type::const_iterator const_iterator;
    typedef data_type::reverse_iterator reverse_iterator;
    typedef data_type::const_reverse_iterator const_reverse_iterator;

    gates();
    explicit gates(allocator_type const& allocator);

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    gates(gates&& other, allocator_type const& allocator);
# endif

    gates(
      std::istream& input_stream, yampi::environment const& environment,
      yampi::rank const root = yampi::rank(),
      yampi::communicator const communicator = yampi::world_communicator(),
      size_type const num_reserved_gates = static_cast<size_type>(0u));

    bool operator==(gates const& other) const;

    bit_integer_type const& num_qubits() const { return num_qubits_; }
    bit_integer_type const& num_lqubits() const { return num_lqubits_; }
    state_integer_type const& initial_state_value() const { return initial_state_value_; }
    std::vector<qubit_type> const& initial_permutation() const { return initial_permutation_; }

    void num_qubits(
      bit_integer_type const new_num_qubits,
      yampi::communicator const communicator, yampi::environment const& environment);
    void num_lqubits(
      bit_integer_type const new_num_lqubits,
      yampi::communicator const communicator, yampi::environment const& environment);

   private:
    void set_num_qubits_params(
      bit_integer_type const new_num_lqubits, bit_integer_type const num_gqubits,
      yampi::communicator const communicator, yampi::environment const& environment);

   public:
    void assign(
      std::istream& input_stream, yampi::environment const& environment,
      yampi::communicator const communicator = yampi::world_communicator(),
      size_type const num_reserved_gates = static_cast<size_type>(0u));
    allocator_type get_allocator() const { return data_.get_allocator(); }

    // Element access
    //reference at(size_type const index) { return data_.at(index); }
    const_reference at(size_type const index) const { return data_.at(index); }
    //reference operator[](size_type const index) { return data_[index]; }
    const_reference operator[](size_type const index) const { return data_[index]; }
    //reference front() { return data_.front(); }
    const_reference front() const { return data_.front(); }
    //reference back() { return data_.back(); }
    const_reference back() const { return data_.back(); }

    // Iterators
    //iterator begin() BOOST_NOEXCEPT { return data_.begin(); }
    const_iterator begin() const BOOST_NOEXCEPT { return data_.begin(); }
    //iterator end() BOOST_NOEXCEPT { return data_.end(); }
    const_iterator end() const BOOST_NOEXCEPT { return data_.end(); }
    //reverse_iterator rbegin() BOOST_NOEXCEPT { return data_.rbegin(); }
    const_reverse_iterator rbegin() const BOOST_NOEXCEPT { return data_.rbegin(); }
    //reverse_iterator rend() BOOST_NOEXCEPT { return data_.rend(); }
    const_reverse_iterator rend() const BOOST_NOEXCEPT { return data_.rend(); }

    // Capacity
    bool empty() const BOOST_NOEXCEPT { return data_.empty(); }
    size_type size() const BOOST_NOEXCEPT { return data_.size(); }
    size_type max_size() const BOOST_NOEXCEPT { return data_.max_size(); }
    void reserve(size_type const new_capacity) { data_.reserve(new_capacity); }
    size_type capacity() const BOOST_NOEXCEPT { return data_.capacity(); }

    // Modifiers
    void clear() BOOST_NOEXCEPT { data_.clear(); }
    /*
# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    iterator insert(const_iterator position, value_type&& value)
    { return data_.insert(position, std::move(value)); }
# endif
# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
    template <typename... Arguments>
    iterator emplace(const_iterator const position, Arguments&&... arguments)
    { return data_.emplace(position, std::forward<Arguments>(arguments)...); }
#   endif
# endif
*/
    iterator erase(iterator const position) { return data_.erase(position); }
    iterator erase(iterator const first, iterator const last) { return data_.erase(first, last); }
    /*
# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    void push_back(value_type&& value) { data_.push_back(std::move(value)); }
#   ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
    template <typename... Arguments>
    void emplace_back(Arguments&&... arguments)
    { data_.emplace_back(std::forward<Arguments>(arguments)...); }
#   endif
# endif
*/
    void pop_back() { data_.pop_back(); }
    //void resize(size_type const count) { data_.resize(count); }
    void swap(gates& other)
      BOOST_NOEXCEPT_IF(
        ket::utility::is_nothrow_swappable<data_type>::value
        and ket::utility::is_nothrow_swappable<bit_integer_type>::value
        and ket::utility::is_nothrow_swappable<state_integer_type>::value);

   private:
    bit_integer_type read_num_qubits(columns_type const& columns) const;
    state_integer_type read_initial_state_value(columns_type& columns) const;
    bit_integer_type read_num_mpi_processes(columns_type const& columns) const;
    state_integer_type read_mpi_buffer_size(columns_type const& columns) const;
    std::vector<qubit_type> read_initial_permutation(columns_type const& columns) const;

    qubit_type read_target(columns_type const& columns) const;
    boost::tuple<qubit_type, real_type> read_target_phase(columns_type const& columns) const;
    boost::tuple<qubit_type, real_type, real_type> read_target_2phases(columns_type const& columns) const;
    boost::tuple<qubit_type, real_type, real_type, real_type> read_target_3phases(columns_type const& columns) const;
    boost::tuple<qubit_type, int> read_target_phaseexp(columns_type const& columns) const;
    boost::tuple<control_qubit_type, qubit_type> read_control_target(columns_type const& columns) const;
    boost::tuple<control_qubit_type, qubit_type, int> read_control_target_phaseexp(columns_type const& columns) const;
    boost::tuple<control_qubit_type, control_qubit_type, qubit_type> read_2controls_target(columns_type const& columns) const;

    qubit_type read_hadamard(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_pauli_x(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_pauli_y(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_pauli_z(columns_type const& columns) const { return read_target(columns); }
    boost::tuple<qubit_type, real_type> read_u1(columns_type const& columns) const { return read_target_phase(columns); }
    boost::tuple<qubit_type, real_type, real_type> read_u2(columns_type const& columns) const { return read_target_2phases(columns); }
    boost::tuple<qubit_type, real_type, real_type, real_type> read_u3(columns_type const& columns) const { return read_target_3phases(columns); }
    boost::tuple<qubit_type, int> read_phase_shift(columns_type const& columns) const { return read_target_phaseexp(columns); }
    qubit_type read_x_rotation_half_pi(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_adj_x_rotation_half_pi(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_y_rotation_half_pi(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_adj_y_rotation_half_pi(columns_type const& columns) const { return read_target(columns); }
    boost::tuple<control_qubit_type, qubit_type> read_controlled_not(columns_type const& columns) const { return read_control_target(columns); }
    boost::tuple<control_qubit_type, qubit_type, int> read_controlled_phase_shift(columns_type const& columns) const { return read_control_target_phaseexp(columns); }
    boost::tuple<control_qubit_type, qubit_type, int> read_controlled_v(columns_type const& columns) const { return read_control_target_phaseexp(columns); }
    boost::tuple<control_qubit_type, control_qubit_type, qubit_type> read_toffoli(columns_type const& columns) const { return read_2controls_target(columns); }

# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
    begin_statement read_begin_statement(columns_type& columns) const;
    bit_statement read_bit_statement(columns_type& columns) const;
# else
    begin_statement_::begin_statement read_begin_statement(columns_type& columns) const;
    bit_statement_::bit_statement read_bit_statement(columns_type& columns) const;
# endif
  };

  inline bool operator!=(::bra::gates const& lhs, ::bra::gates const& rhs)
  { return not (lhs == rhs); }

  inline void swap(::bra::gates& lhs, ::bra::gates& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  inline ::bra::state& operator<<(::bra::state& state, ::bra::gates const& gates)
  {
    ::bra::gates::const_iterator const last = gates.end();
    for (::bra::gates::const_iterator iter = gates.begin(); iter != last; ++iter)
      state << **iter;
    return state;
  }
}


#endif
