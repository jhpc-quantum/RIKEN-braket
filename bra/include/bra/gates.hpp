#ifndef BRA_GATES_HPP
# define BRA_GATES_HPP

# include <cassert>
# include <iosfwd>
# include <vector>
# include <string>
# include <tuple>
# include <algorithm>
# include <iterator>
# include <utility>
# include <memory>
# include <stdexcept>
# include <initializer_list>
# if __cplusplus >= 201703L
#   include <type_traits>
# else
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <boost/lexical_cast.hpp>

# ifndef BRA_NO_MPI
#   include <yampi/allocator.hpp>
#   include <yampi/communicator.hpp>
#   include <yampi/environment.hpp>
# endif // BRA_NO_MPI

# include <bra/state.hpp>
# include <bra/gate/gate.hpp>

# if __cplusplus >= 201703L
#   define BRA_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define BRA_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace bra
{
  class unsupported_mnemonic_error
    : public std::runtime_error
  {
   public:
    unsupported_mnemonic_error(std::string const& mnemonic);
  }; // class unsupported_mnemonic_error

  class wrong_mnemonics_error
    : public std::runtime_error
  {
   public:
    using columns_type = std::vector<std::string>;
    wrong_mnemonics_error(columns_type const& columns);

   private:
    std::string generate_what_string(columns_type const& columns);
  }; // class wrong_mnemonic_error

# ifndef BRA_NO_MPI
  class wrong_mpi_communicator_size_error
    : public std::runtime_error
  {
   public:
    wrong_mpi_communicator_size_error();
  }; // class wrong_mpi_communicator_size_error
# endif // BRA_NO_MPI

  enum class begin_statement : int { measurement, learning_machine };
  enum class bit_statement : int { assignment };
  enum class generate_statement : int { events };
  enum class depolarizing_statement : int { channel };

  class gates
  {
    using value_type_ = std::unique_ptr< ::bra::gate::gate >;
# ifndef BRA_NO_MPI
    using data_type = std::vector<value_type_, yampi::allocator<value_type_>>;
# else
    using data_type = std::vector<value_type_>;
# endif
    data_type data_;

   public:
    using bit_integer_type = ::bra::state::bit_integer_type;
    using state_integer_type = ::bra::state::state_integer_type;
    using qubit_type = ::bra::state::qubit_type;
    using control_qubit_type = ::bra::state::control_qubit_type;

    using real_type = ::bra::state::real_type;

    using columns_type = wrong_mnemonics_error::columns_type;

   private:
    bit_integer_type num_qubits_;
# ifndef BRA_NO_MPI
    bit_integer_type num_lqubits_;
    bit_integer_type num_uqubits_;
    unsigned int num_processes_per_unit_;
# endif
    state_integer_type initial_state_value_;
# ifndef BRA_NO_MPI
    std::vector<qubit_type> initial_permutation_;
# endif

    using complex_type = ::bra::state::complex_type;
# ifndef BRA_NO_MPI
    using phase_coefficients_type = std::vector<complex_type, yampi::allocator<complex_type>>;
# else
    using phase_coefficients_type = std::vector<complex_type>;
# endif
    phase_coefficients_type phase_coefficients_;

# ifndef BRA_NO_MPI
    yampi::rank root_;
# endif

   public:
    using value_type = data_type::value_type;
    using allocator_type = data_type::allocator_type;
    using size_type = data_type::size_type;
    using difference_type = data_type::difference_type;
    using reference = data_type::reference;
    using const_reference = data_type::const_reference;
    using pointer = data_type::pointer;
    using const_pointer = data_type::const_pointer;
    using iterator = data_type::iterator;
    using const_iterator = data_type::const_iterator;
    using reverse_iterator = data_type::reverse_iterator;
    using const_reverse_iterator = data_type::const_reverse_iterator;

    gates();
    explicit gates(allocator_type const& allocator);

    gates(gates&& other, allocator_type const& allocator);

# ifndef BRA_NO_MPI
    gates(
      std::istream& input_stream,
      bit_integer_type num_uqubits, unsigned int num_processes_per_unit,
      yampi::environment const& environment,
      yampi::rank const root = yampi::rank{},
      yampi::communicator const& communicator = yampi::communicator{::yampi::world_communicator_t()},
      size_type const num_reserved_gates = size_type{0u});
# else // BRA_NO_MPI
    explicit gates(std::istream& input_stream);
    gates(std::istream& input_stream, size_type const num_reserved_gates);
# endif // BRA_NO_MPI

    bool operator==(gates const& other) const;

    bit_integer_type const& num_qubits() const { return num_qubits_; }
# ifndef BRA_NO_MPI
    bit_integer_type const& num_lqubits() const { return num_lqubits_; }
    bit_integer_type const& num_uqubits() const { return num_uqubits_; }
    unsigned int const& num_processes_per_unit() const { return num_processes_per_unit_; }
# endif
    state_integer_type const& initial_state_value() const { return initial_state_value_; }
# ifndef BRA_NO_MPI
    std::vector<qubit_type> const& initial_permutation() const { return initial_permutation_; }
# endif

# ifndef BRA_NO_MPI
    void num_qubits(
      bit_integer_type const new_num_qubits,
      yampi::communicator const& communicator, yampi::environment const& environment);
    void num_lqubits(
      bit_integer_type const new_num_lqubits,
      yampi::communicator const& communicator, yampi::environment const& environment);
# else // BRA_NO_MPI
    void num_qubits(bit_integer_type const new_num_qubits);
# endif // BRA_NO_MPI

   private:
# ifndef BRA_NO_MPI
    void set_num_qubits_params(
      bit_integer_type const new_num_lqubits, bit_integer_type const num_gqubits,
      yampi::communicator const& communicator, yampi::environment const& environment);
# else // BRA_NO_MPI
    void set_num_qubits_params(bit_integer_type const new_num_qubits);
# endif // BRA_NO_MPI

   public:
# ifndef BRA_NO_MPI
    void assign(
      std::istream& input_stream, yampi::environment const& environment,
      yampi::communicator const& communicator = yampi::communicator{yampi::world_communicator_t()},
      size_type const num_reserved_gates = size_type{0u});
# else // BRA_NO_MPI
    void assign(
      std::istream& input_stream,
      size_type const num_reserved_gates = size_type{0u});
# endif // BRA_NO_MPI
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
    //iterator begin() noexcept { return data_.begin(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    //iterator end() noexcept { return data_.end(); }
    const_iterator end() const noexcept { return data_.end(); }
    //reverse_iterator rbegin() noexcept { return data_.rbegin(); }
    const_reverse_iterator rbegin() const noexcept { return data_.rbegin(); }
    //reverse_iterator rend() noexcept { return data_.rend(); }
    const_reverse_iterator rend() const noexcept { return data_.rend(); }

    // Capacity
    bool empty() const noexcept { return data_.empty(); }
    size_type size() const noexcept { return data_.size(); }
    size_type max_size() const noexcept { return data_.max_size(); }
    void reserve(size_type const new_capacity) { data_.reserve(new_capacity); }
    size_type capacity() const noexcept { return data_.capacity(); }

    // Modifiers
    void clear() noexcept { data_.clear(); }
    /*
    iterator insert(const_iterator position, value_type&& value)
    { return data_.insert(position, std::move(value)); }
    template <typename... Arguments>
    iterator emplace(const_iterator const position, Arguments&&... arguments)
    { return data_.emplace(position, std::forward<Arguments>(arguments)...); }
*/
    iterator erase(iterator const position) { return data_.erase(position); }
    iterator erase(iterator const first, iterator const last) { return data_.erase(first, last); }
    /*
    void push_back(value_type&& value) { data_.push_back(std::move(value)); }
    template <typename... Arguments>
    void emplace_back(Arguments&&... arguments)
    { data_.emplace_back(std::forward<Arguments>(arguments)...); }
*/
    void pop_back() { data_.pop_back(); }
    //void resize(size_type const count) { data_.resize(count); }
    void swap(gates& other)
      noexcept(
        BRA_is_nothrow_swappable<data_type>::value
        and BRA_is_nothrow_swappable<bit_integer_type>::value
        and BRA_is_nothrow_swappable<state_integer_type>::value
        and BRA_is_nothrow_swappable<qubit_type>::value);

   private:
    bit_integer_type read_num_qubits(columns_type const& columns) const;
    state_integer_type read_initial_state_value(columns_type& columns) const;
    bit_integer_type read_num_mpi_processes(columns_type const& columns) const;
    state_integer_type read_mpi_buffer_size(columns_type const& columns) const;
# ifndef BRA_NO_MPI
    std::vector<qubit_type> read_initial_permutation(columns_type const& columns) const;
# endif

    qubit_type read_target(columns_type const& columns) const;
    std::tuple<qubit_type, real_type> read_target_phase(columns_type const& columns) const;
    std::tuple<qubit_type, real_type, real_type> read_target_2phases(columns_type const& columns) const;
    std::tuple<qubit_type, real_type, real_type, real_type> read_target_3phases(columns_type const& columns) const;
    std::tuple<qubit_type, int> read_target_phaseexp(columns_type const& columns) const;
    std::tuple<control_qubit_type, qubit_type> read_control_target(columns_type const& columns) const;
    std::tuple<control_qubit_type, qubit_type, int> read_control_target_phaseexp(columns_type const& columns) const;
    std::tuple<control_qubit_type, control_qubit_type, qubit_type> read_2controls_target(columns_type const& columns) const;

    qubit_type read_hadamard(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_pauli_x(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_pauli_y(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_pauli_z(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_s_gate(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_adj_s_gate(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_t_gate(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_adj_t_gate(columns_type const& columns) const { return read_target(columns); }
    std::tuple<qubit_type, real_type> read_u1(columns_type const& columns) const { return read_target_phase(columns); }
    std::tuple<qubit_type, real_type, real_type> read_u2(columns_type const& columns) const { return read_target_2phases(columns); }
    std::tuple<qubit_type, real_type, real_type, real_type> read_u3(columns_type const& columns) const { return read_target_3phases(columns); }
    std::tuple<qubit_type, int> read_phase_shift(columns_type const& columns) const { return read_target_phaseexp(columns); }
    qubit_type read_x_rotation_half_pi(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_adj_x_rotation_half_pi(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_y_rotation_half_pi(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_adj_y_rotation_half_pi(columns_type const& columns) const { return read_target(columns); }
    std::tuple<control_qubit_type, qubit_type> read_controlled_not(columns_type const& columns) const { return read_control_target(columns); }
    std::tuple<control_qubit_type, qubit_type, int> read_controlled_phase_shift(columns_type const& columns) const { return read_control_target_phaseexp(columns); }
    std::tuple<control_qubit_type, qubit_type, int> read_controlled_v(columns_type const& columns) const { return read_control_target_phaseexp(columns); }
    std::tuple<control_qubit_type, control_qubit_type, qubit_type> read_toffoli(columns_type const& columns) const { return read_2controls_target(columns); }
    qubit_type read_projective_measurement(columns_type const& columns) const { return read_target(columns); }

    ::bra::begin_statement read_begin_statement(columns_type& columns) const;
    ::bra::bit_statement read_bit_statement(columns_type& columns) const;
    std::tuple<bit_integer_type, state_integer_type, state_integer_type> read_shor_box(columns_type const& columns) const;
    std::tuple< ::bra::generate_statement, int, int > read_generate_statement(columns_type& columns) const;
    qubit_type read_clear(columns_type const& columns) const { return read_target(columns); }
    qubit_type read_set(columns_type const& columns) const { return read_target(columns); }
    std::tuple< ::bra::depolarizing_statement, real_type, real_type, real_type, int > read_depolarizing_statement(columns_type& columns) const;
  }; // class gates

  inline bool operator!=(::bra::gates const& lhs, ::bra::gates const& rhs)
  { return not (lhs == rhs); }

  inline void swap(::bra::gates& lhs, ::bra::gates& rhs)
    noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  inline ::bra::state& operator<<(::bra::state& state, ::bra::gates const& gates)
  {
    for (auto const& gate_ptr: gates)
      state << *gate_ptr;
    return state;
  }

  namespace gates_detail
  {
    template <typename Value>
    void read_value_in_depolarizing_statement(
      Value& value, std::string& present_string, // "xxx" or "xxx," or "xxx,..."
      ::bra::gates::columns_type::const_iterator& column_iter,
      ::bra::gates::columns_type::const_iterator const& column_last,
      ::bra::gates::columns_type const& columns)
    {
      auto string_found = std::find(present_string.cbegin(), present_string.cend(), ',');
      value = boost::lexical_cast<Value>(std::string{present_string.cbegin(), string_found});

      if (string_found == present_string.cend()) // present_string == "xxx"
      {
        if (++column_iter == column_last)
        {
          present_string.clear();
          return;
        }

        present_string = *column_iter; // present_string == "," or ",..."
        if (present_string[0] != ',')
          throw wrong_mnemonics_error{columns};

        if (present_string.size() == 1u) // present_string == ","
        {
          if (++column_iter == column_last)
            throw wrong_mnemonics_error{columns};

          present_string = *column_iter; // present_string == "..."
        }
        else // present_string == ",..."
          present_string.assign(present_string, 1u, std::string::npos); // present_string == "..."
      }
      else // present_string == "xxx," or "xxx,..."
      {
        present_string.assign(++string_found, present_string.cend()); // present_string == "" or "..."
        if (present_string.empty()) // present_string == ""
        {
          if (++column_iter == column_last)
            throw wrong_mnemonics_error{columns};

          present_string = *column_iter; // present_string == "..."
        }
      }
    }

    template <typename Value>
    void read_depolarizing_statement(
      Value& value, std::string& present_string,
      ::bra::gates::columns_type::const_iterator& column_iter,
      ::bra::gates::columns_type::const_iterator const& column_last,
      std::string::const_iterator string_found, ::bra::gates::columns_type const& columns)
    {
      if (string_found == present_string.cend()) // present_string == "XXX"
      {
        if (++column_iter == column_last)
          throw wrong_mnemonics_error{columns};

        present_string = *column_iter; // present_string == "=" or "=xxx" or "=xxx," or "=xxx,..."
        if (present_string[0] != '=')
          throw wrong_mnemonics_error{columns};

        if (present_string.size() == 1u) // present_string == "="
        {
          if (++column_iter == column_last)
            throw wrong_mnemonics_error{columns};

          present_string = *column_iter; // present_string == "xxx" or "xxx," or "xxx,..."
        }
        else // presnet_string == "=xxx" or "=xxx," or "=xxx,..."
          present_string.assign(present_string, 1u, std::string::npos); // presnet_string == "xxx" or "xxx," or "xxx,..."
      }
      else // present_string == "XXX=" or "XXX=xxx" or "XXX=xxx," or "XXX=xxx,..."
      {
        present_string.assign(++string_found, present_string.cend()); // present_string == "" or "xxx" or "xxx," or "xxx,..."
        if (present_string.empty()) // present_string == ""
        {
          if (++column_iter == column_last)
            throw wrong_mnemonics_error{columns};

          present_string = *column_iter; // present_string == "xxx" or "xxx," or "xxx,..."
        }
      }

      read_value_in_depolarizing_statement(value, present_string, column_iter, column_last, columns);
    }
  } // namespace gates_detail
} // namespace bra


# undef BRA_is_nothrow_swappable

#endif // BRA_GATES_HPP
