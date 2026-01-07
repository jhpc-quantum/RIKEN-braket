#ifndef BRA_STATE_HPP
# define BRA_STATE_HPP

# include <cstddef>
# include <complex>
# include <string>
# include <vector>
# include <array>
# include <unordered_map>
# include <utility>
# ifdef BRA_NO_MPI
#   include <chrono>
# endif // BRA_NO_MPI
# include <memory>
# include <random>
# include <stdexcept>

# include <boost/optional.hpp>
# include <boost/variant.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/projective_measurement.hpp>
# include <ket/utility/generate_phase_coefficients.hpp>
# ifndef BRA_NO_MPI
#   include <ket/mpi/permutated.hpp>
#   include <ket/mpi/qubit_permutation.hpp>

#   include <yampi/rank.hpp>
#   include <yampi/communicator.hpp>
#   include <yampi/intercommunicator.hpp>
#   include <yampi/environment.hpp>
#   include <yampi/wall_clock.hpp>
# endif // BRA_NO_MPI

# include <bra/types.hpp>
# include <bra/pauli_string_space.hpp>

# ifndef BRA_NO_MPI
#   define BRA_clock yampi::wall_clock
# else // BRA_NO_MPI
#   define BRA_clock std::chrono::system_clock
# endif // BRA_NO_MPI

# ifndef BRA_MAX_NUM_FUSED_QUBITS
#   ifdef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS BOOST_PP_DEC(KET_DEFAULT_NUM_ON_CACHE_QUBITS)
#   else // KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define BRA_MAX_NUM_FUSED_QUBITS 10
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
# endif // BRA_MAX_NUM_FUSED_QUBITS


namespace bra
{
  enum class variable_type : int
  { real = 0, complex_ = 1, integer = 2, pauli_string_space = 3 };

  enum class assign_operation_type : int
  { assign = 0, plus_assign = 1, minus_assign = 2, multiplies_assign = 3, divides_assign = 4 };

  enum class compare_operation_type : int
  { equal_to = 0, not_equal_to = 1, greater = 2, less = 3, greater_equal = 4, less_equal = 5 };

  class too_many_operated_qubits_error
    : public std::runtime_error
  {
   public:
    too_many_operated_qubits_error(std::size_t const num_operated_qubits, std::size_t const max_num_operated_qubits);
  }; // class too_many_operated_qubits_error

  class unsupported_fused_gate_error
    : public std::runtime_error
  {
   public:
    unsupported_fused_gate_error(std::string const& mnemonic);
  }; // class unsupported_fused_gate_error

  class wrong_assignment_argument_error
    : public std::runtime_error
  {
   public:
    wrong_assignment_argument_error(std::string const& lhs_variable_name, ::bra::assign_operation_type const op, std::string const& rhs_literal_or_variable_name);

   private:
    std::string to_string(::bra::assign_operation_type const op);
  }; // class wrong_assignment_argument_error

  class wrong_comparison_argument_error
    : public std::runtime_error
  {
   public:
    wrong_comparison_argument_error(std::string const& lhs_variable_name, ::bra::compare_operation_type const op, std::string const& rhs_literal_or_variable_name);

   private:
    std::string to_string(::bra::compare_operation_type const op);
  }; // class wrong_comparison_argument_error

  class wrong_pauli_string_length_error
    : public std::runtime_error
  {
   public:
    wrong_pauli_string_length_error(std::size_t const num_operated_qubits, std::size_t const pauli_string_length);
  }; // class wrong_pauli_string_length_error

  namespace state_detail
  {
    template <typename StateInteger, typename BitInteger>
    std::string integer_to_bits_string(StateInteger const integer, BitInteger const total_num_qubits)
    {
      auto result = std::string{};
      result.reserve(total_num_qubits);

      for (auto left_bit = total_num_qubits; left_bit > BitInteger{0u}; --left_bit)
      {
        auto const zero_or_one
          = (integer bitand (StateInteger{1u} << (left_bit - BitInteger{1u})))
            >> (left_bit - BitInteger{1u});
        if (zero_or_one == StateInteger{0u})
          result.push_back('0');
        else
          result.push_back('1');
      }
      return result;
    }

    template <typename Clock, typename Duration>
    double duration_to_second(
      std::chrono::time_point<Clock, Duration> const& from,
      std::chrono::time_point<Clock, Duration> const& to)
    { return 0.000001 * std::chrono::duration_cast<std::chrono::microseconds>(to - from).count(); }
  }

  enum class found_qubit : int
  { not_found = 0, control_qubit = 1, ez_qubit = 2, cez_qubit = 3, qubit = 4 };

  inline auto is_weaker(::bra::found_qubit const lhs, ::bra::found_qubit const rhs) -> bool
  { return static_cast<int>(lhs) < static_cast<int>(rhs); }

  template <typename FoundQubitsAllocator>
  inline auto set_found_qubits(
    std::vector< ::bra::found_qubit, FoundQubitsAllocator >& found_qubits,
    ::bra::qubit_type const qubit)
  -> void
  {
    /*
    if (::bra::is_weaker(found_qubits[static_cast< ::bra::bit_integer_type >(qubit)], ::bra::found_qubit::qubit)
      found_qubits[static_cast< ::bra::bit_integer_type >(qubit)] = ::bra::found_qubit::qubit;
    */
    found_qubits[static_cast< ::bra::bit_integer_type >(qubit)] = ::bra::found_qubit::qubit;
  }

  template <typename FoundQubitsAllocator>
  inline auto set_found_qubits(
    std::vector< ::bra::found_qubit, FoundQubitsAllocator >& found_qubits,
    ::bra::control_qubit_type const control_qubit)
  -> void
  {
    if (::bra::is_weaker(found_qubits[static_cast< ::bra::bit_integer_type >(control_qubit.qubit())], ::bra::found_qubit::control_qubit))
      found_qubits[static_cast< ::bra::bit_integer_type >(control_qubit.qubit())] = ::bra::found_qubit::control_qubit;
  }

  template <typename FoundQubitsAllocator, typename Allocator>
  inline auto set_found_qubits(
    std::vector< ::bra::found_qubit, FoundQubitsAllocator >& found_qubits,
    std::vector< ::bra::qubit_type, Allocator > const& qubits)
  -> void
  {
    for (auto const qubit: qubits)
      set_found_qubits(found_qubits, qubit);
  }

  template <typename FoundQubitsAllocator, typename Allocator>
  inline auto set_found_qubits(
    std::vector< ::bra::found_qubit, FoundQubitsAllocator >& found_qubits,
    std::vector< ::bra::control_qubit_type, Allocator> const& control_qubits)
  -> void
  {
    for (auto const control_qubit: control_qubits)
      set_found_qubits(found_qubits, control_qubit);
  }

  class wait_reason
  {
    enum class status_t : int
    {
      no_wait = 0,
      inner_product, inner_product_all, inner_product_op, inner_product_all_op,
      fidelity, fidelity_all, fidelity_op, fidelity_all_op
    };

    status_t status_;
    int other_circuit_index_;
    std::string operator_literal_or_variable_name_;
    std::vector< ::bra::qubit_type > operated_qubits_;

   public:
    struct no_wait_t { };
    struct inner_product_t { };
    struct inner_product_all_t { };
    struct inner_product_op_t { };
    struct inner_product_all_op_t { };
    struct fidelity_t { };
    struct fidelity_all_t { };
    struct fidelity_op_t { };
    struct fidelity_all_op_t { };

    wait_reason(no_wait_t const)
      : status_{status_t::no_wait}, other_circuit_index_{},
        operator_literal_or_variable_name_{}, operated_qubits_{}
    { }

    wait_reason(inner_product_t const, int const other_circuit_index)
      : status_{status_t::inner_product}, other_circuit_index_{other_circuit_index},
        operator_literal_or_variable_name_{}, operated_qubits_{}
    { }

    wait_reason(inner_product_all_t const)
      : status_{status_t::inner_product_all}, other_circuit_index_{},
        operator_literal_or_variable_name_{}, operated_qubits_{}
    { }

    wait_reason(
      inner_product_op_t const, int const other_circuit_index,
      std::string const& operator_literal_or_variable_name,
      std::vector<qubit_type> const& operated_qubits)
      : status_{status_t::inner_product_op}, other_circuit_index_{other_circuit_index},
        operator_literal_or_variable_name_{operator_literal_or_variable_name},
        operated_qubits_{operated_qubits}
    { }

    wait_reason(
      inner_product_all_op_t const,
      std::string const& operator_literal_or_variable_name,
      std::vector<qubit_type> const& operated_qubits)
      : status_{status_t::inner_product_all_op}, other_circuit_index_{},
        operator_literal_or_variable_name_{operator_literal_or_variable_name},
        operated_qubits_{operated_qubits}
    { }

    wait_reason(fidelity_t const, int const other_circuit_index)
      : status_{status_t::fidelity}, other_circuit_index_{other_circuit_index},
        operator_literal_or_variable_name_{}, operated_qubits_{}
    { }

    wait_reason(fidelity_all_t const)
      : status_{status_t::fidelity_all}, other_circuit_index_{},
        operator_literal_or_variable_name_{}, operated_qubits_{}
    { }

    wait_reason(
      fidelity_op_t const, int const other_circuit_index,
      std::string const& operator_literal_or_variable_name,
      std::vector<qubit_type> const& operated_qubits)
      : status_{status_t::fidelity_op}, other_circuit_index_{other_circuit_index},
        operator_literal_or_variable_name_{operator_literal_or_variable_name},
        operated_qubits_{operated_qubits}
    { }

    wait_reason(
      fidelity_all_op_t const,
      std::string const& operator_literal_or_variable_name,
      std::vector<qubit_type> const& operated_qubits)
      : status_{status_t::fidelity_all_op}, other_circuit_index_{},
        operator_literal_or_variable_name_{operator_literal_or_variable_name},
        operated_qubits_{operated_qubits}
    { }

    auto is_inner_product() const -> bool { return status_ == status_t::inner_product; };
    auto is_inner_product_all() const -> bool { return status_ == status_t::inner_product_all; };
    auto is_inner_product_op() const -> bool { return status_ == status_t::inner_product_op; };
    auto is_inner_product_all_op() const -> bool { return status_ == status_t::inner_product_all_op; };
    auto is_fidelity() const -> bool { return status_ == status_t::fidelity; };
    auto is_fidelity_all() const -> bool { return status_ == status_t::fidelity_all; };
    auto is_fidelity_op() const -> bool { return status_ == status_t::fidelity_op; };
    auto is_fidelity_all_op() const -> bool { return status_ == status_t::fidelity_all_op; };

    auto other_circuit_index() const -> int { return other_circuit_index_; }
    auto operator_literal_or_variable_name() const -> std::string const& { return operator_literal_or_variable_name_; }
    auto operated_qubits() const -> std::vector< ::bra::qubit_type > const& { return operated_qubits_; }
  }; // class wait_reason

  class state
  {
   public:
    using state_integer_type = ::bra::state_integer_type;
    using bit_integer_type = ::bra::bit_integer_type;
    using qubit_type = ::bra::qubit_type;
    using control_qubit_type = ::bra::control_qubit_type;
# ifndef BRA_NO_MPI
    using permutated_qubit_type = ::bra::permutated_qubit_type;
    using permutated_control_qubit_type = ::bra::permutated_control_qubit_type;
# endif // BRA_NO_MPI

    using real_type = ::bra::real_type;
    using complex_type = ::bra::complex_type;
    using int_type = ::bra::int_type;

    using spin_type = std::array<real_type, 3u>;
    using spins_type = std::vector<spin_type>;
    using random_number_generator_type = std::mt19937_64;
    using seed_type = random_number_generator_type::result_type;

# ifndef BRA_NO_MPI
    using permutation_type
      = ket::mpi::qubit_permutation<state_integer_type, bit_integer_type>;
# endif // BRA_NO_MPI

   protected:
    bit_integer_type total_num_qubits_;
    std::vector<ket::gate::outcome> last_outcomes_; // return values of ket(::mpi)::gate::projective_measurement
    qubit_type last_measured_qubit_; // return values of ket(::mpi)::gate::projective_measurement
    boost::optional<spins_type> maybe_expectation_values_; // return value of ket(::mpi)::all_spin_expectation_values
    state_integer_type measured_value_; // return value of ket(::mpi)::measure
    std::vector<state_integer_type> generated_events_; // results of ket(::mpi)::generate_events
    ::bra::complex_type result_; // return value of ket(::mpi)::expectation_value, ket(::mpi)::inner_product, or ket(::mpi)::fidelity
    bool is_in_fusion_; // related to begin_fusion/end_fusion
    std::vector< ::bra::found_qubit > found_qubits_; // related to begin_fusion/end_fusion
    int circuit_index_;
    ::bra::wait_reason wait_reason_;
    random_number_generator_type random_number_generator_;
# ifndef BRA_NO_MPI

    permutation_type permutation_;
    ::bra::data_type buffer_;
    yampi::communicator const& circuit_communicator_;
    yampi::communicator const& intercircuit_communicator_;
    std::vector<yampi::intercommunicator> const& intercommunicators_;
    yampi::environment const& environment_;
# endif // BRA_NO_MPI

    BRA_clock::time_point start_time_;
    BRA_clock::time_point last_processed_time_;

    using phase_coefficients_type = std::vector< ::bra::complex_type >;
    phase_coefficients_type phase_coefficients_;

    boost::optional<std::string> maybe_label_;

    using real_variables_type = std::unordered_map<std::string, std::vector<real_type>>;
    real_variables_type real_variables_;

    using complex_variables_type = std::unordered_map<std::string, std::vector<complex_type>>;
    complex_variables_type complex_variables_;

    using int_variables_type = std::unordered_map<std::string, std::vector<int_type>>;
    int_variables_type int_variables_;

    using pauli_string_space_variables_type = std::unordered_map<std::string, std::vector< ::bra::pauli_string_space >>;
    pauli_string_space_variables_type pauli_string_space_variables_;

   public:
# ifndef BRA_NO_MPI
    state(
      bit_integer_type const total_num_qubits,
      seed_type const seed,
      yampi::communicator const& circuit_communicator,
      yampi::communicator const& intercircuit_communicator,
      int const circuit_index,
      std::vector<yampi::intercommunicator> const& intercommunicators,
      yampi::environment const& environment);

    state(
      bit_integer_type const total_num_qubits,
      seed_type const seed,
      unsigned int const num_elements_in_buffer,
      yampi::communicator const& circuit_communicator,
      yampi::communicator const& intercircuit_communicator,
      int const circuit_index,
      std::vector<yampi::intercommunicator> const& intercommunicators,
      yampi::environment const& environment);

    state(
      std::vector<permutated_qubit_type> const& initial_permutation,
      seed_type const seed,
      yampi::communicator const& circuit_communicator,
      yampi::communicator const& intercircuit_communicator,
      int const circuit_index,
      std::vector<yampi::intercommunicator> const& intercommunicators,
      yampi::environment const& environment);

    state(
      std::vector<permutated_qubit_type> const& initial_permutation,
      seed_type const seed,
      unsigned int const num_elements_in_buffer,
      yampi::communicator const& circuit_communicator,
      yampi::communicator const& intercircuit_communicator,
      int const circuit_index,
      std::vector<yampi::intercommunicator> const& intercommunicators,
      yampi::environment const& environment);
# else // BRA_NO_MPI
    state(bit_integer_type const total_num_qubits, seed_type const seed, int const circuit_index);
# endif // BRA_NO_MPI

    virtual ~state() = default;
    state(state const&) = default;
    state& operator=(state const&) = default;
    state(state&&) = default;
    state& operator=(state&&) = default;

    bit_integer_type const& total_num_qubits() const { return total_num_qubits_; }

    bool is_measured(qubit_type const qubit) const
    { return last_outcomes_[static_cast<bit_integer_type>(qubit)] != ket::gate::outcome::unspecified; }
    int outcome(qubit_type const qubit) const
    { return static_cast<int>(last_outcomes_[static_cast<bit_integer_type>(qubit)]); }

    boost::optional<spins_type> const& maybe_expectation_values() const
    { return maybe_expectation_values_; }
    state_integer_type const& measured_value() const { return measured_value_; }
    std::vector<state_integer_type> const& generated_events() const { return generated_events_; }
    random_number_generator_type& random_number_generator() { return random_number_generator_; }

# ifndef BRA_NO_MPI
    permutation_type const& permutation() const { return permutation_; }

    yampi::communicator const& communicator() const { return circuit_communicator_; }
    yampi::communicator const& intercircuit_communicator() const { return intercircuit_communicator_; }
    std::vector<yampi::intercommunicator> const& intercommunicators() const { return intercommunicators_; }
    yampi::environment const& environment() const { return environment_; }
# endif // BRA_NO_MPI

    void generate_new_variable(std::string const& variable_name, ::bra::variable_type const type, int const num_elements);

   private:
    void generate_new_real_variable(std::string const& variable_name, int const num_elements);
    void generate_new_complex_variable(std::string const& variable_name, int const num_elements);
    void generate_new_int_variable(std::string const& variable_name, int const num_elements);
    void generate_new_pauli_string_space_variable(std::string const& variable_name, int const num_elements);

   public:
    void invoke_assign_operation(std::string const& lhs_variable_name, ::bra::assign_operation_type const op, std::string const& rhs_literal_or_variable_name);

   private:
    void generate_print_string(std::ostringstream& oss, std::vector<std::string> const& variables_or_literals);

   public:
    void invoke_print_operation(std::vector<std::string> const& variables_or_literals);
    void invoke_println_operation(std::vector<std::string> const& variables_or_literals);

    void invoke_jump_operation(std::string const& label);
    void invoke_jump_operation(std::string const& label, std::string const& lhs_variable_name, ::bra::compare_operation_type const op, std::string const& rhs_literal_or_variable_name);
    boost::optional<std::string> const& maybe_label() const { return maybe_label_; }
    void delete_label() { maybe_label_ = boost::none; }

    auto is_waiting() const -> bool { return do_is_waiting(); }
    ::bra::wait_reason const& wait_reason() const { return wait_reason_; }
    void cancel_waiting() { do_cancel_waiting(); wait_reason_ = ::bra::wait_reason{::bra::wait_reason::no_wait_t{}}; }

    auto send_variable(int const destination_circuit_index, std::string const& variable_name, ::bra::variable_type const type, int const num_elements) const -> void;
    auto receive_variable(int const source_circuit_index, std::string const& variable_name, ::bra::variable_type const type, int const num_elements) -> void;

# ifndef BRA_NO_MPI
    unsigned int num_page_qubits() const { return do_num_page_qubits(); }
    unsigned int num_pages() const { return do_num_pages(); }
# endif // BRA_NO_MPI

   protected:
    auto is_int_symbol(std::string const& symbol_name) const -> bool;
    auto to_int(std::string const& colon_separated_string) const -> int_type;
    auto to_int_variable(std::string const& colon_separated_string) const -> int_type const&;
    auto to_int_variable(std::string const& colon_separated_string) -> int_type&;
    auto is_real_symbol(std::string const& symbol_name) const -> bool;
    auto to_real(std::string const& colon_separated_string) const -> real_type;
    auto to_real_variable(std::string const& colon_separated_string) const -> real_type const&;
    auto to_real_variable(std::string const& colon_separated_string) -> real_type&;
    auto is_complex_symbol(std::string const& symbol_name) const -> bool;
    auto to_complex(std::string const& colon_separated_string) const -> complex_type;
    auto to_complex_variable(std::string const& colon_separated_string) const -> complex_type const&;
    auto to_complex_variable(std::string const& colon_separated_string) -> complex_type&;
    auto is_pauli_string_space_symbol(std::string const& symbol_name) const -> bool;
    auto to_pauli_string_space(std::string const& colon_separated_string) const -> ::bra::pauli_string_space;
    auto to_pauli_string_space_variable(std::string const& colon_separated_string) const -> ::bra::pauli_string_space const&;
    auto to_pauli_string_space_variable(std::string const& colon_separated_string) -> ::bra::pauli_string_space&;

   private:
    friend class real_visitor;
    friend class int_visitor;

    class real_visitor
      : public boost::static_visitor<real_type>
    {
      state const* state_ptr_;

     public:
      real_visitor(state const& s) : state_ptr_{std::addressof(s)} { }

      real_type operator()(real_type const value) const { return value; }

      real_type operator()(std::string const& colon_separated_string) const
      { return state_ptr_->to_real(colon_separated_string); }
    }; // class real_visitor

    class int_visitor
      : public boost::static_visitor<int_type>
    {
      state const* state_ptr_;

     public:
      int_visitor(state const& s) : state_ptr_{std::addressof(s)} { }

      int_type operator()(int_type const value) const { return value; }

      int_type operator()(std::string const& colon_separated_string) const
      { return state_ptr_->to_int(colon_separated_string); }
    }; // class int_visitor

   public:
    state& i_gate(qubit_type const qubit);
    state& ic_gate(control_qubit_type const control_qubit);
    state& ii_gate(qubit_type const qubit1, qubit_type const qubit2);
    state& in_gate(std::vector<qubit_type> const& qubits);
    state& hadamard(qubit_type const qubit);
    state& not_(qubit_type const qubit);
    state& pauli_x(qubit_type const qubit);
    state& pauli_xx(qubit_type const qubit1, qubit_type const qubit2);
    state& pauli_xn(std::vector<qubit_type> const& qubits);
    state& pauli_y(qubit_type const qubit);
    state& pauli_yy(qubit_type const qubit1, qubit_type const qubit2);
    state& pauli_yn(std::vector<qubit_type> const& qubits);
    state& pauli_z(control_qubit_type const control_qubit);
    state& pauli_zz(qubit_type const qubit1, qubit_type const qubit2);
    state& pauli_zn(std::vector<qubit_type> const& qubits);
    state& swap(qubit_type const qubit1, qubit_type const qubit2);
    state& sqrt_pauli_x(qubit_type const qubit);
    state& adj_sqrt_pauli_x(qubit_type const qubit);
    state& sqrt_pauli_y(qubit_type const qubit);
    state& adj_sqrt_pauli_y(qubit_type const qubit);
    state& sqrt_pauli_z(control_qubit_type const control_qubit);
    state& adj_sqrt_pauli_z(control_qubit_type const control_qubit);
    state& sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2);
    state& adj_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2);
    state& sqrt_pauli_zn(std::vector<qubit_type> const& qubits);
    state& adj_sqrt_pauli_zn(std::vector<qubit_type> const& qubits);
    state& u1(
      boost::variant<real_type, std::string> const& phase,
      control_qubit_type const control_qubit);
    state& adj_u1(
      boost::variant<real_type, std::string> const& phase,
      control_qubit_type const control_qubit);
    state& u2(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      qubit_type const qubit);
    state& adj_u2(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      qubit_type const qubit);
    state& u3(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      boost::variant<real_type, std::string> const& phase3,
      qubit_type const qubit);
    state& adj_u3(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      boost::variant<real_type, std::string> const& phase3,
      qubit_type const qubit);
    state& phase_shift(
      boost::variant<int_type, std::string> const& phase_exponent,
      control_qubit_type const control_qubit);
    state& adj_phase_shift(
      boost::variant<int_type, std::string> const& phase_exponent,
      control_qubit_type const control_qubit);
    state& x_rotation_half_pi(qubit_type const qubit);
    state& adj_x_rotation_half_pi(qubit_type const qubit);
    state& y_rotation_half_pi(qubit_type const qubit);
    state& adj_y_rotation_half_pi(qubit_type const qubit);
    state& exponential_pauli_x(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit);
    state& adj_exponential_pauli_x(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit);
    state& exponential_pauli_xx(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit1, qubit_type const qubit2);
    state& adj_exponential_pauli_xx(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit1, qubit_type const qubit2);
    state& exponential_pauli_xn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& qubits);
    state& adj_exponential_pauli_xn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& qubits);
    state& exponential_pauli_y(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit);
    state& adj_exponential_pauli_y(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit);
    state& exponential_pauli_yy(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit1, qubit_type const qubit2);
    state& adj_exponential_pauli_yy(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit1, qubit_type const qubit2);
    state& exponential_pauli_yn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& qubits);
    state& adj_exponential_pauli_yn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& qubits);
    state& exponential_pauli_z(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit);
    state& adj_exponential_pauli_z(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit);
    state& exponential_pauli_zz(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit1, qubit_type const qubit2);
    state& adj_exponential_pauli_zz(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit1, qubit_type const qubit2);
    state& exponential_pauli_zn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& qubits);
    state& adj_exponential_pauli_zn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& qubits);
    state& exponential_swap(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit1, qubit_type const qubit2);
    state& adj_exponential_swap(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const qubit1, qubit_type const qubit2);
    state& toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);

# ifndef BRA_NO_MPI
    state& projective_measurement(qubit_type const qubit, yampi::rank const root);
    state& measurement(yampi::rank const root, int const precision);
    state& amplitudes(yampi::rank const root, std::vector< ::bra::state_integer_type > const& amplitude_indices);
    state& generate_events(yampi::rank const root, int const num_events, int const seed);
    state& exit(yampi::rank const root);
# else // BRA_NO_MPI
    state& projective_measurement(qubit_type const qubit);
    state& measurement(int const precision);
    state& amplitudes(std::vector< ::bra::state_integer_type > const& amplitude_indices);
    state& generate_events(int const num_events, int const seed);
    state& exit();
# endif // BRA_NO_MPI
    state& expectation_value(std::string const& pauli_string_space, std::vector<qubit_type> const& operated_qubits);
    state& inner_product(std::string const& remote_circuit_index_or_all);
    state& inner_product(std::string const& remote_circuit_index_or_all, std::string const& pauli_string_space, std::vector<qubit_type> const& operated_qubits);
    state& fidelity(std::string const& remote_circuit_index_or_all);
    state& fidelity(std::string const& remote_circuit_index_or_all, std::string const& pauli_string_space, std::vector<qubit_type> const& operated_qubits);
    state& shor_box(bit_integer_type const num_exponent_qubits, state_integer_type const divisor, state_integer_type const base);

    state& begin_fusion();
    state& end_fusion();

    state& clear(qubit_type const qubit);
    state& set(qubit_type const qubit);

    state& depolarizing_channel(
      real_type const px, real_type const py, real_type const pz,
      int const seed);

    state& controlled_i_gate(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& controlled_ic_gate(control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);
    state& multi_controlled_in_gate(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& multi_controlled_ic_gate(std::vector<control_qubit_type> const& control_qubits);
    state& controlled_hadamard(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_hadamard(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_not(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_not(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_pauli_xn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_pauli_yn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_pauli_z(control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);
    state& multi_controlled_pauli_z(std::vector<control_qubit_type> const& control_qubits);
    state& multi_controlled_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& multi_controlled_swap(qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_sqrt_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_sqrt_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_sqrt_pauli_x(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_sqrt_pauli_x(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_sqrt_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_sqrt_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_sqrt_pauli_y(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_sqrt_pauli_y(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_sqrt_pauli_z(control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);
    state& adj_controlled_sqrt_pauli_z(control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);
    state& multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits);
    state& multi_controlled_sqrt_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_sqrt_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_phase_shift(
      boost::variant<int_type, std::string> const& phase_exponent,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);
    state& adj_controlled_phase_shift(
      boost::variant<int_type, std::string> const& phase_exponent,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);
    state& multi_controlled_phase_shift(
      boost::variant<int_type, std::string> const& phase_exponent,
      std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_phase_shift(
      boost::variant<int_type, std::string> const& phase_exponent,
      std::vector<control_qubit_type> const& control_qubits);
    state& controlled_u1(
      boost::variant<real_type, std::string> const& phase,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);
    state& adj_controlled_u1(
      boost::variant<real_type, std::string> const& phase,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);
    state& multi_controlled_u1(
      boost::variant<real_type, std::string> const& phase,
      std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_u1(
      boost::variant<real_type, std::string> const& phase,
      std::vector<control_qubit_type> const& control_qubits);
    state& controlled_u2(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_u2(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_u2(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_u2(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_u3(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      boost::variant<real_type, std::string> const& phase3,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_u3(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      boost::variant<real_type, std::string> const& phase3,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_u3(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      boost::variant<real_type, std::string> const& phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_u3(
      boost::variant<real_type, std::string> const& phase1,
      boost::variant<real_type, std::string> const& phase2,
      boost::variant<real_type, std::string> const& phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_x_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_x_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_x_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_x_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_y_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_y_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_y_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_y_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_exponential_pauli_x(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_exponential_pauli_x(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_exponential_pauli_xn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_exponential_pauli_xn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_exponential_pauli_y(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_exponential_pauli_y(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_exponential_pauli_yn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_exponential_pauli_yn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_exponential_pauli_z(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_exponential_pauli_z(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_exponential_pauli_z(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_exponential_pauli_z(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& multi_controlled_exponential_pauli_zn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_exponential_pauli_zn(
      boost::variant<real_type, std::string> const& phase,
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& multi_controlled_exponential_swap(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_exponential_swap(
      boost::variant<real_type, std::string> const& phase,
      qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits);

   private:
    virtual auto do_is_waiting() const -> bool { return false; }
    virtual auto do_cancel_waiting() -> void { }

    virtual auto do_send_real_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void = 0;
    virtual auto do_send_complex_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void = 0;
    virtual auto do_send_int_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void = 0;
    virtual auto do_receive_real_variable(int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void = 0;
    virtual auto do_receive_complex_variable(int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void = 0;
    virtual auto do_receive_int_variable(int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void = 0;

# ifndef BRA_NO_MPI
    virtual unsigned int do_num_page_qubits() const = 0;
    virtual unsigned int do_num_pages() const = 0;

# endif
    virtual void do_i_gate(qubit_type const qubit) = 0;
    virtual void do_ic_gate(control_qubit_type const control_qubit) = 0;
    virtual void do_ii_gate(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_in_gate(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_hadamard(qubit_type const qubit) = 0;
    virtual void do_not_(qubit_type const qubit) = 0;
    virtual void do_pauli_x(qubit_type const qubit) = 0;
    virtual void do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_pauli_xn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_pauli_y(qubit_type const qubit) = 0;
    virtual void do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_pauli_yn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_pauli_z(control_qubit_type const control_qubit) = 0;
    virtual void do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_pauli_zn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_swap(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_sqrt_pauli_x(qubit_type const qubit) = 0;
    virtual void do_adj_sqrt_pauli_x(qubit_type const qubit) = 0;
    virtual void do_sqrt_pauli_y(qubit_type const qubit) = 0;
    virtual void do_adj_sqrt_pauli_y(qubit_type const qubit) = 0;
    virtual void do_sqrt_pauli_z(control_qubit_type const control_qubit) = 0;
    virtual void do_adj_sqrt_pauli_z(control_qubit_type const control_qubit) = 0;
    virtual void do_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_adj_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_sqrt_pauli_zn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_adj_sqrt_pauli_zn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_u1(real_type const phase, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_u1(real_type const phase, control_qubit_type const control_qubit) = 0;
    virtual void do_u2(
      real_type const phase1, real_type const phase2,
      qubit_type const qubit) = 0;
    virtual void do_adj_u2(
      real_type const phase1, real_type const phase2,
      qubit_type const qubit) = 0;
    virtual void do_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const qubit) = 0;
    virtual void do_adj_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const qubit) = 0;
    virtual void do_phase_shift(
      complex_type const& phase_coefficient, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_phase_shift(
      complex_type const& phase_coefficient, control_qubit_type const control_qubit) = 0;
    virtual void do_x_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_adj_x_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_y_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_adj_y_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_exponential_pauli_x(real_type const phase, qubit_type const qubit) = 0;
    virtual void do_adj_exponential_pauli_x(real_type const phase, qubit_type const qubit) = 0;
    virtual void do_exponential_pauli_xx(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_adj_exponential_pauli_xx(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& qubits) = 0;
    virtual void do_adj_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& qubits) = 0;
    virtual void do_exponential_pauli_y(real_type const phase, qubit_type const qubit) = 0;
    virtual void do_adj_exponential_pauli_y(real_type const phase, qubit_type const qubit) = 0;
    virtual void do_exponential_pauli_yy(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_adj_exponential_pauli_yy(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& qubits) = 0;
    virtual void do_adj_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& qubits) = 0;
    virtual void do_exponential_pauli_z(real_type const phase, qubit_type const qubit) = 0;
    virtual void do_adj_exponential_pauli_z(real_type const phase, qubit_type const qubit) = 0;
    virtual void do_exponential_pauli_zz(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_adj_exponential_pauli_zz(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& qubits) = 0;
    virtual void do_adj_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& qubits) = 0;
    virtual void do_exponential_swap(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_adj_exponential_swap(
      real_type const phase, qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2)
      = 0;

# ifndef BRA_NO_MPI
    virtual ket::gate::outcome do_projective_measurement(
      qubit_type const qubit, yampi::rank const root) = 0;
    virtual void do_expectation_values(yampi::rank const root) = 0;
    virtual void do_amplitudes(yampi::rank const root, std::vector< ::bra::state_integer_type > const& amplitude_indices) = 0;
    virtual void do_measure(yampi::rank const root) = 0;
    virtual void do_generate_events(yampi::rank const root, int const num_events, int const seed) = 0;
# else // BRA_NO_MPI
    virtual ket::gate::outcome do_projective_measurement(qubit_type const qubit) = 0;
    virtual void do_expectation_values() = 0;
    virtual void do_amplitudes(std::vector< ::bra::state_integer_type > const& amplitude_indices) = 0;
    virtual void do_measure() = 0;
    virtual void do_generate_events(int const num_events, int const seed) = 0;
# endif // BRA_NO_MPI
    virtual void do_expectation_value(std::string const& pauli_string_space, std::vector<qubit_type> const& operated_qubits) = 0;
    virtual void do_inner_product(std::string const& remote_circuit_index_or_all) = 0;
    virtual void do_inner_product(std::string const& remote_circuit_index_or_all, std::string const& pauli_string_space, std::vector<qubit_type> const& operated_qubits) = 0;
    virtual void do_fidelity(std::string const& remote_circuit_index_or_all) = 0;
    virtual void do_fidelity(std::string const& remote_circuit_index_or_all, std::string const& pauli_string_space, std::vector<qubit_type> const& operated_qubits) = 0;
    virtual void do_shor_box(
      state_integer_type const divisor, state_integer_type const base,
      std::vector<qubit_type> const& exponent_qubits,
      std::vector<qubit_type> const& modular_exponentiation_qubits) = 0;
    virtual void do_begin_fusion() = 0;
    virtual void do_end_fusion() = 0;
    virtual void do_clear(qubit_type const qubit) = 0;
    virtual void do_set(qubit_type const qubit) = 0;

    virtual void do_controlled_i_gate(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_controlled_ic_gate(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) = 0;
    virtual void do_multi_controlled_in_gate(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_multi_controlled_ic_gate(
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_hadamard(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_hadamard(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_not(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_pauli_xn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_pauli_yn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_pauli_z(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) = 0;
    virtual void do_multi_controlled_pauli_z(std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_multi_controlled_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_multi_controlled_swap(
      qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_sqrt_pauli_x(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_sqrt_pauli_y(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_sqrt_pauli_z(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) = 0;
    virtual void do_adj_controlled_sqrt_pauli_z(
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) = 0;
    virtual void do_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_multi_controlled_sqrt_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_sqrt_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_phase_shift(
      complex_type const& phase_coefficient,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) = 0;
    virtual void do_adj_controlled_phase_shift(
      complex_type const& phase_coefficient,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) = 0;
    virtual void do_multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_u1(
      real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) = 0;
    virtual void do_adj_controlled_u1(
      real_type const phase, control_qubit_type const control_qubit1, control_qubit_type const control_qubit2) = 0;
    virtual void do_multi_controlled_u1(
      real_type const phase, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_u1(
      real_type const phase, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_exponential_pauli_x(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_exponential_pauli_x(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_exponential_pauli_y(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_exponential_pauli_y(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_multi_controlled_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_multi_controlled_exponential_swap(
      real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_exponential_swap(
      real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) = 0;
  }; // class state
} // namespace bra


# undef BRA_clock

#endif // BRA_STATE_HPP
