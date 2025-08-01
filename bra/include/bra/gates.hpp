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
#   include <yampi/communicator.hpp>
#   include <yampi/environment.hpp>
# endif // BRA_NO_MPI

# include <bra/types.hpp>
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

  enum class begin_statement : int { measurement, fusion, learning_machine };
  enum class end_statement : int { fusion };
  enum class bit_statement : int { assignment };
  enum class generate_statement : int { events };
  enum class depolarizing_statement : int { channel };

  class gates
  {
    using value_type_ = std::unique_ptr< ::bra::gate::gate >;
    using data_type = std::vector<value_type_>;
    data_type data_;

   public:
    using columns_type = wrong_mnemonics_error::columns_type;

   private:
    ::bra::bit_integer_type num_qubits_;
# ifndef BRA_NO_MPI
    ::bra::bit_integer_type num_lqubits_;
    ::bra::bit_integer_type num_uqubits_;
    unsigned int num_processes_per_unit_;
# endif
    ::bra::state_integer_type initial_state_value_;
# ifndef BRA_NO_MPI
    std::vector< ::bra::permutated_qubit_type > initial_permutation_;
# endif

    using phase_coefficients_type = std::vector< ::bra::complex_type >;
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
      ::bra::bit_integer_type num_uqubits, unsigned int num_processes_per_unit,
      yampi::environment const& environment,
      yampi::rank const root = yampi::rank{},
      yampi::communicator const& communicator = yampi::communicator{::yampi::tags::world_communicator},
      size_type const num_reserved_gates = size_type{0u});
# else // BRA_NO_MPI
    explicit gates(std::istream& input_stream);
    gates(std::istream& input_stream, size_type const num_reserved_gates);
# endif // BRA_NO_MPI

    bool operator==(gates const& other) const;

    ::bra::bit_integer_type const& num_qubits() const { return num_qubits_; }
# ifndef BRA_NO_MPI
    ::bra::bit_integer_type const& num_lqubits() const { return num_lqubits_; }
    ::bra::bit_integer_type const& num_uqubits() const { return num_uqubits_; }
    unsigned int const& num_processes_per_unit() const { return num_processes_per_unit_; }
# endif
    ::bra::state_integer_type const& initial_state_value() const { return initial_state_value_; }
# ifndef BRA_NO_MPI
    std::vector< ::bra::permutated_qubit_type > const& initial_permutation() const { return initial_permutation_; }
# endif

# ifndef BRA_NO_MPI
    void num_qubits(
      ::bra::bit_integer_type const new_num_qubits,
      yampi::communicator const& communicator, yampi::environment const& environment);
    void num_lqubits(
      ::bra::bit_integer_type const new_num_lqubits,
      yampi::communicator const& communicator, yampi::environment const& environment);
# else // BRA_NO_MPI
    void num_qubits(::bra::bit_integer_type const new_num_qubits);
# endif // BRA_NO_MPI

   private:
# ifndef BRA_NO_MPI
    void set_num_qubits_params(
      ::bra::bit_integer_type const new_num_lqubits, ::bra::bit_integer_type const num_gqubits,
      yampi::communicator const& communicator, yampi::environment const& environment);
# else // BRA_NO_MPI
    void set_num_qubits_params(::bra::bit_integer_type const new_num_qubits);
# endif // BRA_NO_MPI

   public:
# ifndef BRA_NO_MPI
    void assign(
      std::istream& input_stream, yampi::environment const& environment,
      yampi::communicator const& communicator = yampi::communicator{yampi::tags::world_communicator},
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
        and BRA_is_nothrow_swappable< ::bra::bit_integer_type >::value
        and BRA_is_nothrow_swappable< ::bra::state_integer_type >::value
        and BRA_is_nothrow_swappable< ::bra::qubit_type >::value);

   private:
    ::bra::bit_integer_type read_num_qubits(columns_type const& columns) const;
    ::bra::state_integer_type read_initial_state_value(columns_type& columns) const;
    ::bra::bit_integer_type read_num_mpi_processes(columns_type const& columns) const;
    ::bra::state_integer_type read_mpi_buffer_size(columns_type const& columns) const;
# ifndef BRA_NO_MPI
    std::vector< ::bra::permutated_qubit_type > read_initial_permutation(columns_type const& columns) const;
# endif

    ::bra::qubit_type read_target(columns_type const& columns) const;
    ::bra::control_qubit_type read_control(columns_type const& columns) const;
    std::tuple< ::bra::qubit_type, ::bra::qubit_type > read_2targets(columns_type const& columns) const;
    std::tuple< ::bra::control_qubit_type, ::bra::control_qubit_type > read_2controls(columns_type const& columns) const;
    void read_multi_targets(columns_type const& columns, std::vector< ::bra::qubit_type >& targets) const;
    void read_multi_controls(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const;
    std::tuple< ::bra::qubit_type, ::bra::real_type > read_target_phase(columns_type const& columns) const;
    std::tuple< ::bra::control_qubit_type, ::bra::real_type > read_control_phase(columns_type const& columns) const;
    std::tuple< ::bra::qubit_type, ::bra::real_type, ::bra::real_type > read_target_2phases(columns_type const& columns) const;
    std::tuple< ::bra::qubit_type, ::bra::real_type, ::bra::real_type, ::bra::real_type > read_target_3phases(columns_type const& columns) const;
    std::tuple< ::bra::qubit_type, int > read_target_phaseexp(columns_type const& columns) const;
    std::tuple< ::bra::control_qubit_type, int > read_control_phaseexp(columns_type const& columns) const;
    std::tuple< ::bra::qubit_type, ::bra::qubit_type, ::bra::real_type > read_2targets_phase(columns_type const& columns) const;
    ::bra::real_type read_multi_targets_phase(columns_type const& columns, std::vector< ::bra::qubit_type >& targets) const;
    std::tuple< ::bra::control_qubit_type, ::bra::qubit_type > read_control_target(columns_type const& columns) const;
    std::tuple< ::bra::control_qubit_type, ::bra::qubit_type, int > read_control_target_phaseexp(columns_type const& columns) const;
    std::tuple< ::bra::control_qubit_type, ::bra::control_qubit_type, int > read_2controls_phaseexp(columns_type const& columns) const;
    std::tuple< ::bra::control_qubit_type, ::bra::control_qubit_type, ::bra::qubit_type > read_2controls_target(columns_type const& columns) const;
    ::bra::real_type read_multi_controls_phase(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const;
    std::tuple< ::bra::qubit_type, ::bra::qubit_type > read_multi_controls_2targets(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const;
    ::bra::qubit_type read_multi_controls_target(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const;
    void read_multi_controls_multi_targets(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls, std::vector< ::bra::qubit_type >& targets) const;
    std::tuple< ::bra::control_qubit_type, ::bra::qubit_type, ::bra::real_type > read_control_target_phase(columns_type const& columns) const;
    std::tuple< ::bra::control_qubit_type, ::bra::control_qubit_type, ::bra::real_type > read_2controls_phase(columns_type const& columns) const;
    std::tuple< ::bra::qubit_type, ::bra::real_type > read_multi_controls_target_phase(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const;
    std::tuple< ::bra::control_qubit_type, ::bra::qubit_type, ::bra::real_type, ::bra::real_type > read_control_target_2phases(columns_type const& columns) const;
    std::tuple< ::bra::qubit_type, ::bra::real_type, ::bra::real_type > read_multi_controls_target_2phases(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const;
    std::tuple< ::bra::control_qubit_type, ::bra::qubit_type, ::bra::real_type, ::bra::real_type, ::bra::real_type > read_control_target_3phases(columns_type const& columns) const;
    std::tuple< ::bra::qubit_type, ::bra::real_type, ::bra::real_type, ::bra::real_type > read_multi_controls_target_3phases(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const;
    int read_multi_controls_phaseexp(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const;
    std::tuple< ::bra::qubit_type, int > read_multi_controls_target_phaseexp(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const;
    ::bra::real_type read_multi_controls_multi_targets_phase(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls, std::vector< ::bra::qubit_type >& targets) const;
    std::tuple< ::bra::qubit_type, ::bra::qubit_type, ::bra::real_type > read_multi_controls_2targets_phase(columns_type const& columns, std::vector< ::bra::control_qubit_type >& controls) const;

    ::bra::begin_statement read_begin_statement(columns_type const& columns) const;
    ::bra::end_statement read_end_statement(columns_type const& columns) const;
    ::bra::bit_statement read_bit_statement(columns_type const& columns) const;
    std::tuple< ::bra::bit_integer_type, ::bra::state_integer_type, ::bra::state_integer_type > read_shor_box(columns_type const& columns) const;
    std::tuple< ::bra::generate_statement, int, int > read_generate_statement(columns_type const& columns) const;
    std::tuple< ::bra::depolarizing_statement, ::bra::real_type, ::bra::real_type, ::bra::real_type, int > read_depolarizing_statement(columns_type const& columns) const;

    void add_begin_fusion(columns_type const& columns);

    void add_i(columns_type const& columns);
    void add_ic(columns_type const& columns);
    void add_ii(columns_type const& columns);
    void add_is(columns_type const& columns, std::string const& mnemonic);
    void add_in(columns_type const& columns, std::string const& mnemonic);
    void add_h(columns_type const& columns);
    void add_not(columns_type const& columns);
    void add_x(columns_type const& columns);
    void add_xx(columns_type const& columns);
    void add_xs(columns_type const& columns, std::string const& mnemonic);
    void add_xn(columns_type const& columns, std::string const& mnemonic);
    void add_y(columns_type const& columns);
    void add_yy(columns_type const& columns);
    void add_ys(columns_type const& columns, std::string const& mnemonic);
    void add_yn(columns_type const& columns, std::string const& mnemonic);
    void add_z(columns_type const& columns);
    void add_zz(columns_type const& columns);
    void add_zs(columns_type const& columns, std::string const& mnemonic);
    void add_zn(columns_type const& columns, std::string const& mnemonic);
    void add_swap(columns_type const& columns);
    void add_sx(columns_type const& columns);
    void add_adj_sx(columns_type const& columns);
    void add_sy(columns_type const& columns);
    void add_adj_sy(columns_type const& columns);
    void add_sz(columns_type const& columns);
    void add_adj_sz(columns_type const& columns);
    void add_szz(columns_type const& columns);
    void add_adj_szz(columns_type const& columns);
    void add_szs(columns_type const& columns, std::string const& mnemonic);
    void add_adj_szs(columns_type const& columns, std::string const& mnemonic);
    void add_szn(columns_type const& columns, std::string const& mnemonic);
    void add_adj_szn(columns_type const& columns, std::string const& mnemonic);
    void add_s(columns_type const& columns);
    void add_adj_s(columns_type const& columns);
    void add_t(columns_type const& columns);
    void add_adj_t(columns_type const& columns);
    void add_u1(columns_type const& columns);
    void add_adj_u1(columns_type const& columns);
    void add_u2(columns_type const& columns);
    void add_adj_u2(columns_type const& columns);
    void add_u3(columns_type const& columns);
    void add_adj_u3(columns_type const& columns);
    void add_r(columns_type const& columns);
    void add_adj_r(columns_type const& columns);
    void add_rotx(columns_type const& columns);
    void add_adj_rotx(columns_type const& columns);
    void add_roty(columns_type const& columns);
    void add_adj_roty(columns_type const& columns);
    void add_u(columns_type const& columns);
    void add_adj_u(columns_type const& columns);
    void add_ex(columns_type const& columns);
    void add_adj_ex(columns_type const& columns);
    void add_exx(columns_type const& columns);
    void add_adj_exx(columns_type const& columns);
    void add_exs(columns_type const& columns, std::string const& mnemonic);
    void add_adj_exs(columns_type const& columns, std::string const& mnemonic);
    void add_exn(columns_type const& columns, std::string const& mnemonic);
    void add_adj_exn(columns_type const& columns, std::string const& mnemonic);
    void add_ey(columns_type const& columns);
    void add_adj_ey(columns_type const& columns);
    void add_eyy(columns_type const& columns);
    void add_adj_eyy(columns_type const& columns);
    void add_eys(columns_type const& columns, std::string const& mnemonic);
    void add_adj_eys(columns_type const& columns, std::string const& mnemonic);
    void add_eyn(columns_type const& columns, std::string const& mnemonic);
    void add_adj_eyn(columns_type const& columns, std::string const& mnemonic);
    void add_ez(columns_type const& columns);
    void add_adj_ez(columns_type const& columns);
    void add_ezz(columns_type const& columns);
    void add_adj_ezz(columns_type const& columns);
    void add_ezs(columns_type const& columns, std::string const& mnemonic);
    void add_adj_ezs(columns_type const& columns, std::string const& mnemonic);
    void add_ezn(columns_type const& columns, std::string const& mnemonic);
    void add_adj_ezn(columns_type const& columns, std::string const& mnemonic);
    void add_eswap(columns_type const& columns);
    void add_adj_eswap(columns_type const& columns);
    void add_toffoli(columns_type const& columns);
    void add_m(columns_type const& columns);
    void add_shor_box(columns_type const& columns);
    void add_clear(columns_type const& columns);
    void add_set(columns_type const& columns);
    void add_depolarizing(columns_type const& columns, std::string const& mnemonic);

    void interpret_controlled_gates(columns_type const& columns, std::string const& mnemonic);
    void add_ci(columns_type const& columns, int const num_control_qubits);
    void add_cic(columns_type const& columns, int const num_control_qubits);
    void add_cis(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_cin(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
    void add_ch(columns_type const& columns, int const num_control_qubits);
    void add_cnot(columns_type const& columns, int const num_control_qubits);
    void add_cx(columns_type const& columns, int const num_control_qubits);
    void add_cxs(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_cxn(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
    void add_cy(columns_type const& columns, int const num_control_qubits);
    void add_cys(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_cyn(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
    void add_cz(columns_type const& columns, int const num_control_qubits);
    void add_czs(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_czn(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
    void add_cswap(columns_type const& columns, int const num_control_qubits);
    void add_cs(columns_type const& columns, int const num_control_qubits);
    void add_adj_cs(columns_type const& columns, int const num_control_qubits);
    void add_ct(columns_type const& columns, int const num_control_qubits);
    void add_adj_ct(columns_type const& columns, int const num_control_qubits);
    void add_cu1(columns_type const& columns, int const num_control_qubits);
    void add_adj_cu1(columns_type const& columns, int const num_control_qubits);
    void add_cu2(columns_type const& columns, int const num_control_qubits);
    void add_adj_cu2(columns_type const& columns, int const num_control_qubits);
    void add_cu3(columns_type const& columns, int const num_control_qubits);
    void add_adj_cu3(columns_type const& columns, int const num_control_qubits);
    void add_cr(columns_type const& columns, int const num_control_qubits);
    void add_adj_cr(columns_type const& columns, int const num_control_qubits);
    void add_crotx(columns_type const& columns, int const num_control_qubits);
    void add_adj_crotx(columns_type const& columns, int const num_control_qubits);
    void add_croty(columns_type const& columns, int const num_control_qubits);
    void add_adj_croty(columns_type const& columns, int const num_control_qubits);
    void add_cex(columns_type const& columns, int const num_control_qubits);
    void add_adj_cex(columns_type const& columns, int const num_control_qubits);
    void add_cexs(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_adj_cexs(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_cexn(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
    void add_adj_cexn(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
    void add_cey(columns_type const& columns, int const num_control_qubits);
    void add_adj_cey(columns_type const& columns, int const num_control_qubits);
    void add_ceys(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_adj_ceys(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_ceyn(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
    void add_adj_ceyn(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
    void add_cez(columns_type const& columns, int const num_control_qubits);
    void add_adj_cez(columns_type const& columns, int const num_control_qubits);
    void add_cezs(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_adj_cezs(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_cezn(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
    void add_adj_cezn(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
    void add_ceswap(columns_type const& columns, int const num_control_qubits);
    void add_adj_ceswap(columns_type const& columns, int const num_control_qubits);
    void add_csx(columns_type const& columns, int const num_control_qubits);
    void add_adj_csx(columns_type const& columns, int const num_control_qubits);
    void add_csy(columns_type const& columns, int const num_control_qubits);
    void add_adj_csy(columns_type const& columns, int const num_control_qubits);
    void add_csz(columns_type const& columns, int const num_control_qubits);
    void add_adj_csz(columns_type const& columns, int const num_control_qubits);
    void add_cszs(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_adj_cszs(columns_type const& columns, int const num_control_qubits, std::string const& noncontrol_mnemonic);
    void add_cszn(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
    void add_adj_cszn(
      columns_type const& columns, int const num_control_qubits,
      std::string const& noncontrol_mnemonic, std::string const& mnemonic);
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
