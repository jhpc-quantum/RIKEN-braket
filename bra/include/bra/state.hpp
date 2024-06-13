#ifndef BRA_STATE_HPP
# define BRA_STATE_HPP

# include <cstddef>
# include <complex>
# include <vector>
# include <array>
# include <utility>
# ifdef BRA_NO_MPI
#   include <chrono>
#   include <memory>
# endif // BRA_NO_MPI
# include <random>
# include <stdexcept>

# include <boost/optional.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/projective_measurement.hpp>
# ifndef BRA_NO_MPI
#   include <ket/mpi/permutated.hpp>
#   include <ket/mpi/qubit_permutation.hpp>

#   include <yampi/allocator.hpp>
#   include <yampi/datatype.hpp>
#   include <yampi/rank.hpp>
#   include <yampi/communicator.hpp>
#   include <yampi/environment.hpp>
#   include <yampi/wall_clock.hpp>
# endif // BRA_NO_MPI

# ifndef BRA_NO_MPI
#   define BRA_clock yampi::wall_clock
# else // BRA_NO_MPI
#   define BRA_clock std::chrono::system_clock
# endif // BRA_NO_MPI


namespace bra
{
  enum class finished_process : int { operations, begin_measurement, generate_events, ket_measure };

  class too_many_qubits_error
    : public std::runtime_error
  {
   public:
    too_many_qubits_error(std::size_t const num_qubits);
  }; // class too_many_qubits_error

  class state
  {
   public:
    using state_integer_type = std::uint64_t;
    using bit_integer_type = unsigned int;
    using qubit_type = ket::qubit<state_integer_type, bit_integer_type>;
    using control_qubit_type = ket::control<qubit_type>;
# ifndef BRA_NO_MPI
    using permutated_qubit_type = ket::mpi::permutated<qubit_type>;
    using permutated_control_qubit_type = ket::mpi::permutated<control_qubit_type>;
# endif // BRA_NO_MPI

# ifdef BRA_REAL_TYPE
#   if BRA_REAL_TYPE == 0
    using real_type = long double;
#   elif BRA_REAL_TYPE == 1
    using real_type = double;
#   elif BRA_REAL_TYPE == 2
    using real_type = float;
#   else
    using real_type = double;
#   endif
# else // BRA_REAL_TYPE
    using real_type = double;
# endif // BRA_REAL_TYPE
    using complex_type = std::complex<real_type>;

    using spin_type = std::array<real_type, 3u>;
# ifndef BRA_NO_MPI
    using spins_allocator_type = yampi::allocator<spin_type>;
# else // BRA_NO_MPI
    using spins_allocator_type = std::allocator<spin_type>;
# endif // BRA_NO_MPI
    using spins_type = std::vector<spin_type, spins_allocator_type>;
    using random_number_generator_type = std::mt19937_64;
    using seed_type = random_number_generator_type::result_type;

# ifndef BRA_NO_MPI
    using permutation_type
      = ket::mpi::qubit_permutation<
          state_integer_type, bit_integer_type, yampi::allocator<permutated_qubit_type>>;
# endif // BRA_NO_MPI

    using time_and_process_type
      = std::pair<BRA_clock::time_point, ::bra::finished_process>;

   protected:
    bit_integer_type total_num_qubits_;
    std::vector<ket::gate::outcome> last_outcomes_; // return values of ket(::mpi)::gate::projective_measurement
    boost::optional<spins_type> maybe_expectation_values_; // return value of ket(::mpi)::all_spin_expectation_values
    state_integer_type measured_value_; // return value of ket(::mpi)::measure
    std::vector<state_integer_type> generated_events_; // results of ket(::mpi)::generate_events
    random_number_generator_type random_number_generator_;
# ifndef BRA_NO_MPI

    permutation_type permutation_;
    std::vector<complex_type, yampi::allocator<complex_type>> buffer_;
    yampi::datatype real_pair_datatype_;
    yampi::communicator const& communicator_;
    yampi::environment const& environment_;
# endif // BRA_NO_MPI

    std::vector<time_and_process_type> finish_times_and_processes_;

   public:
# ifndef BRA_NO_MPI
    state(
      bit_integer_type const total_num_qubits,
      seed_type const seed,
      yampi::communicator const& communicator,
      yampi::environment const& environment);

    state(
      bit_integer_type const total_num_qubits,
      seed_type const seed,
      unsigned int const num_elements_in_buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment);

    state(
      std::vector<permutated_qubit_type> const& initial_permutation,
      seed_type const seed,
      yampi::communicator const& communicator,
      yampi::environment const& environment);

    state(
      std::vector<permutated_qubit_type> const& initial_permutation,
      seed_type const seed,
      unsigned int const num_elements_in_buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment);
# else // BRA_NO_MPI
    state(bit_integer_type const total_num_qubits, seed_type const seed);
# endif // BRA_NO_MPI

    virtual ~state() = default;
    state(state const&) = delete;
    state& operator=(state const&) = delete;
    state(state&&) = delete;
    state& operator=(state&&) = delete;

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

    yampi::communicator const& communicator() const { return communicator_; }
    yampi::environment const& environment() const { return environment_; }
# endif // BRA_NO_MPI

    std::size_t num_finish_processes() const { return finish_times_and_processes_.size(); }
    time_and_process_type const& finish_time_and_process(std::size_t const n) const
    { return finish_times_and_processes_[n]; }


# ifndef BRA_NO_MPI
    unsigned int num_page_qubits() const { return do_num_page_qubits(); }
    unsigned int num_pages() const { return do_num_pages(); }
# endif // BRA_NO_MPI

    ::bra::state& hadamard(qubit_type const qubit)
    { do_hadamard(qubit); return *this; }

    ::bra::state& adj_hadamard(qubit_type const qubit)
    { do_adj_hadamard(qubit); return *this; }

    ::bra::state& not_(qubit_type const qubit)
    { do_not_(qubit); return *this; }

    ::bra::state& adj_not_(qubit_type const qubit)
    { do_adj_not_(qubit); return *this; }

    ::bra::state& pauli_x(qubit_type const qubit)
    { do_pauli_x(qubit); return *this; }

    ::bra::state& adj_pauli_x(qubit_type const qubit)
    { do_adj_pauli_x(qubit); return *this; }

    ::bra::state& pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
    { do_pauli_xx(qubit1, qubit2); return *this; }

    ::bra::state& adj_pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
    { do_adj_pauli_xx(qubit1, qubit2); return *this; }

    ::bra::state& pauli_xn(std::vector<qubit_type> const& qubits)
    { do_pauli_xn(qubits); return *this; }

    ::bra::state& adj_pauli_xn(std::vector<qubit_type> const& qubits)
    { do_adj_pauli_xn(qubits); return *this; }

    ::bra::state& pauli_y(qubit_type const qubit)
    { do_pauli_y(qubit); return *this; }

    ::bra::state& adj_pauli_y(qubit_type const qubit)
    { do_adj_pauli_y(qubit); return *this; }

    ::bra::state& pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
    { do_pauli_yy(qubit1, qubit2); return *this; }

    ::bra::state& adj_pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
    { do_adj_pauli_yy(qubit1, qubit2); return *this; }

    ::bra::state& pauli_yn(std::vector<qubit_type> const& qubits)
    { do_pauli_yn(qubits); return *this; }

    ::bra::state& adj_pauli_yn(std::vector<qubit_type> const& qubits)
    { do_adj_pauli_yn(qubits); return *this; }

    ::bra::state& pauli_z(qubit_type const qubit)
    { do_pauli_z(qubit); return *this; }

    ::bra::state& adj_pauli_z(qubit_type const qubit)
    { do_adj_pauli_z(qubit); return *this; }

    ::bra::state& pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
    { do_pauli_zz(qubit1, qubit2); return *this; }

    ::bra::state& adj_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
    { do_adj_pauli_zz(qubit1, qubit2); return *this; }

    ::bra::state& pauli_zn(std::vector<qubit_type> const& qubits)
    { do_pauli_zn(qubits); return *this; }

    ::bra::state& adj_pauli_zn(std::vector<qubit_type> const& qubits)
    { do_adj_pauli_zn(qubits); return *this; }

    ::bra::state& swap(qubit_type const qubit1, qubit_type const qubit2)
    { do_swap(qubit1, qubit2); return *this; }

    ::bra::state& adj_swap(qubit_type const qubit1, qubit_type const qubit2)
    { do_adj_swap(qubit1, qubit2); return *this; }

    ::bra::state& u1(real_type const phase, qubit_type const qubit)
    { do_u1(phase, qubit); return *this; }

    ::bra::state& adj_u1(real_type const phase, qubit_type const qubit)
    { do_adj_u1(phase, qubit); return *this; }

    ::bra::state& u2(
      real_type const phase1, real_type const phase2, qubit_type const qubit)
    { do_u2(phase1, phase2, qubit); return *this; }

    ::bra::state& adj_u2(
      real_type const phase1, real_type const phase2, qubit_type const qubit)
    { do_adj_u2(phase1, phase2, qubit); return *this; }

    ::bra::state& u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const qubit)
    { do_u3(phase1, phase2, phase3, qubit); return *this; }

    ::bra::state& adj_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const qubit)
    { do_adj_u3(phase1, phase2, phase3, qubit); return *this; }

    ::bra::state& phase_shift(
      complex_type const& phase_coefficient, qubit_type const qubit)
    { do_phase_shift(phase_coefficient, qubit); return *this; }

    ::bra::state& adj_phase_shift(
      complex_type const& phase_coefficient, qubit_type const qubit)
    { do_adj_phase_shift(phase_coefficient, qubit); return *this; }

    ::bra::state& x_rotation_half_pi(qubit_type const qubit)
    { do_x_rotation_half_pi(qubit); return *this; }

    ::bra::state& adj_x_rotation_half_pi(qubit_type const qubit)
    { do_adj_x_rotation_half_pi(qubit); return *this; }

    ::bra::state& y_rotation_half_pi(qubit_type const qubit)
    { do_y_rotation_half_pi(qubit); return *this; }

    ::bra::state& adj_y_rotation_half_pi(qubit_type const qubit)
    { do_adj_y_rotation_half_pi(qubit); return *this; }

    ::bra::state& controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    {
      do_controlled_v(phase_coefficient, target_qubit, control_qubit);
      return *this;
    }

    ::bra::state& adj_controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    {
      do_adj_controlled_v(phase_coefficient, target_qubit, control_qubit);
      return *this;
    }

    ::bra::state& exponential_pauli_x(real_type const phase, qubit_type const qubit)
    { do_exponential_pauli_x(phase, qubit); return *this; }

    ::bra::state& adj_exponential_pauli_x(real_type const phase, qubit_type const qubit)
    { do_adj_exponential_pauli_x(phase, qubit); return *this; }

    ::bra::state& exponential_pauli_xx(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
    { do_exponential_pauli_xx(phase, qubit1, qubit2); return *this; }

    ::bra::state& adj_exponential_pauli_xx(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
    { do_adj_exponential_pauli_xx(phase, qubit1, qubit2); return *this; }

    ::bra::state& exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& qubits)
    { do_exponential_pauli_xn(phase, qubits); return *this; }

    ::bra::state& adj_exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& qubits)
    { do_adj_exponential_pauli_xn(phase, qubits); return *this; }

    ::bra::state& exponential_pauli_y(real_type const phase, qubit_type const qubit)
    { do_exponential_pauli_y(phase, qubit); return *this; }

    ::bra::state& adj_exponential_pauli_y(real_type const phase, qubit_type const qubit)
    { do_adj_exponential_pauli_y(phase, qubit); return *this; }

    ::bra::state& exponential_pauli_yy(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
    { do_exponential_pauli_yy(phase, qubit1, qubit2); return *this; }

    ::bra::state& adj_exponential_pauli_yy(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
    { do_adj_exponential_pauli_yy(phase, qubit1, qubit2); return *this; }

    ::bra::state& exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& qubits)
    { do_exponential_pauli_yn(phase, qubits); return *this; }

    ::bra::state& adj_exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& qubits)
    { do_adj_exponential_pauli_yn(phase, qubits); return *this; }

    ::bra::state& exponential_pauli_z(real_type const phase, qubit_type const qubit)
    { do_exponential_pauli_z(phase, qubit); return *this; }

    ::bra::state& adj_exponential_pauli_z(real_type const phase, qubit_type const qubit)
    { do_adj_exponential_pauli_z(phase, qubit); return *this; }

    ::bra::state& exponential_pauli_zz(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
    { do_exponential_pauli_zz(phase, qubit1, qubit2); return *this; }

    ::bra::state& adj_exponential_pauli_zz(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
    { do_adj_exponential_pauli_zz(phase, qubit1, qubit2); return *this; }

    ::bra::state& exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& qubits)
    { do_exponential_pauli_zn(phase, qubits); return *this; }

    ::bra::state& adj_exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& qubits)
    { do_adj_exponential_pauli_zn(phase, qubits); return *this; }

    ::bra::state& exponential_swap(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
    { do_exponential_swap(phase, qubit1, qubit2); return *this; }

    ::bra::state& adj_exponential_swap(real_type const phase, qubit_type const qubit1, qubit_type const qubit2)
    { do_adj_exponential_swap(phase, qubit1, qubit2); return *this; }

    ::bra::state& toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2)
    {
      do_toffoli(target_qubit, control_qubit1, control_qubit2);
      return *this;
    }

    ::bra::state& adj_toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2)
    {
      do_adj_toffoli(target_qubit, control_qubit1, control_qubit2);
      return *this;
    }

# ifndef BRA_NO_MPI
    ::bra::state& projective_measurement(qubit_type const qubit, yampi::rank const root);

    ::bra::state& measurement(yampi::rank const root);

    ::bra::state& generate_events(yampi::rank const root, int const num_events, int const seed);

    ::bra::state& exit(yampi::rank const root);
# else // BRA_NO_MPI
    ::bra::state& projective_measurement(qubit_type const qubit);

    ::bra::state& measurement();

    ::bra::state& generate_events(int const num_events, int const seed);

    ::bra::state& exit();
# endif // BRA_NO_MPI

    ::bra::state& shor_box(bit_integer_type const num_exponent_qubits, state_integer_type const divisor, state_integer_type const base);

    ::bra::state& clear(qubit_type const qubit)
    { do_clear(qubit); return *this; }

    ::bra::state& set(qubit_type const qubit)
    { do_set(qubit); return *this; }

    ::bra::state& depolarizing_channel(real_type const px, real_type const py, real_type const pz, int const seed);

    ::bra::state& controlled_hadamard(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_hadamard(target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_hadamard(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_hadamard(target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_hadamard(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_hadamard(target_qubit, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_hadamard(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_hadamard(target_qubit, control_qubits); return *this; }

    ::bra::state& controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_not(target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_adj_controlled_not(target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_not(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_not(target_qubit, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_not(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_not(target_qubit, control_qubits); return *this; }

    ::bra::state& controlled_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_pauli_x(target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_pauli_x(target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_pauli_xn(
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_pauli_xn(target_qubits, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_pauli_xn(
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_pauli_xn(target_qubits, control_qubits); return *this; }

    ::bra::state& controlled_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_pauli_y(target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_pauli_y(target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_pauli_yn(
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_pauli_yn(target_qubits, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_pauli_yn(
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_pauli_yn(target_qubits, control_qubits); return *this; }

    ::bra::state& controlled_pauli_z(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_pauli_z(target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_pauli_z(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_pauli_z(target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_pauli_zn(
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_pauli_zn(target_qubits, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_pauli_zn(
      std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_pauli_zn(target_qubits, control_qubits); return *this; }

    ::bra::state& multi_controlled_swap(
      qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_swap(target_qubit1, target_qubit2, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_swap(
      qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_swap(target_qubit1, target_qubit2, control_qubits); return *this; }

    ::bra::state& controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_phase_shift(phase_coefficient, target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_adj_controlled_phase_shift(phase_coefficient, target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_phase_shift(phase_coefficient, target_qubit, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_adj_multi_controlled_phase_shift(phase_coefficient, target_qubit, control_qubits); return *this; }

    ::bra::state& controlled_u1(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_u1(phase, target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_u1(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_adj_controlled_u1(phase, target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_u1(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_u1(phase, target_qubit, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_u1(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_adj_multi_controlled_u1(phase, target_qubit, control_qubits); return *this; }

    ::bra::state& controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_u2(phase1, phase2, target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_adj_controlled_u2(phase1, phase2, target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_u2(phase1, phase2, target_qubit, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_u2(
      real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_adj_multi_controlled_u2(phase1, phase2, target_qubit, control_qubits); return *this; }

    ::bra::state& controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_u3(phase1, phase2, phase3, target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_adj_controlled_u3(phase1, phase2, phase3, target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_u3(phase1, phase2, phase3, target_qubit, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_adj_multi_controlled_u3(phase1, phase2, phase3, target_qubit, control_qubits); return *this; }

    ::bra::state& controlled_x_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_x_rotation_half_pi(target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_x_rotation_half_pi(target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_x_rotation_half_pi(target_qubit, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_x_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_x_rotation_half_pi(target_qubit, control_qubits); return *this; }

    ::bra::state& controlled_y_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_y_rotation_half_pi(target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_y_rotation_half_pi(target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_y_rotation_half_pi(target_qubit, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_y_rotation_half_pi(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_y_rotation_half_pi(target_qubit, control_qubits); return *this; }

    ::bra::state& multi_controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_v(phase_coefficient, target_qubit, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
    { do_adj_multi_controlled_v(phase_coefficient, target_qubit, control_qubits); return *this; }

    ::bra::state& controlled_exponential_pauli_x(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_exponential_pauli_x(phase, target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_exponential_pauli_x(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_adj_controlled_exponential_pauli_x(phase, target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_exponential_pauli_xn(phase, target_qubits, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_exponential_pauli_xn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_adj_multi_controlled_exponential_pauli_xn(phase, target_qubits, control_qubits); return *this; }

    ::bra::state& controlled_exponential_pauli_y(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_exponential_pauli_x(phase, target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_exponential_pauli_y(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_adj_controlled_exponential_pauli_x(phase, target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_exponential_pauli_xn(phase, target_qubits, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_exponential_pauli_yn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_adj_multi_controlled_exponential_pauli_xn(phase, target_qubits, control_qubits); return *this; }

    ::bra::state& controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_exponential_pauli_x(phase, target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_exponential_pauli_z(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_adj_controlled_exponential_pauli_x(phase, target_qubit, control_qubit); return *this; }

    ::bra::state& multi_controlled_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_exponential_pauli_xn(phase, target_qubits, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_exponential_pauli_zn(
      real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
    { do_adj_multi_controlled_exponential_pauli_xn(phase, target_qubits, control_qubits); return *this; }

    ::bra::state& multi_controlled_exponential_swap(
      real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits)
    { do_multi_controlled_exponential_swap(phase, target_qubit1, target_qubit2, control_qubits); return *this; }

    ::bra::state& adj_multi_controlled_exponential_swap(
      real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits)
    { do_adj_multi_controlled_exponential_swap(phase, target_qubit1, target_qubit2, control_qubits); return *this; }

   private:
# ifndef BRA_NO_MPI
    virtual unsigned int do_num_page_qubits() const = 0;
    virtual unsigned int do_num_pages() const = 0;

# endif
    virtual void do_hadamard(qubit_type const qubit) = 0;
    virtual void do_adj_hadamard(qubit_type const qubit) = 0;
    virtual void do_not_(qubit_type const qubit) = 0;
    virtual void do_adj_not_(qubit_type const qubit) = 0;
    virtual void do_pauli_x(qubit_type const qubit) = 0;
    virtual void do_adj_pauli_x(qubit_type const qubit) = 0;
    virtual void do_pauli_xx(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_adj_pauli_xx(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_pauli_xn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_adj_pauli_xn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_pauli_y(qubit_type const qubit) = 0;
    virtual void do_adj_pauli_y(qubit_type const qubit) = 0;
    virtual void do_pauli_yy(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_adj_pauli_yy(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_pauli_yn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_adj_pauli_yn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_pauli_z(qubit_type const qubit) = 0;
    virtual void do_adj_pauli_z(qubit_type const qubit) = 0;
    virtual void do_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_adj_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_pauli_zn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_adj_pauli_zn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_swap(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_adj_swap(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_u1(real_type const phase, qubit_type const qubit) = 0;
    virtual void do_adj_u1(real_type const phase, qubit_type const qubit) = 0;
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
      complex_type const& phase_coefficient, qubit_type const qubit) = 0;
    virtual void do_adj_phase_shift(
      complex_type const& phase_coefficient, qubit_type const qubit) = 0;
    virtual void do_x_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_adj_x_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_y_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_adj_y_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
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
    virtual void do_adj_toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1,
      control_qubit_type const control_qubit2)
      = 0;
# ifndef BRA_NO_MPI
    virtual ket::gate::outcome do_projective_measurement(
      qubit_type const qubit, yampi::rank const root) = 0;
    virtual void do_expectation_values(yampi::rank const root) = 0;
    virtual void do_measure(yampi::rank const root) = 0;
    virtual void do_generate_events(yampi::rank const root, int const num_events, int const seed) = 0;
# else // BRA_NO_MPI
    virtual ket::gate::outcome do_projective_measurement(qubit_type const qubit) = 0;
    virtual void do_expectation_values() = 0;
    virtual void do_measure() = 0;
    virtual void do_generate_events(int const num_events, int const seed) = 0;
# endif // BRA_NO_MPI
    virtual void do_shor_box(
      state_integer_type const divisor, state_integer_type const base,
      std::vector<qubit_type> const& exponent_qubits,
      std::vector<qubit_type> const& modular_exponentiation_qubits) = 0;
    virtual void do_clear(qubit_type const qubit) = 0;
    virtual void do_set(qubit_type const qubit) = 0;
    virtual void do_controlled_hadamard(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_hadamard(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_hadamard(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_hadamard(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_not(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_not(
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_pauli_x(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_pauli_xn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_pauli_xn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_pauli_y(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_pauli_yn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_pauli_yn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_pauli_z(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_pauli_z(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_multi_controlled_swap(
      qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_swap(
      qubit_type const target_qubit1, qubit_type const target_qubit2,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_phase_shift(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_controlled_u1(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_u1(
      real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_u1(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_u1(
      real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
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
    virtual void do_multi_controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_v(
      complex_type const& phase_coefficient,
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
