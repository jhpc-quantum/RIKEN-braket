#ifndef BRA_STATE_HPP
# define BRA_STATE_HPP

# include <cstddef>
# include <complex>
# include <vector>
# include <unordered_map>
# include <array>
# include <utility>
# ifdef BRA_NO_MPI
#   include <chrono>
# endif // BRA_NO_MPI
# include <memory>
# include <random>
# include <stdexcept>

# include <boost/optional.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/projective_measurement.hpp>
# ifndef BRA_NO_MPI
#   include <ket/mpi/permutated.hpp>
#   include <ket/mpi/qubit_permutation.hpp>

#   include <yampi/rank.hpp>
#   include <yampi/communicator.hpp>
#   include <yampi/environment.hpp>
#   include <yampi/wall_clock.hpp>
# endif // BRA_NO_MPI

# include <bra/types.hpp>

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
  enum class finished_process : int { operations, begin_measurement, generate_events, ket_measure };

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
} // namespace bra

template <typename StateInteger, typename BitInteger>
struct std::hash<ket::qubit<StateInteger, BitInteger>>
{
  std::size_t operator()(ket::qubit<StateInteger, BitInteger> const& qubit) const noexcept
  { return static_cast<std::size_t>(static_cast<BitInteger>(qubit)); }
};

namespace bra
{
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

    using spin_type = std::array<real_type, 3u>;
    using spins_type = std::vector<spin_type>;
    using random_number_generator_type = std::mt19937_64;
    using seed_type = random_number_generator_type::result_type;

# ifndef BRA_NO_MPI
    using permutation_type
      = ket::mpi::qubit_permutation<state_integer_type, bit_integer_type>;
# endif // BRA_NO_MPI

    using time_and_process_type
      = std::pair<BRA_clock::time_point, ::bra::finished_process>;

   protected:
    bit_integer_type total_num_qubits_;
    std::vector<ket::gate::outcome> last_outcomes_; // return values of ket(::mpi)::gate::projective_measurement
    boost::optional<spins_type> maybe_expectation_values_; // return value of ket(::mpi)::all_spin_expectation_values
    state_integer_type measured_value_; // return value of ket(::mpi)::measure
    std::vector<state_integer_type> generated_events_; // results of ket(::mpi)::generate_events
    bool is_in_fusion_; // related to begin_fusion/end_fusion
    std::vector< ::bra::qubit_type > fused_qubits_; // related to begin_fusion/end_fusion
    std::unordered_map< ::bra::qubit_type,  ::bra::qubit_type > to_qubit_in_fused_gate_; // related to begin_fusion/end_fusion
    random_number_generator_type random_number_generator_;
# ifndef BRA_NO_MPI

    permutation_type permutation_;
    ::bra::data_type buffer_;
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

    state& i_gate(qubit_type const qubit);
    state& adj_i_gate(qubit_type const qubit);
    state& ii_gate(qubit_type const qubit1, qubit_type const qubit2);
    state& adj_ii_gate(qubit_type const qubit1, qubit_type const qubit2);
    state& in_gate(std::vector<qubit_type> const& qubits);
    state& adj_in_gate(std::vector<qubit_type> const& qubits);
    state& hadamard(qubit_type const qubit);
    state& adj_hadamard(qubit_type const qubit);
    state& not_(qubit_type const qubit);
    state& adj_not_(qubit_type const qubit);
    state& pauli_x(qubit_type const qubit);
    state& adj_pauli_x(qubit_type const qubit);
    state& pauli_xx(qubit_type const qubit1, qubit_type const qubit2);
    state& adj_pauli_xx(qubit_type const qubit1, qubit_type const qubit2);
    state& pauli_xn(std::vector<qubit_type> const& qubits);
    state& adj_pauli_xn(std::vector<qubit_type> const& qubits);
    state& pauli_y(qubit_type const qubit);
    state& adj_pauli_y(qubit_type const qubit);
    state& pauli_yy(qubit_type const qubit1, qubit_type const qubit2);
    state& adj_pauli_yy(qubit_type const qubit1, qubit_type const qubit2);
    state& pauli_yn(std::vector<qubit_type> const& qubits);
    state& adj_pauli_yn(std::vector<qubit_type> const& qubits);
    state& pauli_z(qubit_type const qubit);
    state& adj_pauli_z(qubit_type const qubit);
    state& pauli_zz(qubit_type const qubit1, qubit_type const qubit2);
    state& adj_pauli_zz(qubit_type const qubit1, qubit_type const qubit2);
    state& pauli_zn(std::vector<qubit_type> const& qubits);
    state& adj_pauli_zn(std::vector<qubit_type> const& qubits);
    state& swap(qubit_type const qubit1, qubit_type const qubit2);
    state& adj_swap(qubit_type const qubit1, qubit_type const qubit2);
    state& sqrt_pauli_x(qubit_type const qubit);
    state& adj_sqrt_pauli_x(qubit_type const qubit);
    state& sqrt_pauli_y(qubit_type const qubit);
    state& adj_sqrt_pauli_y(qubit_type const qubit);
    state& sqrt_pauli_z(qubit_type const qubit);
    state& adj_sqrt_pauli_z(qubit_type const qubit);
    state& sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2);
    state& adj_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2);
    state& sqrt_pauli_zn(std::vector<qubit_type> const& qubits);
    state& adj_sqrt_pauli_zn(std::vector<qubit_type> const& qubits);
    state& u1(real_type const phase, qubit_type const qubit);
    state& adj_u1(real_type const phase, qubit_type const qubit);
    state& u2(
      real_type const phase1, real_type const phase2, qubit_type const qubit);
    state& adj_u2(
      real_type const phase1, real_type const phase2, qubit_type const qubit);
    state& u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const qubit);
    state& adj_u3(
      real_type const phase1, real_type const phase2, real_type const phase3,
      qubit_type const qubit);
    state& phase_shift(
      complex_type const& phase_coefficient, qubit_type const qubit);
    state& adj_phase_shift(
      complex_type const& phase_coefficient, qubit_type const qubit);
    state& x_rotation_half_pi(qubit_type const qubit);
    state& adj_x_rotation_half_pi(qubit_type const qubit);
    state& y_rotation_half_pi(qubit_type const qubit);
    state& adj_y_rotation_half_pi(qubit_type const qubit);
    state& controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_v(
      complex_type const& phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& exponential_pauli_x(real_type const phase, qubit_type const qubit);
    state& adj_exponential_pauli_x(real_type const phase, qubit_type const qubit);
    state& exponential_pauli_xx(real_type const phase, qubit_type const qubit1, qubit_type const qubit2);
    state& adj_exponential_pauli_xx(real_type const phase, qubit_type const qubit1, qubit_type const qubit2);
    state& exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& qubits);
    state& adj_exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& qubits);
    state& exponential_pauli_y(real_type const phase, qubit_type const qubit);
    state& adj_exponential_pauli_y(real_type const phase, qubit_type const qubit);
    state& exponential_pauli_yy(real_type const phase, qubit_type const qubit1, qubit_type const qubit2);
    state& adj_exponential_pauli_yy(real_type const phase, qubit_type const qubit1, qubit_type const qubit2);
    state& exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& qubits);
    state& adj_exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& qubits);
    state& exponential_pauli_z(real_type const phase, qubit_type const qubit);
    state& adj_exponential_pauli_z(real_type const phase, qubit_type const qubit);
    state& exponential_pauli_zz(real_type const phase, qubit_type const qubit1, qubit_type const qubit2);
    state& adj_exponential_pauli_zz(real_type const phase, qubit_type const qubit1, qubit_type const qubit2);
    state& exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& qubits);
    state& adj_exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& qubits);
    state& exponential_swap(real_type const phase, qubit_type const qubit1, qubit_type const qubit2);
    state& adj_exponential_swap(real_type const phase, qubit_type const qubit1, qubit_type const qubit2);
    state& toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);
    state& adj_toffoli(
      qubit_type const target_qubit,
      control_qubit_type const control_qubit1, control_qubit_type const control_qubit2);

# ifndef BRA_NO_MPI
    state& projective_measurement(qubit_type const qubit, yampi::rank const root);
    state& measurement(yampi::rank const root);
    state& generate_events(yampi::rank const root, int const num_events, int const seed);
    state& exit(yampi::rank const root);
# else // BRA_NO_MPI
    state& projective_measurement(qubit_type const qubit);
    state& measurement();
    state& generate_events(int const num_events, int const seed);
    state& exit();
# endif // BRA_NO_MPI
    state& shor_box(bit_integer_type const num_exponent_qubits, state_integer_type const divisor, state_integer_type const base);

    state& begin_fusion(std::vector<qubit_type> const& fused_qubits);
    state& end_fusion();

    state& clear(qubit_type const qubit);
    state& set(qubit_type const qubit);

    state& depolarizing_channel(real_type const px, real_type const py, real_type const pz, int const seed);

    state& controlled_i_gate(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_i_gate(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_in_gate(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_in_gate(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_hadamard(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_hadamard(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_hadamard(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_hadamard(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_not(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_not(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_not(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_not(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_pauli_xn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_pauli_xn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_pauli_yn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_pauli_yn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_pauli_z(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_pauli_z(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& multi_controlled_swap(qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_swap(qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_sqrt_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_sqrt_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_sqrt_pauli_x(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_sqrt_pauli_x(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_sqrt_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_sqrt_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_sqrt_pauli_y(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_sqrt_pauli_y(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_sqrt_pauli_z(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_sqrt_pauli_z(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_sqrt_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_sqrt_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_phase_shift(complex_type const& phase_coefficient, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_phase_shift(complex_type const& phase_coefficient, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_phase_shift(complex_type const& phase_coefficient, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_phase_shift(complex_type const& phase_coefficient, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_u1(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_u1(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_u1(real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_u1(real_type const phase, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_u2(real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_u2(real_type const phase1, real_type const phase2, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_u2(real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_u2(real_type const phase1, real_type const phase2, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_u3(real_type const phase1, real_type const phase2, real_type const phase3, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_u3(real_type const phase1, real_type const phase2, real_type const phase3, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_u3(real_type const phase1, real_type const phase2, real_type const phase3, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_u3(real_type const phase1, real_type const phase2, real_type const phase3, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_x_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_x_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_x_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_x_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_y_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_y_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_y_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_y_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& multi_controlled_v(complex_type const& phase_coefficient, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_v(complex_type const& phase_coefficient, qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_exponential_pauli_x(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_exponential_pauli_x(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_exponential_pauli_xn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_exponential_pauli_y(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_exponential_pauli_y(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_exponential_pauli_yn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& controlled_exponential_pauli_z(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& adj_controlled_exponential_pauli_z(real_type const phase, qubit_type const target_qubit, control_qubit_type const control_qubit);
    state& multi_controlled_exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_exponential_pauli_zn(real_type const phase, std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits);
    state& multi_controlled_exponential_swap(real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits);
    state& adj_multi_controlled_exponential_swap(real_type const phase, qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits);

   private:
# ifndef BRA_NO_MPI
    virtual unsigned int do_num_page_qubits() const = 0;
    virtual unsigned int do_num_pages() const = 0;

# endif
    virtual void do_i_gate(qubit_type const qubit) = 0;
    virtual void do_adj_i_gate(qubit_type const qubit) = 0;
    virtual void do_ii_gate(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_adj_ii_gate(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_in_gate(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_adj_in_gate(std::vector<qubit_type> const& qubits) = 0;
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
    virtual void do_sqrt_pauli_x(qubit_type const qubit) = 0;
    virtual void do_adj_sqrt_pauli_x(qubit_type const qubit) = 0;
    virtual void do_sqrt_pauli_y(qubit_type const qubit) = 0;
    virtual void do_adj_sqrt_pauli_y(qubit_type const qubit) = 0;
    virtual void do_sqrt_pauli_z(qubit_type const qubit) = 0;
    virtual void do_adj_sqrt_pauli_z(qubit_type const qubit) = 0;
    virtual void do_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_adj_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2) = 0;
    virtual void do_sqrt_pauli_zn(std::vector<qubit_type> const& qubits) = 0;
    virtual void do_adj_sqrt_pauli_zn(std::vector<qubit_type> const& qubits) = 0;
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
    virtual void do_begin_fusion() = 0;
    virtual void do_end_fusion() = 0;
    virtual void do_clear(qubit_type const qubit) = 0;
    virtual void do_set(qubit_type const qubit) = 0;
    virtual void do_controlled_i_gate(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_i_gate(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_in_gate(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_in_gate(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
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
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_adj_controlled_sqrt_pauli_z(
      qubit_type const target_qubit, control_qubit_type const control_qubit) = 0;
    virtual void do_multi_controlled_sqrt_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
      std::vector<control_qubit_type> const& control_qubits) = 0;
    virtual void do_adj_multi_controlled_sqrt_pauli_zn(
      std::vector<qubit_type> const& target_qubits,
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
