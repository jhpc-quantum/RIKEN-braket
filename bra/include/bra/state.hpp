#ifndef BRA_STATE_HPP
# define BRA_STATE_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <complex>
# include <vector>
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif
# include <utility>
# ifdef BRA_NO_MPI
#   ifndef BOOST_NO_CXX11_HDR_CHRONO
#     include <chrono>
#   else
#     include <boost/chrono/chrono.hpp>
#   endif
#   include <memory>
# endif

# include <boost/cstdint.hpp>

# ifndef BOOST_NO_CXX11_HDR_RANDOM
#   include <random>
# else
#   include <boost/random/mersenne_twister.hpp>
# endif

# include <boost/optional.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/gate/projective_measurement.hpp>
# ifndef BRA_NO_MPI
#   include <ket/mpi/qubit_permutation.hpp>

#   include <yampi/allocator.hpp>
#   include <yampi/datatype.hpp>
#   include <yampi/uncommitted_datatype.hpp>
#   include <yampi/rank.hpp>
#   include <yampi/communicator.hpp>
#   include <yampi/environment.hpp>
#   include <yampi/wall_clock.hpp>
# endif // BRA_NO_MPI

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define BRA_array std::array
# else
#   define BRA_array boost::array
# endif

# ifndef BOOST_NO_CXX11_HDR_RANDOM
#   define BRA_mt19937_64 std::mt19937_64
# else
#   define BRA_mt19937_64 boost::random::mt19937_64
# endif

# ifndef BRA_NO_MPI
#   define BRA_clock yampi::wall_clock
# else
#   ifndef BOOST_NO_CXX11_HDR_CHRONO
#     define BRA_clock std::chrono::system_clock
#   else
#     define BRA_clock boost::chrono::system_clock
#   endif
# endif


namespace bra
{
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
  enum class finished_process : int { operations, begin_measurement, generate_events, ket_measure };

#  define BRA_FINISHED_PROCESS_TYPE bra::finished_process
#  define BRA_FINISHED_PROCESS_VALUE(value) bra::finished_process::value
# else // BOOST_NO_CXX11_SCOPED_ENUMS
  namespace finished_process_ { enum finished_process { operations, begin_measurement, generate_events, ket_measure }; }

#  define BRA_FINISHED_PROCESS_TYPE bra::finished_process_::finished_process
#  define BRA_FINISHED_PROCESS_VALUE(value) bra::finished_process_::value
# endif // BOOST_NO_CXX11_SCOPED_ENUMS

  class state
  {
   public:
    typedef boost::uint64_t state_integer_type;
    typedef unsigned int bit_integer_type;
    typedef ket::qubit<state_integer_type, bit_integer_type> qubit_type;
    typedef ket::control<qubit_type> control_qubit_type;

# ifdef BRA_REAL_TYPE
#   if BRA_REAL_TYPE == 0
    typedef long double real_type;
#   elif BRA_REAL_TYPE == 1
    typedef double real_type;
#   elif BRA_REAL_TYPE == 2
    typedef float real_type;
#   else
    typedef double real_type;
#   endif
# else // BRA_REAL_TYPE
    typedef double real_type;
# endif
    typedef std::complex<real_type> complex_type;

    typedef BRA_array<real_type, 3u> spin_type;
# ifndef BRA_NO_MPI
    typedef yampi::allocator<spin_type> spins_allocator_type;
# else
    typedef std::allocator<spin_type> spins_allocator_type;
# endif
    typedef std::vector<spin_type, spins_allocator_type> spins_type;
    typedef BRA_mt19937_64 random_number_generator_type;
    typedef random_number_generator_type::result_type seed_type;

# ifndef BRA_NO_MPI
    typedef
      ket::mpi::qubit_permutation<
        state_integer_type, bit_integer_type, yampi::allocator<qubit_type> >
      permutation_type;
# endif

    typedef
      std::pair<BRA_clock::time_point, BRA_FINISHED_PROCESS_TYPE>
      time_and_process_type;

   protected:
    bit_integer_type total_num_qubits_;
    std::vector<KET_GATE_OUTCOME_TYPE> last_outcomes_; // return values of ket(::mpi)::gate::projective_measurement
    boost::optional<spins_type> maybe_expectation_values_; // return value of ket(::mpi)::all_spin_expectation_values
    state_integer_type measured_value_; // return value of ket(::mpi)::measure
    std::vector<state_integer_type> generated_events_; // results of ket(::mpi)::generate_events
    random_number_generator_type random_number_generator_;
# ifndef BRA_NO_MPI

    permutation_type permutation_;
    std::vector<complex_type, yampi::allocator<complex_type> > buffer_;
    yampi::datatype state_integer_datatype_;
    yampi::datatype real_datatype_;
    yampi::uncommitted_datatype uncommitted_real_pair_datatype_;
    yampi::datatype real_pair_datatype_;
    yampi::datatype complex_datatype_;
    yampi::communicator const& communicator_;
    yampi::environment const& environment_;
# endif

    std::vector<time_and_process_type> finish_times_and_processes_;

   public:
# ifndef BRA_NO_MPI
    state(
      bit_integer_type const total_num_qubits,
      seed_type const seed,
      yampi::communicator const& communicator,
      yampi::environment const& environment);

    state(
      std::vector<qubit_type> const& initial_permutation,
      seed_type const seed,
      yampi::communicator const& communicator,
      yampi::environment const& environment);
# else // BRA_NO_MPI
    state(bit_integer_type const total_num_qubits, seed_type const seed);
# endif // BRA_NO_MPI

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    virtual ~state() = default;
# else
    virtual ~state() { }
# endif

   private:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    state(state const&) = delete;
    state& operator=(state const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    state(state&&) = delete;
    state& operator=(state&&) = delete;
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
    state(state const&);
    state& operator=(state const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    state(state&&);
    state& operator=(state&&);
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

   public:
    bit_integer_type const& total_num_qubits() const { return total_num_qubits_; }

    bool is_measured(qubit_type const qubit) const
    {
      return last_outcomes_[static_cast<bit_integer_type>(qubit)]
        != KET_GATE_OUTCOME_VALUE(unspecified);
    }
    int outcome(qubit_type const qubit) const
    { return static_cast<int>(last_outcomes_[static_cast<bit_integer_type>(qubit)]); }

    boost::optional<spins_type> const& maybe_expectation_values() const
    { return maybe_expectation_values_; }
    state_integer_type const& measured_value() const { return measured_value_; }
    std::vector<state_integer_type> const& generated_events() const { return generated_events_; }
    random_number_generator_type& random_number_generator() { return random_number_generator_; }

# ifndef BRA_NO_MPI
    permutation_type const& permutation() const { return permutation_; }

    yampi::datatype const& state_integer_datatype() const { return state_integer_datatype_; }
    yampi::datatype const& real_datatype() const { return real_datatype_; }
    yampi::datatype const& complex_datatype() const { return complex_datatype_; }
    yampi::communicator const& communicator() const { return communicator_; }
    yampi::environment const& environment() const { return environment_; }
# endif

    std::size_t num_finish_processes() const { return finish_times_and_processes_.size(); }
    time_and_process_type const& finish_time_and_process(std::size_t const n) const
    { return finish_times_and_processes_[n]; }


# ifndef BRA_NO_MPI
    unsigned int num_page_qubits() const { return do_num_page_qubits(); }
    unsigned int num_pages() const { return do_num_pages(); }
# endif

    ::bra::state& hadamard(qubit_type const qubit)
    { do_hadamard(qubit); return *this; }

    ::bra::state& adj_hadamard(qubit_type const qubit)
    { do_adj_hadamard(qubit); return *this; }

    ::bra::state& pauli_x(qubit_type const qubit)
    { do_pauli_x(qubit); return *this; }

    ::bra::state& adj_pauli_x(qubit_type const qubit)
    { do_adj_pauli_x(qubit); return *this; }

    ::bra::state& pauli_y(qubit_type const qubit)
    { do_pauli_y(qubit); return *this; }

    ::bra::state& adj_pauli_y(qubit_type const qubit)
    { do_adj_pauli_y(qubit); return *this; }

    ::bra::state& pauli_z(qubit_type const qubit)
    { do_pauli_z(qubit); return *this; }

    ::bra::state& adj_pauli_z(qubit_type const qubit)
    { do_adj_pauli_z(qubit); return *this; }

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
      complex_type const phase_coefficient, qubit_type const qubit)
    { do_phase_shift(phase_coefficient, qubit); return *this; }

    ::bra::state& adj_phase_shift(
      complex_type const phase_coefficient, qubit_type const qubit)
    { do_adj_phase_shift(phase_coefficient, qubit); return *this; }

    ::bra::state& x_rotation_half_pi(qubit_type const qubit)
    { do_x_rotation_half_pi(qubit); return *this; }

    ::bra::state& adj_x_rotation_half_pi(qubit_type const qubit)
    { do_adj_x_rotation_half_pi(qubit); return *this; }

    ::bra::state& y_rotation_half_pi(qubit_type const qubit)
    { do_y_rotation_half_pi(qubit); return *this; }

    ::bra::state& adj_y_rotation_half_pi(qubit_type const qubit)
    { do_adj_y_rotation_half_pi(qubit); return *this; }

    ::bra::state& controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_controlled_not(target_qubit, control_qubit); return *this; }

    ::bra::state& adj_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    { do_adj_controlled_not(target_qubit, control_qubit); return *this; }

    ::bra::state& controlled_phase_shift(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    {
      do_controlled_phase_shift(phase_coefficient, target_qubit, control_qubit);
      return *this;
    }

    ::bra::state& adj_controlled_phase_shift(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    {
      do_adj_controlled_phase_shift(phase_coefficient, target_qubit, control_qubit);
      return *this;
    }

    ::bra::state& controlled_v(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    {
      do_controlled_v(phase_coefficient, target_qubit, control_qubit);
      return *this;
    }

    ::bra::state& adj_controlled_v(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
    {
      do_adj_controlled_v(phase_coefficient, target_qubit, control_qubit);
      return *this;
    }

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

    ::bra::state& shor_box(bit_integer_type const num_exponent_qubits, state_integer_type const divisor, state_integer_type const base)
    { do_shor_box(num_exponent_qubits, divisor, base); return *this; }

    ::bra::state& clear(qubit_type const qubit)
    { do_clear(qubit); return *this; }

    ::bra::state& set(qubit_type const qubit)
    { do_set(qubit); return *this; }

    ::bra::state& depolarizing_channel(real_type const px, real_type const py, real_type const pz, int const seed)
    { do_depolarizing_channel(px, py, pz, seed); return *this; }

   private:
# ifndef BRA_NO_MPI
    virtual unsigned int do_num_page_qubits() const = 0;
    virtual unsigned int do_num_pages() const = 0;

# endif
    virtual void do_hadamard(qubit_type const qubit) = 0;
    virtual void do_adj_hadamard(qubit_type const qubit) = 0;
    virtual void do_pauli_x(qubit_type const qubit) = 0;
    virtual void do_adj_pauli_x(qubit_type const qubit) = 0;
    virtual void do_pauli_y(qubit_type const qubit) = 0;
    virtual void do_adj_pauli_y(qubit_type const qubit) = 0;
    virtual void do_pauli_z(qubit_type const qubit) = 0;
    virtual void do_adj_pauli_z(qubit_type const qubit) = 0;
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
      complex_type const phase_coefficient, qubit_type const qubit) = 0;
    virtual void do_adj_phase_shift(
      complex_type const phase_coefficient, qubit_type const qubit) = 0;
    virtual void do_x_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_adj_x_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_y_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_adj_y_rotation_half_pi(qubit_type const qubit) = 0;
    virtual void do_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
      = 0;
    virtual void do_adj_controlled_not(
      qubit_type const target_qubit, control_qubit_type const control_qubit)
      = 0;
    virtual void do_controlled_phase_shift(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
      = 0;
    virtual void do_adj_controlled_phase_shift(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
      = 0;
    virtual void do_controlled_v(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
      = 0;
    virtual void do_adj_controlled_v(
      complex_type const phase_coefficient,
      qubit_type const target_qubit, control_qubit_type const control_qubit)
      = 0;
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
    virtual KET_GATE_OUTCOME_TYPE do_projective_measurement(
      qubit_type const qubit, yampi::rank const root) = 0;
    virtual void do_expectation_values(yampi::rank const root) = 0;
    virtual void do_measure(yampi::rank const root) = 0;
    virtual void do_generate_events(yampi::rank const root, int const num_events, int const seed) = 0;
# else // BRA_NO_MPI
    virtual KET_GATE_OUTCOME_TYPE do_projective_measurement(qubit_type const qubit) = 0;
    virtual void do_expectation_values() = 0;
    virtual void do_measure() = 0;
    virtual void do_generate_events(int const num_events, int const seed) = 0;
# endif // BRA_NO_MPI
    virtual void do_shor_box(
      bit_integer_type const num_exponent_qubits,
      state_integer_type const divisor, state_integer_type const base) = 0;
    virtual void do_clear(qubit_type const qubit) = 0;
    virtual void do_set(qubit_type const qubit) = 0;
    virtual void do_depolarizing_channel(real_type const px, real_type const py, real_type const pz, int const seed) = 0;
  };
}


# undef BRA_clock
# undef BRA_mt19937_64
# undef BRA_array

#endif

