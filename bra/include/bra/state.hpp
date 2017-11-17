#ifndef BRA_STATE_HPP
# define BRA_STATE_HPP

# include <boost/config.hpp>

# include <complex>
# include <vector>
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
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
# include <ket/mpi/qubit_permutation.hpp>

# include <yampi/allocator.hpp>
# include <yampi/datatype.hpp>
# include <yampi/derived_datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/wall_clock.hpp>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define BRA_array std::array
# else
#   define BRA_array boost::array
# endif

# ifndef BOOST_NO_CXX11_HDR_RANDOM
#   define BRA_mt19937_64 std::mt19937_64
# else
#   define BRA_mt19937_64 boost::mt19937_64
# endif


namespace bra
{
  class state
  {
   public:
    typedef boost::uint64_t state_integer_type;
    typedef unsigned int bit_integer_type;
    typedef ket::qubit<state_integer_type, bit_integer_type> qubit_type;
    typedef ket::control<qubit_type> control_qubit_type;

    typedef double real_type;
    typedef std::complex<real_type> complex_type;

    typedef BRA_array<real_type, 3u> spin_type;
    typedef yampi::allocator<spin_type> spins_allocator_type;
    typedef std::vector<spin_type, spins_allocator_type> spins_type;
    typedef BRA_mt19937_64 random_number_generator_type;
    typedef random_number_generator_type::result_type seed_type;

    typedef
      ket::mpi::qubit_permutation<
        state_integer_type, bit_integer_type, yampi::allocator<qubit_type> >
      permutation_type;

   protected:
    bit_integer_type total_num_qubits_;
    boost::optional<spins_type> maybe_expectation_values_;
    state_integer_type measured_value_;
    random_number_generator_type random_number_generator_;

    permutation_type permutation_;
    std::vector<complex_type, yampi::allocator<complex_type> > buffer_;
    yampi::datatype state_integer_datatype_;
    yampi::datatype real_datatype_;
# if MPI_VERSION < 3
    yampi::derived_datatype derived_complex_datatype_;
# endif
    yampi::datatype complex_datatype_;
    yampi::communicator communicator_;
    yampi::environment const& environment_;

    yampi::wall_clock::time_point operations_finish_time_;
    yampi::wall_clock::time_point expectation_values_finish_time_;
    yampi::wall_clock::time_point measurement_finish_time_;

   public:
    state(
      bit_integer_type const total_num_qubits,
      seed_type const seed,
      yampi::communicator const communicator,
      yampi::environment const& environment);

    state(
      std::vector<qubit_type> const& initial_permutation,
      seed_type const seed,
      yampi::communicator const communicator,
      yampi::environment const& environment);

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
    boost::optional<spins_type> const& maybe_expectation_values() const
    { return maybe_expectation_values_; }
    state_integer_type const& measured_value() const { return measured_value_; }
    random_number_generator_type& random_number_generator() { return random_number_generator_; }

    permutation_type const& permutation() const { return permutation_; }

    yampi::datatype const& state_integer_datatype() const { return state_integer_datatype_; }
    yampi::datatype const& real_datatype() const { return real_datatype_; }
    yampi::datatype const& complex_datatype() const { return complex_datatype_; }
    yampi::communicator const& communicator() const { return communicator_; }
    yampi::environment const& environment() const { return environment_; }

    yampi::wall_clock::time_point const& operations_finish_time() const
    { return operations_finish_time_; }
    yampi::wall_clock::time_point const& expectation_values_finish_time() const
    { return expectation_values_finish_time_; }
    yampi::wall_clock::time_point const& measurement_finish_time() const
    { return measurement_finish_time_; }


    unsigned int num_pages() const { return do_num_pages(); }

    ::bra::state& hadamard(qubit_type const qubit)
    { do_hadamard(qubit); return *this; }

    ::bra::state& adj_hadamard(qubit_type const qubit)
    { do_adj_hadamard(qubit); return *this; }

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

    /*
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
    */

    ::bra::state& measurement(yampi::rank const root);

   private:
    virtual unsigned int do_num_pages() const = 0;

    virtual void do_hadamard(qubit_type const qubit) = 0;
    virtual void do_adj_hadamard(qubit_type const qubit) = 0;
    virtual void do_phase_shift(complex_type const phase_coefficient, qubit_type const qubit) = 0;
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
    /*
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
      */
    virtual void do_expectation_values(yampi::rank const root) = 0;
    virtual void do_measure(yampi::rank const root) = 0;
  };
}


# undef BRA_mt19937_64
# undef BRA_array

#endif

