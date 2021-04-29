#ifndef KET_MPI_SHOR_BOX_HPP
# define KET_MPI_SHOR_BOX_HPP

# include <boost/config.hpp>

# include <cassert>
# include <iterator>
# include <type_traits>

# include <boost/range/size.hpp>
# include <boost/range/value_type.hpp>
# ifndef NDEBUG
#   include <boost/range/join.hpp>
# endif // NDEBUG

# include <yampi/environment.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>

# include <ket/shor_box.hpp>
# include <ket/qubit.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/is_unique_if_sorted.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/fill.hpp>
# include <ket/mpi/utility/logger.hpp>


namespace ket
{
  namespace mpi
  {
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename StateInteger, typename Qubits,
      typename BitInteger, typename Allocator>
    inline RandomAccessRange& shor_box(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      StateInteger const base, StateInteger const divisor,
      Qubits const& exponent_qubits, Qubits const& modular_exponentiation_qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using qubit_type = typename boost::range_value<Qubits>::type;
      static_assert(
        (std::is_same<
           StateInteger, typename ::ket::meta::state_integer_of<qubit_type>::type>::value),
        "StateInteger should be state_integer_type of qubit_type");
      static_assert(
        (std::is_same<
           BitInteger, typename ::ket::meta::bit_integer_of<qubit_type>::type>::value),
        "BitInteger should be bit_integer_type of qubit_type");
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      assert(::ket::utility::ranges::is_unique_if_sorted(boost::join(exponent_qubits, modular_exponentiation_qubits)));

      ::ket::mpi::utility::log_with_time_guard<char> print{"Shor", environment};

      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      ::ket::mpi::utility::fill(
        mpi_policy, parallel_policy, local_state, complex_type{real_type{0}},
        communicator, environment);

      auto const num_exponent_qubits = static_cast<BitInteger>(boost::size(exponent_qubits));
      auto const num_exponents = ::ket::utility::integer_exp2<StateInteger>(num_exponent_qubits);
      auto modular_exponentiation_value = StateInteger{1u};

      using std::pow;
      auto const constant_coefficient
        = static_cast<complex_type>(static_cast<real_type>(pow(static_cast<real_type>(num_exponents), -0.5)));

      auto const present_rank = communicator.rank(environment);
      auto const first = ::ket::utility::begin(local_state);
      for (auto exponent = StateInteger{0u}; exponent < num_exponents; ++exponent)
      {
        auto const qubit_value
          = ::ket::shor_box_detail::calculate_index(
              ::ket::shor_box_detail::reverse_bits(exponent, num_exponent_qubits), exponent_qubits,
              modular_exponentiation_value, modular_exponentiation_qubits);

        using ::ket::mpi::permutate_bits;
        auto const rank_index
          = ::ket::mpi::utility::qubit_value_to_rank_index(
              mpi_policy, local_state, permutate_bits(permutation, qubit_value),
              communicator, environment);

        if (rank_index.first == present_rank)
          *(first + rank_index.second) = constant_coefficient;

        modular_exponentiation_value *= base;
        modular_exponentiation_value %= divisor;
      }

      return local_state;
    }

    template <
      typename RandomAccessRange, typename StateInteger, typename Qubits,
      typename BitInteger, typename Allocator>
    inline RandomAccessRange& shor_box(
      RandomAccessRange& local_state,
      StateInteger const base, StateInteger const divisor,
      Qubits const& exponent_qubits, Qubits const& modular_exponentiation_qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return shor_box(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, base, divisor, exponent_qubits, modular_exponentiation_qubits, permutation,
        communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename StateInteger, typename Qubits,
      typename BitInteger, typename Allocator>
    inline RandomAccessRange& shor_box(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      StateInteger const base, StateInteger const divisor,
      Qubits const& exponent_qubits, Qubits const& modular_exponentiation_qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return shor_box(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, base, divisor, exponent_qubits, modular_exponentiation_qubits, permutation,
        communicator, environment);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_SHOR_BOX_HPP
