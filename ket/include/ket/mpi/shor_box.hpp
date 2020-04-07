#ifndef KET_MPI_SHOR_BOX_HPP
# define KET_MPI_SHOR_BOX_HPP

# include <boost/config.hpp>

# include <cassert>
# include <iterator>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_unsigned.hpp>
#   include <boost/type_traits/is_same.hpp>
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <boost/range/size.hpp>
# include <boost/range/value_type.hpp>
# include <boost/range/join.hpp>

# include <yampi/environment.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>

# include <ket/shor_box.hpp>
# include <ket/qubit.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/is_unique.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/meta/iterator_of.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_is_unsigned std::is_unsigned
#   define KET_is_same std::is_same
# else
#   define KET_is_unsigned boost::is_unsigned
#   define KET_is_same boost::is_same
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif


namespace ket
{
  namespace mpi
  {
    namespace shor_box_detail
    {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
      template <typename ParallelPolicy>
      struct call_fill
      {
        ParallelPolicy parallel_policy_;

        explicit call_fill(ParallelPolicy const parallel_policy) : parallel_policy_(parallel_policy) { }

        template <typename RandomAccessIterator>
        void operator()(
          RandomAccessIterator const first,
          RandomAccessIterator const last) const
        {
          typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
          typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
          ::ket::utility::fill(parallel_policy_, first, last, static_cast<complex_type>(static_cast<real_type>(0)));
        }
      };

      template <typename ParallelPolicy>
      inline call_fill<ParallelPolicy> make_call_fill(ParallelPolicy const parallel_policy)
      { return call_fill<ParallelPolicy>(parallel_policy); }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
    }

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
      typedef typename boost::range_value<Qubits>::type qubit_type;
      typedef typename ::ket::meta::bit_integer_of<qubit_type>::type bit_integer_type;
      static_assert(
        (KET_is_same<
           StateInteger, typename ::ket::meta::state_integer_of<qubit_type>::type>::value),
        "StateInteger should be state_integer_type of qubit_type");
      static_assert(KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(KET_is_unsigned<bit_integer_type>::value, "BitInteger should be unsigned");

      assert(::ket::utility::ranges::is_unique(boost::join(exponent_qubits, modular_exponentiation_qubits)));

      ::ket::mpi::utility::log_with_time_guard<char> print("Shor", environment);

      typedef typename boost::range_value<RandomAccessRange>::type complex_type;
      typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
      ::ket::mpi::utility::for_each_local_range(
        mpi_policy, local_state,
        [parallel_policy](auto const first, auto const last)
        { ::ket::utility::fill(parallel_policy, first, last, static_cast<complex_type>(static_cast<real_type>(0))); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
      ::ket::mpi::utility::for_each_local_range(
        mpi_policy, local_state,
        ::ket::mpi::shor_box_detail::make_call_fill(parallel_policy));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

      bit_integer_type const num_exponent_qubits = static_cast<bit_integer_type>(boost::size(exponent_qubits));
      StateInteger const num_exponents = ::ket::utility::integer_exp2<StateInteger>(num_exponent_qubits);
      StateInteger modular_exponentiation_value = static_cast<StateInteger>(1u);

      using std::pow;
      complex_type const constant_coefficient
        = static_cast<complex_type>(static_cast<real_type>(pow(num_exponents, -0.5)));

      yampi::rank const present_rank = communicator.rank(environment);
      typename ::ket::utility::meta::iterator_of<RandomAccessRange>::type iter = ::ket::utility::begin(local_state);
      for (StateInteger exponent = static_cast<StateInteger>(0u); exponent < num_exponents; ++exponent)
      {
        StateInteger const qubit_value
          = ::ket::shor_box_detail::calculate_index(
              ::ket::shor_box_detail::reverse_bits(exponent, num_exponent_qubits), exponent_qubits,
              modular_exponentiation_value, modular_exponentiation_qubits);

        using ::ket::mpi::permutate_bits;
        std::pair<yampi::rank, StateInteger> const rank_index
          = ::ket::mpi::utility::qubit_value_to_rank_index(
              mpi_policy, local_state, permutate_bits(permutation, qubit_value));

        if (rank_index.first == present_rank)
          *(iter + rank_index.second) = constant_coefficient;

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


# undef KET_is_unsigned
# undef KET_is_same
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

