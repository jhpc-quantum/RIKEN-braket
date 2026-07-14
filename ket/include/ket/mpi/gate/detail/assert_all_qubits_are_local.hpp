#ifndef KET_MPI_GATE_DETAIL_ASSERT_ALL_QUBITS_ARE_LOCAL_HPP
# define KET_MPI_GATE_DETAIL_ASSERT_ALL_QUBITS_ARE_LOCAL_HPP

# include <cassert>
# include <algorithm>
# include <iterator>

# include <ket/qubit.hpp>
# ifndef NDEBUG
#   include <ket/meta/state_integer_of.hpp>
#   include <ket/meta/bit_integer_of.hpp>
# endif // NDEBUG
# include <ket/mpi/permutated.hpp>
# ifndef NDEBUG
#   include <ket/mpi/utility/simple_mpi.hpp>
# endif // NDEBUG

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace detail
      {
        namespace runtime
        {
          template <typename BitInteger, typename PermutatedQubitIterator>
          inline auto assert_all_qubits_are_local(
            BitInteger const num_local_qubits,
            PermutatedQubitIterator const permutated_qubit_first, PermutatedQubitIterator const permutated_qubit_last)
          -> void
          {
            using permutated_qubit_type = typename std::iterator_traits<PermutatedQubitIterator>::value_type;
            static_assert(
              std::is_same<BitInteger, ::ket::meta::bit_integer_t<permutated_qubit_type>>::value,
              "BitInteger should be the same as bit_integer_type of value_type of PermutatedQubitIterator");
# ifndef NDEBUG
            using state_integer_type = ::ket::meta::state_integer_t<permutated_qubit_type>;
            auto const least_nonlocal_permutated_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<state_integer_type>(num_local_qubits));
# endif // NDEBUG
            assert(
              std::all_of(
                permutated_qubit_first, permutated_qubit_last,
                [least_nonlocal_permutated_qubit](permutated_qubit_type const permutated_qubit)
                { return ::ket::mpi::remove_control(permutated_qubit) < least_nonlocal_permutated_qubit; }));
          }

          template <typename BitInteger, typename Qubit, typename PermutatedQubitIterator>
          inline auto assert_all_qubits_are_local(
            BitInteger const num_local_qubits,
            ::ket::mpi::permutated<Qubit> const permutated_qubit,
            PermutatedQubitIterator const permutated_qubit_first, PermutatedQubitIterator const permutated_qubit_last)
          -> void
          {
            using permutated_qubit_type = typename std::iterator_traits<PermutatedQubitIterator>::value_type;
            static_assert(
              std::is_same<BitInteger, ::ket::meta::bit_integer_t<permutated_qubit_type>>::value,
              "BitInteger should be the same as bit_integer_type of value_type of PermutatedQubitIterator");
# ifndef NDEBUG
            using state_integer_type = ::ket::meta::state_integer_t<permutated_qubit_type>;
            auto const least_nonlocal_permutated_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<state_integer_type>(num_local_qubits));
# endif // NDEBUG
            assert(::ket::mpi::remove_control(permutated_qubit) < least_nonlocal_permutated_qubit);
            ::ket::mpi::gate::detail::runtime::assert_all_qubits_are_local(num_local_qubits, permutated_qubit_first, permutated_qubit_last);
          }

          template <typename BitInteger, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
          inline auto assert_all_qubits_are_local(
            BitInteger const num_local_qubits,
            PermutatedQubitIterator1 const permutated_qubit_first1, PermutatedQubitIterator1 const permutated_qubit_last1,
            PermutatedQubitIterator2 const permutated_qubit_first2, PermutatedQubitIterator2 const permutated_qubit_last2)
          -> void
          {
            ::ket::mpi::gate::detail::runtime::assert_all_qubits_are_local(num_local_qubits, permutated_qubit_first1, permutated_qubit_last1);
            ::ket::mpi::gate::detail::runtime::assert_all_qubits_are_local(num_local_qubits, permutated_qubit_first2, permutated_qubit_last2);
          }

          template <typename MpiPolicy, typename RandomAccessRange, typename PermutatedQubitIterator>
          inline auto assert_all_qubits_are_local(
            MpiPolicy const& mpi_policy, RandomAccessRange& local_state,
            yampi::communicator const& communicator, yampi::environment const& environment,
            PermutatedQubitIterator const permutated_qubit_first, PermutatedQubitIterator const permutated_qubit_last)
          -> void
          {
            using permutated_qubit_type = typename std::iterator_traits<PermutatedQubitIterator>::value_type;
            using bit_integer_type = ::ket::meta::bit_integer_t<permutated_qubit_type>;
            ::ket::mpi::gate::detail::runtime::assert_all_qubits_are_local(
              static_cast<bit_integer_type>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment)),
              permutated_qubit_first, permutated_qubit_last);
          }

          template <typename MpiPolicy, typename RandomAccessRange, typename Qubit, typename PermutatedQubitIterator>
          inline auto assert_all_qubits_are_local(
            MpiPolicy const& mpi_policy, RandomAccessRange& local_state,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::mpi::permutated<Qubit> const permutated_qubit,
            PermutatedQubitIterator const permutated_qubit_first, PermutatedQubitIterator const permutated_qubit_last)
          -> void
          {
            using permutated_qubit_type = typename std::iterator_traits<PermutatedQubitIterator>::value_type;
            using bit_integer_type = ::ket::meta::bit_integer_t<permutated_qubit_type>;
            ::ket::mpi::gate::detail::runtime::assert_all_qubits_are_local(
              static_cast<bit_integer_type>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment)),
              permutated_qubit, permutated_qubit_first, permutated_qubit_last);
          }

          template <typename MpiPolicy, typename RandomAccessRange, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
          inline auto assert_all_qubits_are_local(
            MpiPolicy const& mpi_policy, RandomAccessRange& local_state,
            yampi::communicator const& communicator, yampi::environment const& environment,
            PermutatedQubitIterator1 const permutated_qubit_first1, PermutatedQubitIterator1 const permutated_qubit_last1,
            PermutatedQubitIterator2 const permutated_qubit_first2, PermutatedQubitIterator2 const permutated_qubit_last2)
          -> void
          {
            using permutated_qubit_type = typename std::iterator_traits<PermutatedQubitIterator1>::value_type;
            using bit_integer_type = ::ket::meta::bit_integer_t<permutated_qubit_type>;
            ::ket::mpi::gate::detail::runtime::assert_all_qubits_are_local(
              static_cast<bit_integer_type>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment)),
              permutated_qubit_first1, permutated_qubit_last1, permutated_qubit_first2, permutated_qubit_last2);
          }

          namespace ranges
          {
            template <typename MpiPolicy, typename RandomAccessRange, typename PermutatedQubits>
            inline auto assert_all_qubits_are_local(
              MpiPolicy const& mpi_policy, RandomAccessRange& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment,
              PermutatedQubits const& permutated_qubits)
            -> void
            {
              using std::begin;
              using std::end;
              ::ket::mpi::gate::detail::runtime::assert_all_qubits_are_local(
                mpi_policy, local_state, communicator, environment,
                begin(permutated_qubits), end(permutated_qubits));
            }

            template <typename MpiPolicy, typename RandomAccessRange, typename Qubit, typename PermutatedQubits>
            inline auto assert_all_qubits_are_local(
              MpiPolicy const& mpi_policy, RandomAccessRange& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment,
              ::ket::mpi::permutated<Qubit> const permutated_qubit,
              PermutatedQubits const& permutated_qubits)
            -> void
            {
              using std::begin;
              using std::end;
              ::ket::mpi::gate::detail::runtime::assert_all_qubits_are_local(
                mpi_policy, local_state, communicator, environment,
                permutated_qubit, begin(permutated_qubits), end(permutated_qubits));
            }

            template <typename MpiPolicy, typename RandomAccessRange, typename PermutatedQubits1, typename PermutatedQubits2>
            inline auto assert_all_qubits_are_local(
              MpiPolicy const& mpi_policy, RandomAccessRange& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment,
              PermutatedQubits1 const& permutated_qubits1, PermutatedQubits2 const& permutated_qubits2)
            -> void
            {
              using std::begin;
              using std::end;
              ::ket::mpi::gate::detail::runtime::assert_all_qubits_are_local(
                mpi_policy, local_state, communicator, environment,
                begin(permutated_qubits1), end(permutated_qubits1), begin(permutated_qubits2), end(permutated_qubits2));
            }
          } // namespace ranges
        } // namespace runtime

        template <typename BitInteger>
        inline auto assert_all_qubits_are_local(BitInteger const num_local_qubits) -> void
        { }

        template <typename BitInteger, typename PermutatedQubit, typename... PermutatedQubits>
        inline auto assert_all_qubits_are_local(
          BitInteger const num_local_qubits,
          PermutatedQubit const permutated_qubit, PermutatedQubits const... permutated_qubits)
        -> void
        {
          static_assert(std::is_same<BitInteger, ::ket::meta::bit_integer_t<PermutatedQubit>>::value, "BitInteger should be the same as bit_integer_type of PermutatedQubit");
# ifndef NDEBUG
          using state_integer_type = ::ket::meta::state_integer_t<PermutatedQubit>;
          auto const least_nonlocal_permutated_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<state_integer_type>(num_local_qubits));
# endif // NDEBUG
          assert(::ket::mpi::remove_control(permutated_qubit) < least_nonlocal_permutated_qubit);
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(num_local_qubits, permutated_qubits...);
        }

        template <typename MpiPolicy, typename RandomAccessRange>
        inline auto assert_all_qubits_are_local(
          MpiPolicy const& mpi_policy, RandomAccessRange& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> void
        { }

        template <typename MpiPolicy, typename RandomAccessRange, typename PermutatedQubit, typename... PermutatedQubits>
        inline auto assert_all_qubits_are_local(
          MpiPolicy const& mpi_policy, RandomAccessRange& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment,
          PermutatedQubit const permutated_qubit, PermutatedQubits const... permutated_qubits)
        -> void
        {
          using bit_integer_type = ::ket::meta::bit_integer_t<PermutatedQubit>;
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            static_cast<bit_integer_type>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment)),
            permutated_qubits...);
        }
      } // namespace detail
    } // namespace gate
  } // namespace mpi
} // namespace ket

#endif // KET_MPI_GATE_DETAIL_ASSERT_ALL_QUBITS_ARE_LOCAL_HPP
