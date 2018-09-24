#ifndef KET_MPI_UTILITY_DEBUG_PRINT_DATA_HPP
# define KET_MPI_UTILITY_DEBUG_PRINT_DATA_HPP

# include <boost/config.hpp>

# include <ostream>

# include <yampi/environment.hpp>
# include <yampi/communicator.hpp>

# include <ket/utility/begin.hpp>
# include <ket/utility/meta/const_iterator_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace debug
      {
        template <
          typename MpiPolicy, typename Character, typename CharacterTraits, typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        std::basic_ostream<Character, CharacterTraits>& print_data(
          MpiPolicy const mpi_policy,
          std::basic_ostream<Character, CharacterTraits>& output_stream,
          RandomAccessRange const& local_state,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          typedef
            typename ::ket::utility::meta::const_iterator_of<RandomAccessRange const>::type
            local_state_iterator;
          local_state_iterator const local_state_first
            = ::ket::utility::begin(local_state);

          StateInteger const num_local_states
            = static_cast<StateInteger>(boost::size(local_state));

          for (StateInteger local_index = 0; local_index < num_local_states; ++local_index)
          {
            using ket::mpi::inverse_permutate_bits;
            using ket::mpi::utility::rank_index_to_qubit_value;
            StateInteger const qubit_value
              = inverse_permutate_bits(
                  permutation,
                  rank_index_to_qubit_value(
                    mpi_policy, local_state, communicator.rank(environment), local_index));

            output_stream << '[' << qubit_value << ": " << *(local_state_first+local_index) << "] ";
          }

          return output_stream;
        }
      }
    }
  }
}


#endif

