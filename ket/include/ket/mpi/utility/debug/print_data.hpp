#ifndef KET_MPI_UTILITY_DEBUG_PRINT_DATA_HPP
# define KET_MPI_UTILITY_DEBUG_PRINT_DATA_HPP

# include <ostream>
# include <iterator>

# include <yampi/environment.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/send.hpp>
# include <yampi/receive.hpp>

# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>


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
          MpiPolicy const& mpi_policy,
          std::basic_ostream<Character, CharacterTraits>& output_stream,
          RandomAccessRange const& local_state,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation,
          yampi::rank const root,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          auto const first = std::begin(local_state);
          auto const present_rank = communicator.rank(environment);

          auto const num_qubits = ::ket::mpi::utility::policy::num_qubits(mpi_policy, local_state, communicator, environment);
          auto const last_qubit_value = StateInteger{1u} << num_qubits;
          for (auto qubit_value = StateInteger{0u}; qubit_value < last_qubit_value; ++qubit_value)
          {
            using ket::mpi::utility::qubit_value_to_rank_index;
            using ket::mpi::permutate_bits;
            auto const rank_index
              = qubit_value_to_rank_index(
                  mpi_policy, local_state, permutate_bits(permutation, qubit_value), communicator, environment);

            if (present_rank == root)
            {
              auto coefficient = *first;

              if (present_rank == rank_index.first)
                coefficient = *(first + rank_index.second);
              else
                yampi::receive(yampi::ignore_status, yampi::make_buffer(coefficient), rank_index.first, yampi::tag{static_cast<int>(rank_index.second)}, communicator, environment);

              output_stream << '[' << qubit_value << ": " << coefficient << "] ";
            }
            else if (present_rank == rank_index.first)
              yampi::send(yampi::make_buffer(*(first + rank_index.second)), root, yampi::tag{static_cast<int>(rank_index.second)}, communicator, environment);
          }

          return output_stream;
        }
      } // namespace debug
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_DEBUG_PRINT_DATA_HPP
