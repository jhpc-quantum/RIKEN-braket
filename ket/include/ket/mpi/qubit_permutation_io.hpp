#ifndef KET_MPI_QUBIT_PERMUTATION_IO_HPP
# define KET_MPI_QUBIT_PERMUTATION_IO_HPP

# include <ostream>

# include <ket/qubit.hpp>
# include <ket/qubit_io.hpp>
# include <ket/mpi/permutated_io.hpp>
# include <ket/mpi/qubit_permutation.hpp>


namespace ket
{
  namespace mpi
  {
    template <
      typename Character, typename CharacterTraits,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline std::basic_ostream<Character, CharacterTraits>& operator<<(
      std::basic_ostream<Character, CharacterTraits>& output_stream,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const&
        permutation)
    {
      using qubit_type = ket::qubit<StateInteger, BitInteger>;
      auto const last_qubit = qubit_type{static_cast<BitInteger>(permutation.size())};

      for (auto qubit = qubit_type{0u}; qubit < last_qubit; ++qubit)
        output_stream << '(' << qubit << ',' << permutation[qubit] << ')';

      return output_stream;
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_QUBIT_PERMUTATION_IO_HPP
