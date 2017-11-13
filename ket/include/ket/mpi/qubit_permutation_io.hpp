#ifndef KET_MPI_QUBIT_PERMUTATION_IO_HPP
# define KET_MPI_QUBIT_PERMUTATION_IO_HPP

# include <boost/config.hpp>

# include <ostream>

# include <ket/qubit.hpp>
# include <ket/qubit_io.hpp>
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
      typedef ket::qubit<StateInteger, BitInteger> qubit_type;

      qubit_type const last_qubit(
        static_cast<BitInteger>(permutation.size()));

      for (qubit_type qubit = qubit_type(0u); qubit < last_qubit; ++qubit)
        output_stream << '(' << qubit << ',' << permutation[qubit] << ')';

      return output_stream;
    }
  }
}


#endif

