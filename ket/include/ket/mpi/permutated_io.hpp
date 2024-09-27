#ifndef KET_MPI_PERMUTATED_IO_HPP
# define KET_MPI_PERMUTATED_IO_HPP

# include <istream>
# include <ostream>

# include <ket/mpi/permutated.hpp>


namespace ket
{
  template <
    typename Character, typename CharacterTraits, typename Qubit>
  inline auto operator<<(
    std::basic_ostream<Character, CharacterTraits>& output_stream,
    ::ket::mpi::permutated<Qubit> const permutated_qubit)
  -> std::basic_ostream<Character, CharacterTraits>&
  { return output_stream << permutated_qubit.qubit(); }

  template <
    typename Character, typename CharacterTraits, typename Qubit>
  inline auto operator>>(
    std::basic_ostream<Character, CharacterTraits>& input_stream,
    ::ket::mpi::permutated<Qubit>& permutated_qubit)
  -> std::basic_istream<Character, CharacterTraits>&
  { return input_stream >> permutated_qubit.qubit(); }
} // namespace ket


#endif // KET_MPI_PERMUTATED_IO_HPP
