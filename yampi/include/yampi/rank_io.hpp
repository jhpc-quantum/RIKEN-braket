#ifndef YAMPI_RANK_IO_HPP
# define YAMPI_RANK_IO_HPP

# include <ostream>

# include <yampi/rank.hpp>


namespace yampi
{
  template <typename Character, typename CharacterTraits>
  inline std::basic_ostream<Character, CharacterTraits>&
  operator<<(
    std::basic_ostream<Character, CharacterTraits>& output_stream,
    ::yampi::rank const& rank)
  { return output_stream << rank.mpi_rank(); }
}


#endif

