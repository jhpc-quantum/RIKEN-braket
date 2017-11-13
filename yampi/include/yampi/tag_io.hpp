#ifndef YAMPI_TAG_IO_HPP
# define YAMPI_TAG_IO_HPP

# include <ostream>

# include <yampi/tag.hpp>


namespace yampi
{
  template <typename Character, typename CharacterTraits>
  inline std::basic_ostream<Character, CharacterTraits>&
  operator<<(
    std::basic_ostream<Character, CharacterTraits>& output_stream,
    ::yampi::tag const& tag)
  { return output_stream << tag.mpi_tag(); }
}


#endif

