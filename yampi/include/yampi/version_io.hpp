#ifndef YAMPI_VERSION_IO_HPP
# define YAMPI_VERSION_IO_HPP

# include <ostream>

# include <yampi/version.hpp>


namespace yampi
{
  template <typename Character, typename CharacterTraits>
  inline std::basic_ostream<Character, CharacterTraits>&
  operator<<(
    std::basic_ostream<Character, CharacterTraits>& output_stream,
    ::yampi::mpi_version const& version)
  { return output_stream << version.major() << '.' << version.minor(); }
}


#endif

