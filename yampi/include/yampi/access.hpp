#ifndef YAMPI_ACCESS_HPP
# define YAMPI_ACCESS_HPP

# include <yampi/datatype.hpp>
# include <yampi/environment.hpp>


namespace yampi
{
  struct access
  {
    template <typename Value>
    static ::yampi::datatype derive(
      ::yampi::datatype const& datatype, Value const& value,
      ::yampi::environment const& environment)
    { return value.derive(datatype, environment); }
  };
}


#endif

