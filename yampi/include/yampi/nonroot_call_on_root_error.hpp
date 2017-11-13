#ifndef YAMPI_NONROOT_CALL_ON_ROOT_ERROR_HPP
# define YAMPI_NONROOT_CALL_ON_ROOT_ERROR_HPP

# include <boost/config.hpp>

# include <stdexcept>
# include <string>


namespace yampi
{
  class nonroot_call_on_root_error
    : public std::logic_error
  {
   public:
    nonroot_call_on_root_error(std::string const& where)
      : std::logic_error((std::string("Non-root call on root in ") + where).c_str())
    { }
  };
}


#endif

