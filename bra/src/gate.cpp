#include <boost/config.hpp>

#include <string>
#include <ios>
#include <iomanip>
#include <sstream>

#include <bra/gate/gate.hpp>


namespace bra
{
  namespace gate
  {
    std::string gate::representation() const
    {
      std::ostringstream repr_stream;
      repr_stream << std::left << std::setw(10) << this->name();
      return this->do_representation(repr_stream, 4);
    }
  }
}

