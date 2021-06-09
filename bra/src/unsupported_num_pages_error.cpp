#ifndef BRA_NO_MPI
# include <string>
# include <sstream>
# include <ios>

# include <bra/unsupported_num_pages_error.hpp>


namespace bra
{
  std::string unsupported_num_pages_error::generate_what_string(unsigned int const num_pages)
  {
    auto output_stream = std::ostringstream{"num_pages=", std::ios_base::ate};
    output_stream << num_pages << " is not supported";
    return output_stream.str();
  }
} // namespace bra


#endif // BRA_NO_MPI
