#ifndef BRA_UNSUPPORTED_NUM_PAGES_ERROR_HPP
# define BRA_UNSUPPORTED_NUM_PAGES_ERROR_HPP

# ifndef BRA_NO_MPI
#   include <string>
#   include <stdexcept>


namespace bra
{
  class unsupported_num_pages_error
    : public std::logic_error
  {
   public:
    explicit unsupported_num_pages_error(unsigned int const num_pages)
      : std::logic_error{generate_what_string(num_pages).c_str()}
    { }

   private:
    std::string generate_what_string(unsigned int const num_pages);
  }; // class unsupported_num_pages_error
} // namespace bra


# endif // BRA_NO_MPI

#endif // BRA_UNSUPPORTED_NUM_PAGES_ERROR_HPP
