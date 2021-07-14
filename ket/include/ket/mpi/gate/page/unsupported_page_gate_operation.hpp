#ifndef KET_MPI_GATE_PAGE_UNSUPPORTED_PAGE_GATE_OPERATION_HPP
# define KET_MPI_GATE_PAGE_UNSUPPORTED_PAGE_GATE_OPERATION_HPP

# include <stdexcept>
# include <string>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        class unsupported_page_gate_operation
          : public std::runtime_error
        {
         public:
          explicit unsupported_page_gate_operation(std::string const& gate_name)
            : std::runtime_error{"page is not supported for gate " + gate_name}
          { }
        }; // class unsupported_page_gate_operation
      }
    }
  }
}


#endif // KET_MPI_GATE_PAGE_UNSUPPORTED_PAGE_GATE_OPERATION_HPP
