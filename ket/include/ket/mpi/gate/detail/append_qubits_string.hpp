#ifndef KET_MPI_GATE_DETAIL_APPEND_QUBITS_STRING_HPP
# define KET_MPI_GATE_DETAIL_APPEND_QUBITS_STRING_HPP

# include <string>

# include <ket/qubit.hpp>
# include <ket/mpi/utility/logger.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace detail
      {
        inline std::string append_qubits_string(std::string const& result)
        { return result; }

        template <typename StateInteger, typename BitInteger, typename... Qubits>
        inline std::string append_qubits_string(std::string const& result, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
        { return append_qubits_string(::ket::mpi::utility::generate_logger_string(result, ' ', qubit), qubits...); }
      } // namespace detail
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_DETAIL_APPEND_QUBITS_STRING_HPP
