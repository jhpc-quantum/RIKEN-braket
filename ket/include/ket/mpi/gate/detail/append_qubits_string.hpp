#ifndef KET_MPI_GATE_DETAIL_APPEND_QUBITS_STRING_HPP
# define KET_MPI_GATE_DETAIL_APPEND_QUBITS_STRING_HPP

# include <string>
# ifdef KET_PRINT_LOG
#   include <sstream>
#   include <utility>

#   include <ket/qubit.hpp>
#   include <ket/qubit_io.hpp>
# endif // KET_PRINT_LOG


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace detail
      {
# ifdef KET_PRINT_LOG
        namespace append_qubits_string_detail
        {
          template <typename Character, typename CharacterTraits, typename Allocator>
          inline void insert(std::basic_ostringstream<Character, CharacterTraits, Allocator>&)
          { }

          template <typename Character, typename CharacterTraits, typename Allocator, typename StateInteger, typename BitInteger, typename... Qubits>
          inline void insert(std::basic_ostringstream<Character, CharacterTraits, Allocator>& output_string_stream, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
          {
            output_string_stream << ' ' << qubit;
            ::ket::mpi::utility::logger_detail::insert(output_string_stream, qubits...);
          }
        } // namespace append_qubits_string_detail

        template <typename Character, typename CharacterTraits, typename Allocator, typename... Qubits>
        inline std::basic_string<Character, CharacterTraits, Allocator>
        append_qubits_string(std::basic_string<Character, CharacterTraits, Allocator> const& base_str, Qubits const... qubits)
        {
          auto output_string_stream = std::basic_ostringstream<Character, CharacterTraits, Allocator>{base_str, std::ios_base::ate};
          ::ket::mpi::gate::detail::append_qubits_string_detail::insert(output_string_stream, ::ket::remove_control(qubits)...);
          return output_string_stream.str();
        }
# else // KET_PRINT_LOG
        template <typename Character, typename CharacterTraits, typename Allocator, typename... Qubits>
        inline std::basic_string<Character, CharacterTraits, Allocator>
        append_qubits_string(std::basic_string<Character, CharacterTraits, Allocator> const& base_str, Qubits const...)
        { return base_str; }
# endif // KET_PRINT_LOG
      } // namespace detail
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_DETAIL_APPEND_QUBITS_STRING_HPP
