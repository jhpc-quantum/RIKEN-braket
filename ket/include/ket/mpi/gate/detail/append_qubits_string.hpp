#ifndef KET_MPI_GATE_DETAIL_APPEND_QUBITS_STRING_HPP
# define KET_MPI_GATE_DETAIL_APPEND_QUBITS_STRING_HPP

# include <string>
# include <ket/qubit.hpp>
# ifdef KET_PRINT_LOG
#   include <sstream>
#   include <ket/qubit_io.hpp>
#   include <ket/control_io.hpp>
#   include <ket/runtime_control_io.hpp>
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
          inline auto insert(std::basic_ostringstream<Character, CharacterTraits, Allocator>&) -> void
          { }

          template <typename Character, typename CharacterTraits, typename Allocator, typename Qubit, typename... Qubits>
          inline auto insert(std::basic_ostringstream<Character, CharacterTraits, Allocator>& output_string_stream, Qubit const qubit, Qubits const... qubits) -> void
          {
            // Keep each operand's type so controls retain their required value.
            output_string_stream << static_cast<Character>(' ') << qubit;
            ::ket::mpi::gate::detail::append_qubits_string_detail::insert(output_string_stream, qubits...);
          }
        } // namespace append_qubits_string_detail

        template <typename Character, typename CharacterTraits, typename Allocator, typename... Qubits>
        inline auto append_qubits_string(std::basic_string<Character, CharacterTraits, Allocator> const& base_str, Qubits const... qubits)
        -> std::basic_string<Character, CharacterTraits, Allocator>
        {
          auto output_string_stream = std::basic_ostringstream<Character, CharacterTraits, Allocator>{base_str, std::ios_base::ate};
          ::ket::mpi::gate::detail::append_qubits_string_detail::insert(output_string_stream, qubits...);
          return output_string_stream.str();
        }

        namespace runtime
        {
          namespace append_qubits_string_detail
          {
            template <typename Character, typename CharacterTraits, typename Allocator, typename Qubit>
            inline auto insert(std::basic_ostringstream<Character, CharacterTraits, Allocator>& output_string_stream, Qubit const qubit) -> void
            { ::ket::mpi::gate::detail::append_qubits_string_detail::insert(output_string_stream, qubit); }
          } // namespace append_qubits_string_detail

          template <typename Character, typename CharacterTraits, typename Allocator, typename QubitsRange>
          inline auto append_qubits_string(std::basic_string<Character, CharacterTraits, Allocator> const& base_str, QubitsRange const& qubits)
          -> std::basic_string<Character, CharacterTraits, Allocator>
          {
            auto output_string_stream = std::basic_ostringstream<Character, CharacterTraits, Allocator>{base_str, std::ios_base::ate};
            for (auto const qubit: qubits)
              ::ket::mpi::gate::detail::runtime::append_qubits_string_detail::insert(output_string_stream, qubit);
            return output_string_stream.str();
          }

          template <typename Character, typename CharacterTraits, typename Allocator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
          inline auto append_qubits_string(
            std::basic_string<Character, CharacterTraits, Allocator> const& base_str,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> std::basic_string<Character, CharacterTraits, Allocator>
          {
            auto output_string_stream = std::basic_ostringstream<Character, CharacterTraits, Allocator>{base_str, std::ios_base::ate};
            ::ket::mpi::gate::detail::runtime::append_qubits_string_detail::insert(output_string_stream, target_qubit);
            for (auto const control_qubit: control_qubits)
              ::ket::mpi::gate::detail::runtime::append_qubits_string_detail::insert(output_string_stream, control_qubit);
            return output_string_stream.str();
          }

          template <typename Character, typename CharacterTraits, typename Allocator, typename QubitsRange, typename ControlQubitsRange>
          inline auto append_qubits_string(
            std::basic_string<Character, CharacterTraits, Allocator> const& base_str,
            QubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> std::basic_string<Character, CharacterTraits, Allocator>
          {
            auto output_string_stream = std::basic_ostringstream<Character, CharacterTraits, Allocator>{base_str, std::ios_base::ate};
            for (auto const target_qubit: target_qubits)
              ::ket::mpi::gate::detail::runtime::append_qubits_string_detail::insert(output_string_stream, target_qubit);
            for (auto const control_qubit: control_qubits)
              ::ket::mpi::gate::detail::runtime::append_qubits_string_detail::insert(output_string_stream, control_qubit);
            return output_string_stream.str();
          }
        } // namespace runtime
# else // KET_PRINT_LOG
        template <typename Character, typename CharacterTraits, typename Allocator, typename... Qubits>
        inline auto append_qubits_string(std::basic_string<Character, CharacterTraits, Allocator> const& base_str, Qubits const...)
        -> std::basic_string<Character, CharacterTraits, Allocator>
        { return base_str; }

        namespace runtime
        {
          template <typename Character, typename CharacterTraits, typename Allocator, typename QubitsRange>
          inline auto append_qubits_string(std::basic_string<Character, CharacterTraits, Allocator> const& base_str, QubitsRange const&)
          -> std::basic_string<Character, CharacterTraits, Allocator>
          { return base_str; }

          template <typename Character, typename CharacterTraits, typename Allocator, typename StateInteger, typename BitInteger, typename ControlQubitsRange>
          inline auto append_qubits_string(
            std::basic_string<Character, CharacterTraits, Allocator> const& base_str,
            ::ket::qubit<StateInteger, BitInteger> const, ControlQubitsRange const&)
          -> std::basic_string<Character, CharacterTraits, Allocator>
          { return base_str; }

          template <typename Character, typename CharacterTraits, typename Allocator, typename QubitsRange, typename ControlQubitsRange>
          inline auto append_qubits_string(
            std::basic_string<Character, CharacterTraits, Allocator> const& base_str,
            QubitsRange const&, ControlQubitsRange const&)
          -> std::basic_string<Character, CharacterTraits, Allocator>
          { return base_str; }
        } // namespace runtime
# endif // KET_PRINT_LOG
      } // namespace detail
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_DETAIL_APPEND_QUBITS_STRING_HPP
