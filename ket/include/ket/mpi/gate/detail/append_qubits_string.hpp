#ifndef KET_MPI_GATE_DETAIL_APPEND_QUBITS_STRING_HPP
# define KET_MPI_GATE_DETAIL_APPEND_QUBITS_STRING_HPP

# include <string>
# ifdef KET_PRINT_LOG
#   include <sstream>
#   include <utility>

#   include <boost/range/iterator_range.hpp>

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
          inline auto insert(std::basic_ostringstream<Character, CharacterTraits, Allocator>&) -> void
          { }

          template <typename CharacterTraits, typename Allocator, typename StateInteger, typename BitInteger, typename... Qubits>
          inline auto insert(std::basic_ostringstream<char, CharacterTraits, Allocator>& output_string_stream, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits) -> void
          {
            output_string_stream << ' ' << qubit;
            ::ket::mpi::gate::detail::append_qubits_string_detail::insert(output_string_stream, qubits...);
          }

          template <typename CharacterTraits, typename Allocator, typename StateInteger, typename BitInteger, typename... Qubits>
          inline auto insert(std::basic_ostringstream<wchar_t, CharacterTraits, Allocator>& output_string_stream, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits) -> void
          {
            output_string_stream << L' ' << qubit;
            ::ket::mpi::gate::detail::append_qubits_string_detail::insert(output_string_stream, qubits...);
          }

          template <typename CharacterTraits, typename Allocator, typename StateInteger, typename BitInteger, typename... Qubits>
          inline auto insert(std::basic_ostringstream<char16_t, CharacterTraits, Allocator>& output_string_stream, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits) -> void
          {
            output_string_stream << u' ' << qubit;
            ::ket::mpi::gate::detail::append_qubits_string_detail::insert(output_string_stream, qubits...);
          }

          template <typename CharacterTraits, typename Allocator, typename StateInteger, typename BitInteger, typename... Qubits>
          inline auto insert(std::basic_ostringstream<char32_t, CharacterTraits, Allocator>& output_string_stream, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits) -> void
          {
            output_string_stream << U' ' << qubit;
            ::ket::mpi::gate::detail::append_qubits_string_detail::insert(output_string_stream, qubits...);
          }
        } // namespace append_qubits_string_detail

        template <typename Character, typename CharacterTraits, typename Allocator, typename... Qubits>
        inline auto append_qubits_string(std::basic_string<Character, CharacterTraits, Allocator> const& base_str, Qubits const... qubits)
        -> std::basic_string<Character, CharacterTraits, Allocator>
        {
          auto output_string_stream = std::basic_ostringstream<Character, CharacterTraits, Allocator>{base_str, std::ios_base::ate};
          ::ket::mpi::gate::detail::append_qubits_string_detail::insert(output_string_stream, ::ket::remove_control(qubits)...);
          return output_string_stream.str();
        }

        namespace runtime
        {
          namespace append_qubits_string_detail
          {
            template <typename CharacterTraits, typename Allocator, typename StateInteger, typename BitInteger>
            inline auto insert(std::basic_ostringstream<char, CharacterTraits, Allocator>& output_string_stream, ::ket::qubit<StateInteger, BitInteger> const qubit) -> void
            { output_string_stream << ' ' << qubit; }

            template <typename CharacterTraits, typename Allocator, typename StateInteger, typename BitInteger, typename... Qubits>
            inline auto insert(std::basic_ostringstream<wchar_t, CharacterTraits, Allocator>& output_string_stream, ::ket::qubit<StateInteger, BitInteger> const qubit) -> void
            { output_string_stream << L' ' << qubit; }

            template <typename CharacterTraits, typename Allocator, typename StateInteger, typename BitInteger, typename... Qubits>
            inline auto insert(std::basic_ostringstream<char16_t, CharacterTraits, Allocator>& output_string_stream, ::ket::qubit<StateInteger, BitInteger> const qubit) -> void
            { output_string_stream << u' ' << qubit; }

            template <typename CharacterTraits, typename Allocator, typename StateInteger, typename BitInteger, typename... Qubits>
            inline auto insert(std::basic_ostringstream<char32_t, CharacterTraits, Allocator>& output_string_stream, ::ket::qubit<StateInteger, BitInteger> const qubit) -> void
            { output_string_stream << U' ' << qubit; }
          } // namespace append_qubits_string_detail

          template <typename Character, typename CharacterTraits, typename Allocator, typename QubitsRange>
          inline auto append_qubits_string(std::basic_string<Character, CharacterTraits, Allocator> const& base_str, QubitsRange const& qubits)
          -> std::basic_string<Character, CharacterTraits, Allocator>
          {
            auto output_string_stream = std::basic_ostringstream<Character, CharacterTraits, Allocator>{base_str, std::ios_base::ate};
            for (auto const qubit: qubits)
              ::ket::mpi::gate::detail::runtime::append_qubits_string_detail::insert(output_string_stream, ::ket::remove_control(qubit));
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
              ::ket::mpi::gate::detail::runtime::append_qubits_string_detail::insert(output_string_stream, ::ket::remove_control(control_qubit));
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
              ::ket::mpi::gate::detail::runtime::append_qubits_string_detail::insert(output_string_stream, ::ket::remove_control(target_qubit));
            for (auto const control_qubit: control_qubits)
              ::ket::mpi::gate::detail::runtime::append_qubits_string_detail::insert(output_string_stream, ::ket::remove_control(control_qubit));
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
