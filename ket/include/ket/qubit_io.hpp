#ifndef KET_QUBIT_IO_HPP
# define KET_QUBIT_IO_HPP

# include <istream>
# include <ostream>

# include <ket/qubit.hpp>


namespace ket
{
  template <
    typename Character, typename CharacterTraits,
    typename StateInteger, typename BitInteger>
  inline std::basic_ostream<Character, CharacterTraits>& operator<<(
    std::basic_ostream<Character, CharacterTraits>& output_stream,
    ::ket::qubit<StateInteger, BitInteger> const qubit)
  { return output_stream << static_cast<BitInteger>(qubit); }

  template <
    typename Character, typename CharacterTraits,
    typename StateInteger, typename BitInteger>
  inline std::basic_istream<Character, CharacterTraits>& operator>>(
    std::basic_istream<Character, CharacterTraits>& input_stream,
    ::ket::qubit<StateInteger, BitInteger>& qubit)
  { return input_stream >> static_cast<BitInteger>(qubit); }
} // namespace ket


#endif // KET_QUBIT_IO_HPP
