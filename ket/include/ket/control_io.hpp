#ifndef KET_CONTROL_IO_HPP
#define KET_CONTROL_IO_HPP

# include <boost/config.hpp>

# include <istream>
# include <ostream>

# include <ket/control.hpp>


namespace ket
{
  template <
    typename Character, typename CharacterTraits, typename Qubit>
  inline std::basic_ostream<Character, CharacterTraits>& operator<<(
    std::basic_ostream<Character, CharacterTraits>& output_stream,
    ::ket::control<Qubit> const control_qubit)
  { return output_stream << control_qubit.qubit(); }

  template <
    typename Character, typename CharacterTraits, typename Qubit>
  inline std::basic_istream<Character, CharacterTraits>& operator>>(
    std::basic_ostream<Character, CharacterTraits>& input_stream,
    ::ket::control<Qubit>& control_qubit)
  { return input_stream >> control_qubit.qubit(); }
}


#endif

