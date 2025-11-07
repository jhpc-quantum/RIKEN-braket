#ifndef KET_PRINT_AMPLITUDES_HPP
# define KET_PRINT_AMPLITUDES_HPP

# include <ostream>
# include <string>
# include <iterator>
# include <utility>


namespace ket
{
  template <typename Character, typename CharacterTraits, typename RandomAccessRange, typename Formatter>
  inline auto print_amplitudes(
    std::basic_ostream<Character, CharacterTraits>& output_stream,
    RandomAccessRange const& state, Formatter&& formatter, std::basic_string<Character, CharacterTraits> const& separator = " ")
  -> void
  {
    using std::begin;
    using std::end;
    auto const first = begin(state);
    auto const last = end(state);

    if (first == last)
      return;

    output_stream << formatter(0, *first);

    for (auto iter = std::next(first); iter != last; ++iter)
      output_stream << separator << formatter(iter - first, *iter);
  }

  template <typename Character, typename CharacterTraits, typename RandomAccessRange, typename Formatter>
  inline auto println_amplitudes(
    std::basic_ostream<Character, CharacterTraits>& output_stream,
    RandomAccessRange const& state, Formatter&& formatter, std::basic_string<Character, CharacterTraits> const& separator = " ")
  -> void
  { ::ket::print_amplitudes(output_stream, state, std::forward<Formatter>(formatter), separator); output_stream << '\n'; }
} // namespace ket


#endif // KET_PRINT_AMPLITUDES_HPP
