#ifndef KET_MPI_PRINT_AMPLITUDES_HPP
# define KET_MPI_PRINT_AMPLITUDES_HPP

# include <ostream>
# include <string>
# include <iterator>
# include <utility>

# include <yampi/environment.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/send.hpp>
# include <yampi/receive.hpp>

# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>


namespace ket
{
  namespace mpi
  {
    template <
      typename MpiPolicy, typename Character, typename CharacterTraits, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename Formatter>
    inline auto print_amplitudes(
      MpiPolicy const& mpi_policy,
      std::basic_ostream<Character, CharacterTraits>& output_stream,
      RandomAccessRange const& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Formatter&& formatter, std::basic_string<Character, CharacterTraits> const& separator = " ")
    -> void
    {
      using std::begin;
      auto const first = begin(local_state);
      auto const present_rank = communicator.rank(environment);

      auto const num_qubits = static_cast<BitInteger>(::ket::mpi::utility::policy::num_qubits(mpi_policy, local_state, communicator, environment));
      auto const last_qubit_value = ::ket::utility::integer_exp2<StateInteger>(num_qubits);
      auto is_first_output = true;
      for (auto qubit_value = StateInteger{0u}; qubit_value < last_qubit_value; ++qubit_value)
      {
        auto const rank_index
          = ::ket::mpi::utility::qubit_value_to_rank_index(
              mpi_policy, local_state, ::ket::mpi::permutate_bits(permutation, qubit_value), communicator, environment);

        if (present_rank == root)
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          auto amplitude = complex_type{};

          if (present_rank == rank_index.first)
            amplitude = *(first + rank_index.second);
          else
            yampi::receive(yampi::ignore_status, yampi::make_buffer(amplitude), rank_index.first, yampi::tag{static_cast<int>(rank_index.second)}, communicator, environment);

          if (is_first_output)
          {
            output_stream << formatter(qubit_value, amplitude);
            is_first_output = false;
          }
          else
            output_stream << separator << formatter(qubit_value, amplitude);
        }
        else if (present_rank == rank_index.first)
          yampi::send(yampi::make_buffer(*(first + rank_index.second)), root, yampi::tag{static_cast<int>(rank_index.second)}, communicator, environment);
      }
    }

    template <
      typename MpiPolicy, typename Character, typename CharacterTraits, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename DerivedDatatype,
      typename Formatter>
    inline auto print_amplitudes(
      MpiPolicy const& mpi_policy,
      std::basic_ostream<Character, CharacterTraits>& output_stream,
      RandomAccessRange const& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Formatter&& formatter, std::basic_string<Character, CharacterTraits> const& separator = " ")
    -> void
    {
      using std::begin;
      auto const first = begin(local_state);
      auto const present_rank = communicator.rank(environment);

      auto const num_qubits = static_cast<BitInteger>(::ket::mpi::utility::policy::num_qubits(mpi_policy, local_state, communicator, environment));
      auto const last_qubit_value = ::ket::utility::integer_exp2<StateInteger>(num_qubits);
      auto is_first_output = true;
      for (auto qubit_value = StateInteger{0u}; qubit_value < last_qubit_value; ++qubit_value)
      {
        auto const rank_index
          = ::ket::mpi::utility::qubit_value_to_rank_index(
              mpi_policy, local_state, ::ket::mpi::permutate_bits(permutation, qubit_value), communicator, environment);

        if (present_rank == root)
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          auto amplitude = complex_type{};

          if (present_rank == rank_index.first)
            amplitude = *(first + rank_index.second);
          else
            yampi::receive(yampi::ignore_status, yampi::make_buffer(amplitude, datatype), rank_index.first, yampi::tag{static_cast<int>(rank_index.second)}, communicator, environment);

          if (is_first_output)
          {
            output_stream << formatter(qubit_value, amplitude);
            is_first_output = false;
          }
          else
            output_stream << separator << formatter(qubit_value, amplitude);
        }
        else if (present_rank == rank_index.first)
          yampi::send(yampi::make_buffer(*(first + rank_index.second), datatype), root, yampi::tag{static_cast<int>(rank_index.second)}, communicator, environment);
      }
    }

    template <
      typename MpiPolicy, typename Character, typename CharacterTraits, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename Formatter>
    inline auto print_amplitudes(
      std::basic_ostream<Character, CharacterTraits>& output_stream,
      RandomAccessRange const& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Formatter&& formatter, std::basic_string<Character, CharacterTraits> const& separator = " ")
    -> void
    {
      ::ket::mpi::print_amplitudes(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        output_stream, local_state, permutation, root, communicator, environment,
        std::forward<Formatter>(formatter), separator);
    }

    template <
      typename MpiPolicy, typename Character, typename CharacterTraits, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename DerivedDatatype,
      typename Formatter>
    inline auto print_amplitudes(
      std::basic_ostream<Character, CharacterTraits>& output_stream,
      RandomAccessRange const& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Formatter&& formatter, std::basic_string<Character, CharacterTraits> const& separator = " ")
    -> void
    {
      ::ket::mpi::print_amplitudes(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        output_stream, local_state, permutation, root, communicator, environment,
        std::forward<Formatter>(formatter), separator);
    }

    template <
      typename MpiPolicy, typename Character, typename CharacterTraits, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename Formatter>
    inline auto println_amplitudes(
      MpiPolicy const& mpi_policy,
      std::basic_ostream<Character, CharacterTraits>& output_stream,
      RandomAccessRange const& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Formatter&& formatter, std::basic_string<Character, CharacterTraits> const& separator = " ")
    -> void
    {
      ::ket::mpi::print_amplitudes(
        mpi_policy, output_stream, local_state, permutation, root, communicator, environment,
        std::forward<Formatter>(formatter), separator);
      output_stream << '\n';
    }

    template <
      typename MpiPolicy, typename Character, typename CharacterTraits, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename DerivedDatatype,
      typename Formatter>
    inline auto println_amplitudes(
      MpiPolicy const& mpi_policy,
      std::basic_ostream<Character, CharacterTraits>& output_stream,
      RandomAccessRange const& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Formatter&& formatter, std::basic_string<Character, CharacterTraits> const& separator = " ")
    -> void
    {
      ::ket::mpi::print_amplitudes(
        mpi_policy, output_stream, local_state, permutation, root, communicator, environment,
        std::forward<Formatter>(formatter), separator);
      output_stream << '\n';
    }

    template <
      typename Character, typename CharacterTraits, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename Formatter>
    inline auto println_amplitudes(
      std::basic_ostream<Character, CharacterTraits>& output_stream,
      RandomAccessRange const& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Formatter&& formatter, std::basic_string<Character, CharacterTraits> const& separator = " ")
    -> void
    {
      ::ket::mpi::println_amplitudes(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        output_stream, local_state, permutation, root, communicator, environment,
        std::forward<Formatter>(formatter), separator);
    }

    template <
      typename Character, typename CharacterTraits, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename DerivedDatatype,
      typename Formatter>
    inline auto println_amplitudes(
      std::basic_ostream<Character, CharacterTraits>& output_stream,
      RandomAccessRange const& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Formatter&& formatter, std::basic_string<Character, CharacterTraits> const& separator = " ")
    -> void
    {
      ::ket::mpi::println_amplitudes(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        output_stream, local_state, permutation, root, communicator, environment,
        std::forward<Formatter>(formatter), separator);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_PRINT_AMPLITUDES_HPP
