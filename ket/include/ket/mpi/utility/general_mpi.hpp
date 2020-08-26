#ifndef KET_MPI_GENERAL_MPI_HPP
# define KET_MPI_GENERAL_MPI_HPP

# include <cstddef>
# include <cassert>
# include <iostream>
# include <vector>
# include <algorithm>
# include <numeric>
# include <iterator>
# include <utility>
# include <array>
# include <type_traits>

# include <boost/range/value_type.hpp>
# include <boost/range/size.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# ifndef NDEBUG
#   include <yampi/lowest_io_process.hpp>
# endif

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/const_iterator_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/detail/swap_local_qubits.hpp>
# include <ket/mpi/utility/detail/interchange_qubits.hpp>
# include <ket/mpi/utility/logger.hpp>
# ifndef NDEBUG
#   include <ket/qubit_io.hpp>
#   include <ket/mpi/qubit_permutation_io.hpp>
# endif


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace policy
      {
        class general_mpi
        {
         public:
          explicit general_mpi() noexcept { }
        }; // class general_mpi

        inline general_mpi make_general_mpi() noexcept { return general_mpi();  }

        namespace meta
        {
          template <typename T>
          struct is_mpi_policy
            : std::false_type
          { }; // struct is_mpi_policy<T>

          template <>
          struct is_mpi_policy< ::ket::mpi::utility::policy::general_mpi >
            : std::true_type
          { }; // struct is_mpi_policy< ::ket::mpi::utility::policy::general_mpi >
        } // namespace meta
      } // namespace policy

      namespace general_mpi_detail
      {
        template <
          typename ParallelPolicy, typename LocalState,
          typename StateInteger, typename BitInteger, typename Allocator,
          typename UnswappableQubits>
        inline ::ket::qubit<StateInteger, BitInteger>
        make_local_swap_qubit_swappable(
          ParallelPolicy const parallel_policy,
          LocalState& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          UnswappableQubits const& unswappable_qubits,
          ::ket::qubit<StateInteger, BitInteger> const permutated_local_swap_qubit)
        {
          using qubit_type = ket::qubit<StateInteger, BitInteger>;
          static_assert(
            (std::is_same<
               typename boost::range_value<UnswappableQubits>::type,
               qubit_type>::value),
            "value_type of UnswappableQubits must be the same to qubit_type");

          using ::ket::mpi::inverse;
          auto const local_swap_qubit = inverse(permutation)[permutated_local_swap_qubit];

          auto const last = ::ket::utility::end(unswappable_qubits);

          if (std::find(::ket::utility::begin(unswappable_qubits), last, local_swap_qubit) != last)
          {
            auto permutated_other_qubit = permutated_local_swap_qubit;
            auto other_qubit = qubit_type{};
            do
            {
              --permutated_other_qubit;
              using ::ket::mpi::inverse;
              other_qubit = inverse(permutation)[permutated_other_qubit];
            }
            while (std::find(::ket::utility::begin(unswappable_qubits), last, other_qubit) != last);

            ::ket::mpi::utility::detail::swap_local_qubits(
              parallel_policy, local_state,
              permutated_local_swap_qubit, permutated_other_qubit);
            using ::ket::mpi::permutate;
            permutate(permutation, local_swap_qubit, other_qubit);
          }

          return inverse(permutation)[permutated_local_swap_qubit];
        }
      } // namespace general_mpi_detail

      namespace dispatch
      {
        template <std::size_t num_qubits_of_operation, typename MpiPolicy>
        struct maybe_interchange_qubits;

        template <std::size_t num_qubits_of_operation>
        struct maybe_interchange_qubits<num_qubits_of_operation, ::ket::mpi::utility::policy::general_mpi>
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            static_assert(
              std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);
            assert(
              ::ket::utility::integer_exp2<StateInteger>(
                ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)))
              == boost::size(local_state));

            auto const num_local_qubits
              = ::ket::utility::integer_log2<BitInteger>(boost::size(local_state));

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto permutated_global_swap_qubits = std::array<qubit_type, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutated_global_swap_qubits[index] = permutation[qubits[index]];

            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              if (static_cast<BitInteger>(permutated_global_swap_qubits[index]) < num_local_qubits)
              {
                call_lower_maybe_interchange_qubits(
                  index,
                  parallel_policy, local_state, qubits, unswappable_qubits,
                  permutation, buffer, communicator, environment);
                return;
              }
            }

            do_call(
              parallel_policy, local_state, num_local_qubits,
              permutated_global_swap_qubits,
              qubits, unswappable_qubits, permutation, communicator, environment,
              [&buffer](
                LocalState& local_state,
                StateInteger const source_local_first_index, StateInteger const source_local_last_index,
                StateInteger const target_global_index,
                yampi::communicator const& communicator, yampi::environment const& environment)
              {
                ::ket::mpi::utility::detail::interchange_qubits(
                  local_state, buffer, source_local_first_index, source_local_last_index,
                  static_cast<yampi::rank>(target_global_index), communicator, environment);
              });
          }

          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static void call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            static_assert(
              std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);
            assert(
              ::ket::utility::integer_exp2<StateInteger>(
                ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)))
              == boost::size(local_state));

            auto const num_local_qubits
              = ::ket::utility::integer_log2<BitInteger>(boost::size(local_state));

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto permutated_global_swap_qubits = std::array<qubit_type, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutated_global_swap_qubits[index] = permutation[qubits[index]];

            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              if (static_cast<BitInteger>(permutated_global_swap_qubits[index]) < num_local_qubits)
              {
                call_lower_maybe_interchange_qubits(
                  index,
                  parallel_policy, local_state, qubits, unswappable_qubits,
                  permutation, buffer, datatype, communicator, environment);
                return;
              }
            }

            do_call(
              parallel_policy, local_state, num_local_qubits,
              permutated_global_swap_qubits,
              qubits, unswappable_qubits, permutation, communicator, environment,
              [&buffer, &datatype](
                LocalState& local_state,
                StateInteger const source_local_first_index, StateInteger const source_local_last_index,
                StateInteger const target_global_index,
                yampi::communicator const& communicator, yampi::environment const& environment)
              {
                ::ket::mpi::utility::detail::interchange_qubits(
                  local_state, buffer, source_local_first_index, source_local_last_index,
                  datatype, static_cast<yampi::rank>(target_global_index),
                  communicator, environment);
              });
          }

         private:
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename Function>
          static void do_call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            BitInteger const num_local_qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const&
              permutated_global_swap_qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator,
            yampi::environment const& environment,
            Function&& interchange_qubits)
          {
            auto output_string_stream = std::ostringstream{"interchange_qubits<", std::ios_base::ate};
            output_string_stream << num_qubits_of_operation << '>';
            ::ket::mpi::utility::log_with_time_guard<char> print{output_string_stream.str(), environment};

# ifndef NDEBUG
            auto const maybe_io_rank = yampi::lowest_io_process(environment);
            auto const my_rank = yampi::communicator(yampi::world_communicator_t()).rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation before changing qubits] " << permutation << std::endl;
# endif // NDEBUG

            // (ex.: num_qubits_of_operation == 3)
            //  Swaps between xxbxb'xb''xx|cc'c''xxxxxxxx and
            // xxcxc'xx|bb'xxxxxxxx (c = b or ~b). Upper qubits are global
            // qubits representing MPI rank. Lower qubits are local qubits
            // representing memory address. The first two upper qubits in the
            // local qubits are "local swap qubits". Two consecutive bits in
            // global qubits and the "local swap qubits" would be swapped.

            using qubit_type = ket::qubit<StateInteger, BitInteger>;
            auto permutated_local_swap_qubits = std::array<qubit_type, num_qubits_of_operation>{};
            auto local_swap_qubits = std::array<qubit_type, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              permutated_local_swap_qubits[index]
                = qubit_type{num_local_qubits - BitInteger{1u} - static_cast<BitInteger>(index)};
              local_swap_qubits[index]
                = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                    parallel_policy, local_state, permutation,
                    unswappable_qubits, permutated_local_swap_qubits[index]);
            }

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local swap qubits] " << permutation << std::endl;
# endif // NDEBUG

            // xxbxb'xb''xx(|xxxxxxxxxx)
            auto const source_global_index
              = static_cast<StateInteger>(communicator.rank(environment).mpi_rank());

            for (auto target_global_mask = StateInteger{1u};
                 target_global_mask < ::ket::utility::integer_exp2<StateInteger>(num_qubits_of_operation);
                 ++target_global_mask)
            {
              auto target_global_masks = std::array<StateInteger, num_qubits_of_operation>{};
              for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                target_global_masks[index] = (target_global_mask bitand (StateInteger{1u} << index)) >> index;

              // xxcxc'xc''xx(|xxxxxxxxxx) (c = b or ~b, except for (c, c', c'') = (b, b', b''))
              auto mask = (target_global_masks[0u] << permutated_global_swap_qubits[0u]) >> num_local_qubits;
              for (auto index = std::size_t{1u}; index < num_qubits_of_operation; ++index)
                mask |= (target_global_masks[index] << permutated_global_swap_qubits[index]) >> num_local_qubits;
              auto const target_global_index = source_global_index xor mask;

              // (0000000|)c0000000000, (0000000|)0c'000000000, (0000000|)00c''00000000
              auto source_local_first_indices = std::array<StateInteger, num_qubits_of_operation>{};
              for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                source_local_first_indices[index]
                  = ((target_global_index << num_local_qubits)
                     bitand ::ket::utility::integer_exp2<StateInteger>(permutated_global_swap_qubits[index]))
                    >> (permutated_global_swap_qubits[index] - permutated_local_swap_qubits[index]);
              // (0000000|)cc'c''00000000
              auto source_local_first_index = source_local_first_indices[0u];
              for (auto index = std::size_t{1u}; index < num_qubits_of_operation; ++index)
                source_local_first_index |= source_local_first_indices[index];
              // (0000000|)0001111111
              auto const prev_last_mask
                = (StateInteger{1u} << permutated_local_swap_qubits.back()) - StateInteger{1u};
              // (0000000|)cc'c''11111111 + 1
              auto const source_local_last_index
                = (source_local_first_index bitor prev_last_mask) + StateInteger{1u};

              auto oss_swap = std::ostringstream{"interchange_qubits<", std::ios_base::ate};
              oss_swap << num_qubits_of_operation << ">::swap";
              ::ket::mpi::utility::log_with_time_guard<char> print{oss_swap.str(), environment};

              interchange_qubits(
                local_state,
                source_local_first_index, source_local_last_index, target_global_index,
                communicator, environment);
            }

            using ::ket::mpi::permutate;
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutate(permutation, qubits[index], local_swap_qubits[index]);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local/global qubits] " << permutation << std::endl;
# endif // NDEBUG
          }

          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call_lower_maybe_interchange_qubits(
            std::size_t const new_unswappable_qubit_index,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            assert(new_unswappable_qubit_index < num_qubits_of_operation);

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto new_qubits = std::array<qubit_type, num_qubits_of_operation - 1u>{};
            std::copy(
              ::ket::utility::begin(qubits) + new_unswappable_qubit_index + 1u, ::ket::utility::end(qubits),
              std::copy(
                ::ket::utility::begin(qubits), ::ket::utility::begin(qubits) + new_unswappable_qubit_index,
                ::ket::utility::begin(new_qubits)));

            auto new_unswappable_qubits = std::array<qubit_type, num_unswappable_qubits + 1u>{};
            std::copy(
              ::ket::utility::begin(unswappable_qubits), ::ket::utility::end(unswappable_qubits),
              ::ket::utility::begin(new_unswappable_qubits));
            new_unswappable_qubits.back() = qubits[new_unswappable_qubit_index];

            using lower_maybe_interchange_qubits
              = ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  num_qubits_of_operation - 1u, ::ket::mpi::utility::policy::general_mpi>;
            lower_maybe_interchange_qubits::call(
              parallel_policy, local_state, new_qubits, new_unswappable_qubits,
              permutation, buffer, communicator, environment);
          }

          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static void call_lower_maybe_interchange_qubits(
            std::size_t const new_unswappable_qubit_index,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            assert(new_unswappable_qubit_index < num_qubits_of_operation);

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto new_qubits = std::array<qubit_type, num_qubits_of_operation - 1u>{};
            std::copy(
              ::ket::utility::begin(qubits) + new_unswappable_qubit_index + 1u, ::ket::utility::end(qubits),
              std::copy(
                ::ket::utility::begin(qubits), ::ket::utility::begin(qubits) + new_unswappable_qubit_index,
                ::ket::utility::begin(new_qubits)));

            auto new_unswappable_qubits = std::array<qubit_type, num_unswappable_qubits + 1u>{};
            std::copy(
              ::ket::utility::begin(unswappable_qubits), ::ket::utility::end(unswappable_qubits),
              ::ket::utility::begin(new_unswappable_qubits));
            new_unswappable_qubits.back() = qubits[new_unswappable_qubit_index];

            using lower_maybe_interchange_qubits
              = ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  num_qubits_of_operation - 1u, ::ket::mpi::utility::policy::general_mpi>;
            lower_maybe_interchange_qubits::call(
              parallel_policy, local_state, new_qubits, new_unswappable_qubits,
              permutation, buffer, datatype, communicator, environment);
          }
        }; // struct maybe_interchange_qubits<num_qubits_of_operation, ::ket::mpi::utility::policy::general_mpi>

        template <>
        struct maybe_interchange_qubits<0u, ::ket::mpi::utility::policy::general_mpi>
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call(
            ParallelPolicy const, LocalState&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, 0u > const&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>&,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>&,
            yampi::communicator const&, yampi::environment const&)
          { }

          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static void call(
            ParallelPolicy const, LocalState&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, 0u > const&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>&,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>&,
            yampi::datatype_base<DerivedDatatype> const&,
            yampi::communicator const&, yampi::environment const&)
          { }
        }; // struct maybe_interchange_qubits<0u, ::ket::mpi::utility::policy::general_mpi>

        template <typename MpiPolicy, typename LocalState_>
        struct for_each_local_range
        {
          template <typename LocalState, typename Function>
          static LocalState& call(LocalState& local_state, Function&& function);

          template <typename LocalState, typename Function>
          static LocalState const& call(LocalState const& local_state, Function&& function);
        }; // struct for_each_local_range<MpiPolicy, LocalState_>

        template <typename LocalState_>
        struct for_each_local_range< ::ket::mpi::utility::policy::general_mpi, LocalState_ >
        {
          template <typename LocalState, typename Function>
          static LocalState& call(LocalState& local_state, Function&& function)
          {
            std::forward<Function>(function)(
              ::ket::utility::begin(local_state), ::ket::utility::end(local_state));
            return local_state;
          }

          template <typename LocalState, typename Function>
          static LocalState& call(LocalState const& local_state, Function&& function)
          {
            std::forward<Function>(function)(
              ::ket::utility::begin(local_state), ::ket::utility::end(local_state));
            return local_state;
          }
        }; // struct for_each_local_range< ::ket::mpi::utility::policy::general_mpi. LocalState_ >

        template <typename MpiPolicy>
        struct rank_index_to_qubit_value
        {
          template <typename LocalState, typename StateInteger>
          static StateInteger call(
            LocalState const& local_state,
            yampi::rank const rank, StateInteger const index);
        }; // struct rank_index_to_qubit_value<MpiPolicy>

        template <>
        struct rank_index_to_qubit_value< ::ket::mpi::utility::policy::general_mpi >
        {
          template <typename LocalState, typename StateInteger>
          static StateInteger call(
            LocalState const& local_state,
            yampi::rank const rank, StateInteger const index)
          { return rank.mpi_rank() * boost::size(local_state) + index; }
        }; // struct rank_index_to_qubit_value< ::ket::mpi::utility::policy::general_mpi >

        template <typename MpiPolicy>
        struct qubit_value_to_rank_index
        {
          template <typename LocalState, typename StateInteger>
          static std::pair<yampi::rank, StateInteger> call(
            LocalState const& local_state, StateInteger const qubit_value);
        }; // struct qubit_value_to_rank_index<MpiPolicy>

        template <>
        struct qubit_value_to_rank_index< ::ket::mpi::utility::policy::general_mpi >
        {
          template <typename LocalState, typename StateInteger>
          static std::pair<yampi::rank, StateInteger> call(
            LocalState const& local_state, StateInteger const qubit_value)
          {
            return std::make_pair(
              static_cast<yampi::rank>(qubit_value / boost::size(local_state)),
              qubit_value % boost::size(local_state));
          }
        }; // struct qubit_value_to_rank_index< ::ket::mpi::utility::policy::general_mpi >

# ifdef KET_USE_DIAGONAL_LOOP
        // 170607-: 
        template <typename MpiPolicy>
        struct diagonal_loop
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1, typename... ControlQubits>
          static void call(
            ParallelPolicy const, LocalState const& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::communicator const& communicator,
            yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            ControlQubits... control_qubits);
        }; // struct diagonal_loop<MpiPolicy>

        template <>
        struct diagonal_loop< ::ket::mpi::utility::policy::general_mpi >
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1, typename... ControlQubits>
          static void call(
            ParallelPolicy const parallel_policy, LocalState const& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::communicator const& communicator,
            yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            ControlQubits... control_qubits)
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto local_permutated_control_qubits = std::array<qubit_type, 0u>{};

            auto const least_global_permutated_qubit
              = qubit_type{::ket::utility::integer_log2<BitInteger>(boost::size(local_state))};

            call_impl(
              parallel_policy, permutation, communicator.rank(environment),
              least_global_permutated_qubit, target_qubit,
              std::forward<Function0>(function0),
              std::forward<Function1>(function1),
              local_permutated_control_qubits, control_qubits...);
          }

         private:
          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1,
            std::size_t num_local_control_qubits, typename... ControlQubits>
          static void call_impl(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const rank,
            ::ket::qubit<StateInteger, BitInteger> const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>,
              num_local_control_qubits> const& local_permutated_control_qubits,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ControlQubits... control_qubits)
          {
            auto const permutated_control_qubit = permutation[control_qubit.qubit()];

            if (permutated_control_qubit < least_global_permutated_qubit)
            {
              using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
              auto new_local_permutated_control_qubits
                = std::array<qubit_type, num_local_control_qubits + 1u>{};
              std::copy(
                ::ket::utility::begin(local_permutated_control_qubits),
                ::ket::utility::end(local_permutated_control_qubits),
                ::ket::utility::begin(new_local_permutated_control_qubits));
              new_local_permutated_control_qubits.back() = permutated_control_qubit;

              call_impl(
                parallel_policy, permutation, rank,
                least_global_permutated_qubit, target_qubit,
                std::forward<Function0>(function0),
                std::forward<Function1>(function1),
                new_local_permutated_control_qubits, control_qubits...);
            }
            else
            {
              static constexpr auto zero_state_integer = StateInteger{0u};
              static constexpr auto one_state_integer = StateInteger{1u};

              auto const mask
                = one_state_integer << (permutated_control_qubit - least_global_permutated_qubit);

              if ((static_cast<StateInteger>(rank.mpi_rank()) bitand mask) != zero_state_integer)
                call_impl(
                  parallel_policy, permutation, rank,
                  least_global_permutated_qubit, target_qubit,
                  std::forward<Function0>(function0),
                  std::forward<Function1>(function1),
                  local_permutated_control_qubits, control_qubits...);
            }
          }

          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1,
            std::size_t num_local_control_qubits>
          static void call_impl(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const rank,
            ::ket::qubit<StateInteger, BitInteger> const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>,
              num_local_control_qubits> const& local_permutated_control_qubits)
          {
            auto const permutated_target_qubit = permutation[target_qubit];

            static constexpr auto one_state_integer = StateInteger{1u};

            auto const last_integer
              = (one_state_integer << least_global_permutated_qubit)
                >> boost::size(local_permutated_control_qubits);

            if (permutated_target_qubit < least_global_permutated_qubit)
            {
              auto const mask = one_state_integer << permutated_target_qubit;

              for_each(
                parallel_policy, last_integer, local_permutated_control_qubits,
                [&function0, &function1, mask](StateInteger const state_integer)
                {
                  static constexpr auto zero_state_integer = StateInteger{0u};

                  if ((state_integer bitand mask) == zero_state_integer)
                    function0(state_integer);
                  else
                    function1(state_integer);
                });
            }
            else
            {
              auto const mask
                = one_state_integer << (permutated_target_qubit - least_global_permutated_qubit);

              static constexpr auto zero_state_integer = StateInteger{0u};

              if ((static_cast<StateInteger>(rank.mpi_rank()) bitand mask) == zero_state_integer)
                for_each(
                  parallel_policy, last_integer, local_permutated_control_qubits,
                  std::forward<Function0>(function0));
              else
                for_each(
                  parallel_policy, last_integer, local_permutated_control_qubits,
                  std::forward<Function1>(function1));
            }
          }

          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger,
            std::size_t num_local_control_qubits, typename Function>
          static void for_each(
            ParallelPolicy const parallel_policy, StateInteger const last_integer,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>,
              num_local_control_qubits> const& local_permutated_control_qubits,
            Function&& function)
          {
            auto sorted_local_permutated_control_qubits = local_permutated_control_qubits;
            std::sort(
              ::ket::utility::begin(sorted_local_permutated_control_qubits),
              ::ket::utility::end(sorted_local_permutated_control_qubits));

            for_each_impl(
              parallel_policy, last_integer, sorted_local_permutated_control_qubits,
              std::forward<Function>(function));
          }

          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger,
            std::size_t num_local_control_qubits, typename Function>
          static void for_each_impl(
            ParallelPolicy const parallel_policy, StateInteger const last_integer,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>,
              num_local_control_qubits> const& sorted_local_permutated_control_qubits,
            Function&& function)
          {
            static constexpr auto zero_state_integer = StateInteger{0u};

            // 000101000100
            auto const mask
              = std::accumulate(
                  ::ket::utility::begin(sorted_local_permutated_control_qubits),
                  ::ket::utility::end(sorted_local_permutated_control_qubits),
                  zero_state_integer,
                  [](StateInteger const& partial_mask, qubit_type const& control_qubit)
                  {
                    static constexpr auto one_state_integer = StateInteger{1u};
                    return partial_mask bitor (one_state_integer << control_qubit);
                  });

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy, last_integer,
              [&function, &sorted_local_permutated_control_qubits, mask](StateInteger state_integer, int const)
              {
                static constexpr auto one_state_integer = StateInteger{1u};

                // xxx0x0xxx0xx
                for (qubit_type const& qubit: sorted_local_permutated_control_qubits)
                {
                  auto const lower_mask = (one_state_integer << qubit) - one_state_integer;
                  auto const upper_mask = compl lower_mask;
                  state_integer
                    = (state_integer bitand lower_mask)
                      bitor ((state_integer bitand upper_mask) << 1u);
                }

                // function(xxx1x1xxx1xx)
                function(state_integer bitor mask);
              });
          }
        }; // struct diagonal_loop< ::ket::mpi::utility::policy::general_mpi >
# endif // KET_USE_DIAGONAL_LOOP
      } // namespace dispatch

      template <
        typename MpiPolicy, typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator>
      void maybe_interchange_qubits(
        MpiPolicy const, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::array<
          ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation> const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        using maybe_interchange_qubits_impl
          = ::ket::mpi::utility::dispatch::maybe_interchange_qubits<num_qubits_of_operation, MpiPolicy>;
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;

        auto unswappable_qubits = std::array<qubit_type, 0u>{};

        maybe_interchange_qubits_impl::call(
          parallel_policy,
          local_state, qubits, unswappable_qubits, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      void maybe_interchange_qubits(
        MpiPolicy const, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::array<
          ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation> const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        using maybe_interchange_qubits_impl
          = ::ket::mpi::utility::dispatch::maybe_interchange_qubits<num_qubits_of_operation, MpiPolicy>;
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;

        auto unswappable_qubits = std::array<qubit_type, 0u>{};

        maybe_interchange_qubits_impl::call(
          parallel_policy,
          local_state, qubits, unswappable_qubits, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator>
      void maybe_interchange_qubits(
        ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::array<
          ::ket::qubit<StateInteger, BitInteger>,
          num_qubits_of_operation> const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, qubits, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      void maybe_interchange_qubits(
        ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::array<
          ::ket::qubit<StateInteger, BitInteger>,
          num_qubits_of_operation> const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, qubits, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator>
      void maybe_interchange_qubits(
        LocalState& local_state,
        std::array<
          ket::qubit<StateInteger, BitInteger>,
          num_qubits_of_operation> const& qubits,
        ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubits, permutation, buffer, communicator, environment);
      }

      template <
        typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      void maybe_interchange_qubits(
        LocalState& local_state,
        std::array<
          ket::qubit<StateInteger, BitInteger>,
          num_qubits_of_operation> const& qubits,
        ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubits, permutation, buffer, datatype, communicator, environment);
      }

      template <typename MpiPolicy, typename LocalState, typename Function>
      inline LocalState& for_each_local_range(
        MpiPolicy const, LocalState& local_state, Function&& function)
      {
        return ::ket::mpi::utility::dispatch::for_each_local_range<MpiPolicy, LocalState>::call(
          local_state, std::forward<Function>(function));
      }

      template <typename MpiPolicy, typename LocalState, typename Function>
      inline LocalState const& for_each_local_range(
        MpiPolicy const, LocalState const& local_state, Function&& function)
      {
        return ::ket::mpi::utility::dispatch::for_each_local_range<MpiPolicy, LocalState>::call(
          local_state, std::forward<Function>(function));
      }

      template <
        typename MpiPolicy, typename LocalState, typename StateInteger>
      inline StateInteger rank_index_to_qubit_value(
        MpiPolicy const, LocalState const& local_state, yampi::rank const rank,
        StateInteger const index)
      {
        return ::ket::mpi::utility::dispatch::rank_index_to_qubit_value<MpiPolicy>::call(
          local_state, rank, index);
      }

      template <
        typename MpiPolicy, typename LocalState, typename StateInteger>
      inline std::pair<yampi::rank, StateInteger> qubit_value_to_rank_index(
        MpiPolicy const, LocalState const& local_state, StateInteger const qubit_value)
      {
        return ::ket::mpi::utility::dispatch::qubit_value_to_rank_index<MpiPolicy>::call(
          local_state, qubit_value);
      }

# ifdef KET_USE_DIAGONAL_LOOP
      // 170607-: 
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState,
        typename StateInteger, typename BitInteger, typename Allocator,
        typename Function0, typename Function1, typename... ControlQubits>
      inline void diagonal_loop(
        MpiPolicy const, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        Function0&& function0, Function1&& function1,
        ControlQubits... control_qubits)
      {
        return ::ket::mpi::utility::dispatch::diagonal_loop<MpiPolicy>::call(
          parallel_policy, local_state, permutation, communicator, environment,
          target_qubit,
          std::forward<Function0>(function0), std::forward<Function1>(function1),
          control_qubits...);
      }
# endif // KET_USE_DIAGONAL_LOOP
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GENERAL_MPI_HPP
