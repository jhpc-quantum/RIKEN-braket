#ifndef KET_MPI_UTILITY_SIMPLE_MPI_HPP
# define KET_MPI_UTILITY_SIMPLE_MPI_HPP

# include <cstddef>
# include <cassert>
# ifndef NDEBUG
#   include <iostream>
# endif
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
# ifdef KET_USE_BARRIER
#   include <yampi/barrier.hpp>
# endif // KET_USE_BARRIER
# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
#   include <yampi/complete_exchange.hpp>
# endif // KET_USE_COLLECTIVE_COMMUNICATIONS

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# ifndef NDEBUG
#   include <ket/mpi/page/is_on_page.hpp>
#   include <ket/mpi/page/none_on_page.hpp>
# endif
# include <ket/mpi/utility/detail/make_local_swap_qubit.hpp>
# include <ket/mpi/utility/detail/interchange_qubits.hpp>
# include <ket/mpi/utility/detail/for_each_in_diagonal_loop.hpp>
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
        class simple_mpi
        {
         public:
          explicit simple_mpi() noexcept { }
        }; // class simple_mpi

        inline simple_mpi make_simple_mpi() noexcept { return simple_mpi();  }

        namespace meta
        {
          template <typename T>
          struct is_mpi_policy
            : std::false_type
          { }; // struct is_mpi_policy<T>

          template <>
          struct is_mpi_policy< ::ket::mpi::utility::policy::simple_mpi >
            : std::true_type
          { }; // struct is_mpi_policy< ::ket::mpi::utility::policy::simple_mpi >
        } // namespace meta

        namespace dispatch
        {
          template <typename MpiPolicy>
          struct num_data_blocks;

          template <>
          struct num_data_blocks< ::ket::mpi::utility::policy::simple_mpi >
          {
            static std::size_t call(
              ::ket::mpi::utility::policy::simple_mpi const,
              yampi::communicator const&, yampi::environment const&)
            { return 1u; }
          }; // struct num_data_blocks< ::ket::mpi::utility::policy::simple_mpi >

          // M
          template <typename MpiPolicy>
          struct num_global_qubits;

          template <>
          struct num_global_qubits< ::ket::mpi::utility::policy::simple_mpi >
          {
            static std::size_t call(
              ::ket::mpi::utility::policy::simple_mpi const,
              yampi::communicator const& communicator, yampi::environment const& environment)
            {
              auto const num_processes = communicator.size(environment);
              auto const result = ::ket::utility::integer_log2<std::size_t>(num_processes);
              assert(::ket::utility::integer_exp2<decltype(num_processes)>(result) == num_processes);
              return result;
            }
          }; // struct num_global_qubits< ::ket::mpi::utility::policy::simple_mpi >

          // g
          template <typename MpiPolicy>
          struct global_qubit_value;

          template <>
          struct global_qubit_value< ::ket::mpi::utility::policy::simple_mpi >
          {
            static std::size_t call(
              ::ket::mpi::utility::policy::simple_mpi const, yampi::rank const rank)
            { return static_cast<std::size_t>(rank.mpi_rank()); }

            static std::size_t call(
              ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
              yampi::communicator const& communicator, yampi::environment const& environment)
            { return call(mpi_policy, communicator.rank(environment)); }
          }; // struct global_qubit_value< ::ket::mpi::utility::policy::simple_mpi >

          // r
          template <typename MpiPolicy>
          struct rank;

          template <>
          struct rank< ::ket::mpi::utility::policy::simple_mpi >
          {
            template <typename StateInteger>
            static yampi::rank call(
              ::ket::mpi::utility::policy::simple_mpi const, StateInteger const global_qubit_value,
              yampi::communicator const& communicator, yampi::environment const& environment)
            {
              assert(global_qubit_value < static_cast<StateInteger>(communicator.size(environment)));
              return yampi::rank{static_cast<int>(global_qubit_value)};
            }
          }; // struct rank< ::ket::mpi::utility::policy::simple_mpi >

          // 2^L
          template <typename MpiPolicy>
          struct data_block_size;

          template <>
          struct data_block_size< ::ket::mpi::utility::policy::simple_mpi >
          {
            template <typename LocalState>
            static auto call(
              ::ket::mpi::utility::policy::simple_mpi const, LocalState const& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment)
              -> decltype(boost::size(local_state))
            {
              auto const result = boost::size(local_state);
              assert(::ket::utility::integer_exp2<decltype(result)>(::ket::utility::integer_log2<std::size_t>(result)) == result);
              return result;
            }
          }; // struct data_block_size< ::ket::mpi::utility::policy::simple_mpi >

          // L
          template <typename MpiPolicy>
          struct num_local_qubits;

          template <>
          struct num_local_qubits< ::ket::mpi::utility::policy::simple_mpi >
          {
            template <typename StateInteger>
            static std::size_t call(::ket::mpi::utility::policy::simple_mpi const, StateInteger const data_block_size)
            { return ::ket::utility::integer_log2<std::size_t>(data_block_size); }

            template <typename LocalState>
            static std::size_t call(
              ::ket::mpi::utility::policy::simple_mpi const mpi_policy, LocalState const& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment)
            {
              return call(
                mpi_policy,
                ::ket::mpi::utility::policy::dispatch::data_block_size< ::ket::mpi::utility::policy::simple_mpi >::call(
                  mpi_policy, local_state, communicator, environment));
            }
          }; // struct num_local_qubits< ::ket::mpi::utility::policy::simple_mpi >

          // N = L + M
          template <typename MpiPolicy>
          struct num_qubits;

          template <>
          struct num_qubits< ::ket::mpi::utility::policy::simple_mpi >
          {
            template <typename LocalState>
            static std::size_t call(
              ::ket::mpi::utility::policy::simple_mpi const mpi_policy, LocalState const& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment)
            {
              return ::ket::mpi::utility::policy::dispatch::num_local_qubits< ::ket::mpi::utility::policy::simple_mpi >::call(mpi_policy, local_state, communicator, environment)
                + ::ket::mpi::utility::policy::dispatch::num_global_qubits< ::ket::mpi::utility::policy::simple_mpi >::call(mpi_policy, communicator, environment);
            }
          }; // struct num_qubits< ::ket::mpi::utility::policy::simple_mpi >
        } // namespace dispatch

        template <typename MpiPolicy>
        inline auto num_data_blocks(
          MpiPolicy const& mpi_policy,
          yampi::communicator const& communicator, yampi::environment const& environment)
          -> decltype(::ket::mpi::utility::policy::dispatch::num_data_blocks<MpiPolicy>::call(mpi_policy, communicator, environment))
        { return ::ket::mpi::utility::policy::dispatch::num_data_blocks<MpiPolicy>::call(mpi_policy, communicator, environment); }


        // M
        template <typename MpiPolicy>
        inline auto num_global_qubits(
          MpiPolicy const& mpi_policy,
          yampi::communicator const& communicator, yampi::environment const& environment)
          -> decltype(::ket::mpi::utility::policy::dispatch::num_global_qubits<MpiPolicy>::call(mpi_policy, communicator, environment))
        { return ::ket::mpi::utility::policy::dispatch::num_global_qubits<MpiPolicy>::call(mpi_policy, communicator, environment); }

        // g
        template <typename MpiPolicy>
        inline auto global_qubit_value(MpiPolicy const& mpi_policy, yampi::rank const rank)
          -> decltype(::ket::mpi::utility::policy::dispatch::global_qubit_value<MpiPolicy>::call(mpi_policy, rank))
        { return ::ket::mpi::utility::policy::dispatch::global_qubit_value<MpiPolicy>::call(mpi_policy, rank); }

        // g
        template <typename MpiPolicy>
        inline auto global_qubit_value(
          MpiPolicy const& mpi_policy, yampi::communicator const& communicator, yampi::environment const& environment)
          -> decltype(::ket::mpi::utility::policy::dispatch::global_qubit_value<MpiPolicy>::call(mpi_policy, communicator, environment))
        { return ::ket::mpi::utility::policy::dispatch::global_qubit_value<MpiPolicy>::call(mpi_policy, communicator, environment); }

        // r
        template <typename MpiPolicy, typename StateInteger>
        inline yampi::rank rank(
          MpiPolicy const& mpi_policy, StateInteger const global_qubit_value,
          yampi::communicator const& communicator, yampi::environment const& environment)
        { return ::ket::mpi::utility::policy::dispatch::rank<MpiPolicy>::call(mpi_policy, global_qubit_value, communicator, environment); }

        // 2^L
        template <typename MpiPolicy, typename LocalState>
        inline auto data_block_size(
          MpiPolicy const& mpi_policy, LocalState const& local_state, yampi::communicator const& communicator, yampi::environment const& environment)
          -> decltype(::ket::mpi::utility::policy::dispatch::data_block_size<MpiPolicy>::call(mpi_policy, local_state, communicator, environment))
        { return ::ket::mpi::utility::policy::dispatch::data_block_size<MpiPolicy>::call(mpi_policy, local_state, communicator, environment); }

        // L
        template <typename MpiPolicy, typename StateInteger>
        inline auto num_local_qubits(MpiPolicy const& mpi_policy, StateInteger const data_block_size)
          -> decltype(::ket::mpi::utility::policy::dispatch::num_local_qubits<MpiPolicy>::call(mpi_policy, data_block_size))
        { return ::ket::mpi::utility::policy::dispatch::num_local_qubits<MpiPolicy>::call(mpi_policy, data_block_size); }

        // L
        template <typename MpiPolicy, typename LocalState>
        inline auto num_local_qubits(
          MpiPolicy const& mpi_policy, LocalState const& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment)
          -> decltype(::ket::mpi::utility::policy::dispatch::num_local_qubits<MpiPolicy>::call(mpi_policy, local_state, communicator, environment))
        { return ::ket::mpi::utility::policy::dispatch::num_local_qubits<MpiPolicy>::call(mpi_policy, local_state, communicator, environment); }

        // N = L + M
        template <typename MpiPolicy, typename LocalState>
        inline auto num_qubits(
          MpiPolicy const& mpi_policy, LocalState const& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment)
          -> decltype(::ket::mpi::utility::policy::dispatch::num_qubits<MpiPolicy>::call(mpi_policy, local_state, communicator, environment))
        { return ::ket::mpi::utility::policy::dispatch::num_qubits<MpiPolicy>::call(mpi_policy, local_state, communicator, environment); }
      } // namespace policy

      namespace dispatch
      {
        template <std::size_t num_qubits_of_operation, typename MpiPolicy>
        struct maybe_interchange_qubits;

        template <std::size_t num_qubits_of_operation>
        struct maybe_interchange_qubits<num_qubits_of_operation, ::ket::mpi::utility::policy::simple_mpi>
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            static_assert(
              std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(
              ::ket::utility::integer_exp2<StateInteger>(
                ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)))
              == static_cast<StateInteger>(boost::size(local_state)));

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const least_global_permutated_qubit
              = permutated_qubit_type{::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment)};

            auto permutated_global_swap_qubits = std::array<permutated_qubit_type, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutated_global_swap_qubits[index] = permutation[qubits[index]];

            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              if (permutated_global_swap_qubits[index] >= least_global_permutated_qubit)
                continue;

              call_lower_maybe_interchange_qubits(
                index,
                mpi_policy, parallel_policy, local_state, qubits, unswappable_qubits,
                permutation, buffer, communicator, environment);
              return;
            }

# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
            do_call(
              mpi_policy, parallel_policy, local_state,
              least_global_permutated_qubit, permutated_global_swap_qubits,
              qubits, unswappable_qubits, permutation, communicator, environment,
              [&buffer](
                LocalState& local_state,
                StateInteger const data_block_index, StateInteger const data_block_size,
                StateInteger const source_local_first_index, StateInteger const source_local_last_index,
                yampi::rank const target_rank,
                yampi::communicator const& communicator, yampi::environment const& environment)
              {
                ::ket::mpi::utility::detail::interchange_qubits(
                  local_state, buffer,
                  data_block_index, data_block_size,
                  source_local_first_index, source_local_last_index,
                  target_rank, communicator, environment);
              },
              [](LocalState& local_state, yampi::communicator const& communicator, yampi::environment const& environment)
              { yampi::complete_exchange(yampi::in_place, yampi::range_to_buffer(local_state), communicator, environment); });
# else // KET_USE_COLLECTIVE_COMMUNICATIONS
            do_call(
              mpi_policy, parallel_policy, local_state,
              least_global_permutated_qubit, permutated_global_swap_qubits,
              qubits, unswappable_qubits, permutation, communicator, environment,
              [&buffer](
                LocalState& local_state,
                StateInteger const data_block_index, StateInteger const data_block_size,
                StateInteger const source_local_first_index, StateInteger const source_local_last_index,
                yampi::rank const target_rank,
                yampi::communicator const& communicator, yampi::environment const& environment)
              {
                ::ket::mpi::utility::detail::interchange_qubits(
                  local_state, buffer,
                  data_block_index, data_block_size,
                  source_local_first_index, source_local_last_index,
                  target_rank, communicator, environment);
              });
# endif // KET_USE_COLLECTIVE_COMMUNICATIONS
          }

          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static void call(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            static_assert(
              std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(
              ::ket::utility::integer_exp2<StateInteger>(
                ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)))
              == static_cast<StateInteger>(boost::size(local_state)));

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const least_global_permutated_qubit
              = permutated_qubit_type{::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment)};

            auto permutated_global_swap_qubits = std::array<permutated_qubit_type, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutated_global_swap_qubits[index] = permutation[qubits[index]];

            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              if (permutated_global_swap_qubits[index] >= least_global_permutated_qubit)
                return;

              call_lower_maybe_interchange_qubits(
                index,
                mpi_policy, parallel_policy, local_state, qubits, unswappable_qubits,
                permutation, buffer, datatype, communicator, environment);
              return;
            }

# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
            do_call(
              mpi_policy, parallel_policy, local_state,
              least_global_permutated_qubit, permutated_global_swap_qubits,
              qubits, unswappable_qubits, permutation, communicator, environment,
              [&buffer, &datatype](
                LocalState& local_state,
                StateInteger const data_block_index, StateInteger const data_block_size,
                StateInteger const source_local_first_index, StateInteger const source_local_last_index,
                yampi::rank const target_rank,
                yampi::communicator const& communicator, yampi::environment const& environment)
              {
                ::ket::mpi::utility::detail::interchange_qubits(
                  local_state, buffer,
                  data_block_index, data_block_size,
                  source_local_first_index, source_local_last_index,
                  datatype, target_rank, communicator, environment);
              },
              [&datatype](LocalState& local_state, yampi::communicator const& communicator, yampi::environment const& environment)
              { yampi::complete_exchange(yampi::in_place, yampi::range_to_buffer(local_state, datatype), communicator, environment); });
# else // KET_USE_COLLECTIVE_COMMUNICATIONS
            do_call(
              mpi_policy, parallel_policy, local_state,
              least_global_permutated_qubit, permutated_global_swap_qubits,
              qubits, unswappable_qubits, permutation, communicator, environment,
              [&buffer, &datatype](
                LocalState& local_state,
                StateInteger const data_block_index, StateInteger const data_block_size,
                StateInteger const source_local_first_index, StateInteger const source_local_last_index,
                yampi::rank const target_rank,
                yampi::communicator const& communicator, yampi::environment const& environment)
              {
                ::ket::mpi::utility::detail::interchange_qubits(
                  local_state, buffer,
                  data_block_index, data_block_size,
                  source_local_first_index, source_local_last_index,
                  datatype, target_rank, communicator, environment);
              });
# endif // KET_USE_COLLECTIVE_COMMUNICATIONS
          }

         private:
          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator>
          static void initialize_local_swap_qubits(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            ParallelPolicy const parallel_policy,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation >& local_swap_qubits,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation >& permutated_local_swap_qubits,
            LocalState& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
# ifdef KET_USE_BARRIER
            ::yampi::barrier(communicator, environment);
# endif // KET_USE_BARRIER

            ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"interchange_qubits<"}, num_qubits_of_operation, '>'), environment};

# ifndef NDEBUG
            auto const maybe_io_rank = yampi::lowest_io_process(environment);
            auto const my_rank = yampi::communicator{yampi::tags::world_communicator}.rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation before changing qubits] " << permutation << std::endl;
# endif // NDEBUG

            auto const num_data_blocks = static_cast<StateInteger>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment));
            auto const data_block_size = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment));

            // (ex.: num_qubits_of_operation == 3)
            //  Swaps between xxbxb'xb''xx|cc'c''xxxxxxxx and
            // xxcxc'xc''xx|bb'b''xxxxxxxx (c = b or ~b). Upper qubits are
            // global qubits representing MPI rank. Lower qubits are local
            // qubits representing memory address. The first three upper qubits
            // in the local qubits are "local swap qubits". Three bits in global
            // qubits and the "local swap qubits" would be swapped.

            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              permutated_local_swap_qubits[index] = least_global_permutated_qubit - static_cast<BitInteger>(std::size_t{1u} + index);
              local_swap_qubits[index]
                = ::ket::mpi::utility::detail::make_local_swap_qubit(
                    parallel_policy, local_state, permutation,
                    unswappable_qubits, permutated_local_swap_qubits[index],
                    num_data_blocks, data_block_size, communicator, environment);
            }

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local swap qubits] " << permutation << std::endl;
# endif // NDEBUG
          }

          template <typename StateInteger, typename BitInteger, typename Allocator>
          static void update_permutation(
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& local_swap_qubits,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            using ::ket::mpi::permutate;
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutate(permutation, qubits[index], local_swap_qubits[index]);

# ifndef NDEBUG
            auto const maybe_io_rank = yampi::lowest_io_process(environment);
            auto const my_rank = yampi::communicator{yampi::tags::world_communicator}.rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local/global qubits] " << permutation << std::endl;
# endif // NDEBUG

# ifdef KET_USE_BARRIER
            ::yampi::barrier(communicator, environment);
# endif // KET_USE_BARRIER
          }

# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
          template <typename StateInteger, typename BitInteger>
          static std::pair<yampi::color, int> generate_color_key(
            StateInteger const global_qubit_value,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_global_swap_qubits)
          {
            // initialization of permutated_global_qubit_index_pairs ({sorted_global_qubit, corresponding_index_in_some_arrays}, ...)
            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            using permutated_global_qubit_index_pair_type = std::pair<permutated_qubit_type, std::size_t>;
            auto permutated_global_qubit_index_pairs = std::array<permutated_global_qubit_index_pair_type, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutated_global_qubit_index_pairs[index] = std::make_pair(permutated_global_swap_qubits[index], index);

            std::sort(
              std::begin(permutated_global_qubit_index_pairs),
              std::end(permutated_global_qubit_index_pairs),
              [](permutated_global_qubit_index_pair_type const& lhs, permutated_global_qubit_index_pair_type const& rhs)
              { return lhs.first < rhs.first;});

            // initialization of permutated_global_qubit_masks (000001000000, 000000001000, 001000000000)
            auto permutated_global_qubit_masks = std::array<StateInteger, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutated_global_qubit_masks[index]
                = (StateInteger{1u} << permutated_global_swap_qubits[index]) >> least_global_permutated_qubit;

            // initialization of global_qubit_value_masks (000000111, 000011000, 001100000, 110000000)
            auto global_qubit_value_masks = std::array<StateInteger, num_qubits_of_operation + std::size_t{1u}>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              global_qubit_value_masks[index]
                = (permutated_global_qubit_masks[permutated_global_qubit_index_pairs[index].second] >> index)
                  - StateInteger{1u};
            global_qubit_value_masks[num_qubits_of_operation] = compl StateInteger{0u};

            std::transform(
              global_qubit_value_masks.rbegin(), std::prev(global_qubit_value_masks.rend()),
              std::next(global_qubit_value_masks.rbegin()), global_qubit_value_masks.rbegin(),
              std::minus<StateInteger>{});

            auto key = StateInteger{0u};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              key &= (((global_qubit_value bitand permutated_global_qubit_masks[index]) << least_global_permutated_qubit) >> permutated_global_swap_qubits[index]) << permutated_local_swap_qubits[index];

            auto color_integer = StateInteger{0u};
            for (auto index = std::size_t{0u}; index <= num_qubits_of_operation; ++index)
              color_integer |= ((global_qubit_value >> index) bitand global_qubit_value_masks[index]);

            return {yampi::color{static_cast<int>(color_integer)}, static_cast<int>(key)};
          }

          template <typename LocalState, typename StateInteger, typename BitInteger, typename Function>
          static void interchange_qubits_collective(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            LocalState& local_state,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_global_swap_qubits,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Function&& complete_exchange_qubits)
          {
            // xxbxb'xb''xx(|**********) => xxxxxx: color
            auto const global_qubit_value
              = static_cast<StateInteger>(::ket::mpi::utility::policy::global_qubit_value(mpi_policy, communicator, environment));
            auto const color_key
              = generate_color_key(global_qubit_value, permutated_local_swap_qubits, least_global_permutated_qubit, permutated_global_swap_qubits);

            auto local_communicator = yampi::communicator{communicator, color_key.first, color_key.second, environment};
            complete_exchange_qubits(local_state, local_communicator, environment);
          }
# endif // KET_USE_COLLECTIVE_COMMUNICATIONS

          template <typename LocalState, typename StateInteger, typename BitInteger, typename Function>
          static void interchange_qubits_p2p(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            LocalState& local_state,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_global_swap_qubits,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Function&& interchange_qubits)
          {
            auto const data_block_size = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment));

            // xxbxb'xb''xx(|xxxxxxxxxx)
            auto const source_global_qubit_value
              = static_cast<StateInteger>(::ket::mpi::utility::policy::global_qubit_value(mpi_policy, communicator, environment));

            for (auto target_global_mask = StateInteger{1u};
                 target_global_mask < ::ket::utility::integer_exp2<StateInteger>(num_qubits_of_operation);
                 ++target_global_mask)
            {
              auto target_global_masks = std::array<StateInteger, num_qubits_of_operation>{};
              for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                target_global_masks[index] = (target_global_mask bitand (StateInteger{1u} << index)) >> index;

              // xxcxc'xc''xx(|xxxxxxxxxx) (c = b or ~b, except for (c, c', c'') = (b, b', b''))
              auto mask = (target_global_masks[0u] << permutated_global_swap_qubits[0u]) >> least_global_permutated_qubit;
              for (auto index = std::size_t{1u}; index < num_qubits_of_operation; ++index)
                mask |= (target_global_masks[index] << permutated_global_swap_qubits[index]) >> least_global_permutated_qubit;
              auto const target_global_qubit_value = source_global_qubit_value xor mask;
              auto const target_rank
                = ::ket::mpi::utility::policy::rank(mpi_policy, target_global_qubit_value, communicator, environment);

              // (0000000|)cc'c''00000000
              auto source_local_first_index = StateInteger{0u};
              for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                source_local_first_index
                  |= ((target_global_qubit_value << least_global_permutated_qubit)
                      bitand (StateInteger{1u} << permutated_global_swap_qubits[index]))
                     >> (permutated_global_swap_qubits[index] - permutated_local_swap_qubits[index]);

              // (0000000|)cc'c''11111111 + 1
              auto const source_local_last_index
                = source_local_first_index + (data_block_size >> num_qubits_of_operation);

              ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"interchange_qubits<"}, num_qubits_of_operation, ">::swap"), environment};

              interchange_qubits(
                local_state, StateInteger{0u}, data_block_size,
                source_local_first_index, source_local_last_index,
                target_rank, communicator, environment);
            }
          }

# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
          template <typename NoncontiguousIterator>
          struct interchange_qubits_dispatch2
          {
            template <typename LocalState, typename StateInteger, typename BitInteger, typename Function1, typename Function2>
            static void call(
              ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
              LocalState& local_state,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_global_swap_qubits,
              yampi::communicator const& communicator, yampi::environment const& environment,
              Function1&& interchange_qubits, Function2&&)
            {
              interchange_qubits_p2p(
                mpi_policy, local_state,
                permutated_local_swap_qubits, least_global_permutated_qubit, permutated_global_swap_qubits,
                communicator, environment, std::forward<Function1>(interchange_qubits));
            }
          }; // struct interchange_qubits_dispatch2<NoncontiguousIterator>

          template <typename T>
          struct interchange_qubits_dispatch2<T*>
          {
            template <typename LocalState, typename StateInteger, typename BitInteger, typename Function1, typename Function2>
            static void call(
              ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
              LocalState& local_state,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_global_swap_qubits,
              yampi::communicator const& communicator, yampi::environment const& environment,
              Function1&&, Function2&& complete_exchange_qubits)
            {
              interchange_qubits_collective(
                mpi_policy, local_state,
                permutated_local_swap_qubits, least_global_permutated_qubit, permutated_global_swap_qubits,
                communicator, environment, std::forward<Function2>(complete_exchange_qubits));
            }
          }; // struct interchange_qubits_dispatch2<T*>

          template <typename LocalState_>
          struct interchange_qubits_dispatch1
          {
            template <typename LocalState, typename StateInteger, typename BitInteger, typename Function1, typename Function2>
            static void call(
              ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
              LocalState& local_state,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_global_swap_qubits,
              yampi::communicator const& communicator, yampi::environment const& environment,
              Function1&& interchange_qubits, Function2&& complete_exchange_qubits)
            {
              interchange_qubits_dispatch2<typename boost::range_iterator<LocalState_>::type>::call(
                mpi_policy, local_state,
                permutated_local_swap_qubits, least_global_permutated_qubit, permutated_global_swap_qubits,
                communicator, environment,
                std::forward<Function1>(interchange_qubits), std::forward<Function2>(complete_exchange_qubits));
            }
          }; // struct interchange_qubits_dispatch1<LocalState_>

          template <typename Complex, typename LocalStateAllocator>
          struct interchange_qubits_dispatch1<std::vector<Complex, LocalStateAllocator>>
          {
            template <typename StateInteger, typename BitInteger, typename Function1, typename Function2>
            static void call(
              ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
              std::vector<Complex, LocalStateAllocator>& local_state,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_global_swap_qubits,
              yampi::communicator const& communicator, yampi::environment const& environment,
              Function1&&, Function2&& complete_exchange_qubits)
            {
              interchange_qubits_collective(
                mpi_policy, local_state,
                permutated_local_swap_qubits, least_global_permutated_qubit, permutated_global_swap_qubits,
                communicator, environment, std::forward<Function2>(complete_exchange_qubits));
            }
          }; // struct interchange_qubits_dispatch1<std::vector<Complex, LocalStateAllocator>>
# endif // KET_USE_COLLECTIVE_COMMUNICATIONS

# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename Function1, typename Function2>
          static void do_call(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_global_swap_qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Function1&& interchange_qubits, Function2&& complete_exchange_qubits)
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
            auto permutated_local_swap_qubits = std::array<permutated_qubit_type, num_qubits_of_operation>{};
            auto local_swap_qubits = std::array<qubit_type, num_qubits_of_operation>{};
            initialize_local_swap_qubits(
              mpi_policy, parallel_policy, local_swap_qubits, permutated_local_swap_qubits,
              local_state, least_global_permutated_qubit, unswappable_qubits, permutation, communicator, environment);

            interchange_qubits_dispatch1<LocalState>::call(
              mpi_policy, local_state,
              permutated_local_swap_qubits, least_global_permutated_qubit, permutated_global_swap_qubits,
              communicator, environment,
              std::forward<Function1>(interchange_qubits), std::forward<Function2>(complete_exchange_qubits));

            update_permutation(permutation, qubits, local_swap_qubits, communicator, environment);
          }
# else // KET_USE_COLLECTIVE_COMMUNICATIONS
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename Function>
          static void do_call(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_global_swap_qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Function&& interchange_qubits)
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
            auto permutated_local_swap_qubits = std::array<permutated_qubit_type, num_qubits_of_operation>{};
            auto local_swap_qubits = std::array<qubit_type, num_qubits_of_operation>{};
            initialize_local_swap_qubits(
              mpi_policy, parallel_policy, local_swap_qubits, permutated_local_swap_qubits,
              local_state, least_global_permutated_qubit, unswappable_qubits, permutation, communicator, environment);

            interchange_qubits_p2p(
              mpi_policy, local_state,
              permutated_local_swap_qubits, least_global_permutated_qubit, permutated_global_swap_qubits,
              communicator, environment, std::forward<Function>(interchange_qubits));

            update_permutation(permutation, qubits, local_swap_qubits, communicator, environment);
          }
# endif // KET_USE_COLLECTIVE_COMMUNICATIONS

          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call_lower_maybe_interchange_qubits(
            std::size_t const new_unswappable_qubit_index,
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            assert(new_unswappable_qubit_index < num_qubits_of_operation);

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto new_qubits = std::array<qubit_type, num_qubits_of_operation - 1u>{};
            std::copy(
              std::begin(qubits) + new_unswappable_qubit_index + 1u, std::end(qubits),
              std::copy(
                std::begin(qubits), std::begin(qubits) + new_unswappable_qubit_index,
                std::begin(new_qubits)));

            auto new_unswappable_qubits = std::array<qubit_type, num_unswappable_qubits + 1u>{};
            std::copy(
              std::begin(unswappable_qubits), std::end(unswappable_qubits),
              std::begin(new_unswappable_qubits));
            new_unswappable_qubits.back() = qubits[new_unswappable_qubit_index];

            using lower_maybe_interchange_qubits
              = ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  num_qubits_of_operation - 1u, ::ket::mpi::utility::policy::simple_mpi>;
            lower_maybe_interchange_qubits::call(
              mpi_policy, parallel_policy, local_state, new_qubits, new_unswappable_qubits,
              permutation, buffer, communicator, environment);
          }

          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static void call_lower_maybe_interchange_qubits(
            std::size_t const new_unswappable_qubit_index,
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            assert(new_unswappable_qubit_index < num_qubits_of_operation);

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto new_qubits = std::array<qubit_type, num_qubits_of_operation - 1u>{};
            std::copy(
              std::begin(qubits) + new_unswappable_qubit_index + 1u, std::end(qubits),
              std::copy(
                std::begin(qubits), std::begin(qubits) + new_unswappable_qubit_index,
                std::begin(new_qubits)));

            auto new_unswappable_qubits = std::array<qubit_type, num_unswappable_qubits + 1u>{};
            std::copy(
              std::begin(unswappable_qubits), std::end(unswappable_qubits),
              std::begin(new_unswappable_qubits));
            new_unswappable_qubits.back() = qubits[new_unswappable_qubit_index];

            using lower_maybe_interchange_qubits
              = ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  num_qubits_of_operation - 1u, ::ket::mpi::utility::policy::simple_mpi>;
            lower_maybe_interchange_qubits::call(
              mpi_policy, parallel_policy, local_state, new_qubits, new_unswappable_qubits,
              permutation, buffer, datatype, communicator, environment);
          }
        }; // struct maybe_interchange_qubits<num_qubits_of_operation, ::ket::mpi::utility::policy::simple_mpi>

        template <>
        struct maybe_interchange_qubits<0u, ::ket::mpi::utility::policy::simple_mpi>
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call(
            ::ket::mpi::utility::policy::simple_mpi const,
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
            ::ket::mpi::utility::policy::simple_mpi const,
            ParallelPolicy const, LocalState&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, 0u > const&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>&,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>&,
            yampi::datatype_base<DerivedDatatype> const&,
            yampi::communicator const&, yampi::environment const&)
          { }
        }; // struct maybe_interchange_qubits<0u, ::ket::mpi::utility::policy::simple_mpi>

        template <typename MpiPolicy>
        struct rank_index_to_qubit_value
        {
          template <typename LocalState, typename StateInteger>
          static StateInteger call(
            ::ket::mpi::utility::policy::simple_mpi const,
            LocalState const& local_state,
            yampi::rank const rank, StateInteger const index);
        }; // struct rank_index_to_qubit_value<MpiPolicy>

        template <>
        struct rank_index_to_qubit_value< ::ket::mpi::utility::policy::simple_mpi >
        {
          template <typename LocalState, typename StateInteger>
          static StateInteger call(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            LocalState const& local_state,
            yampi::rank const rank, StateInteger const index)
          {
            // g
            auto const global_qubit_value = static_cast<StateInteger>(::ket::mpi::utility::policy::global_qubit_value(mpi_policy, rank));
            // 2^L
            auto const data_block_size = static_cast<StateInteger>(boost::size(local_state));

            return global_qubit_value * data_block_size + index;
          }
        }; // struct rank_index_to_qubit_value< ::ket::mpi::utility::policy::simple_mpi >

        template <typename MpiPolicy>
        struct qubit_value_to_rank_index
        {
          template <typename LocalState, typename StateInteger>
          static std::pair<yampi::rank, StateInteger> call(
            ::ket::mpi::utility::policy::simple_mpi const,
            LocalState const& local_state, StateInteger const qubit_value,
            yampi::communicator const&, yampi::environment const&);
        }; // struct qubit_value_to_rank_index<MpiPolicy>

        template <>
        struct qubit_value_to_rank_index< ::ket::mpi::utility::policy::simple_mpi >
        {
          template <typename LocalState, typename StateInteger>
          static std::pair<yampi::rank, StateInteger> call(
            ::ket::mpi::utility::policy::simple_mpi const,
            LocalState const& local_state, StateInteger const qubit_value,
            yampi::communicator const&, yampi::environment const&)
          {
            auto const data_block_size = static_cast<StateInteger>(boost::size(local_state));
            return std::make_pair(yampi::rank{static_cast<int>(qubit_value / data_block_size)}, qubit_value % data_block_size);
          }
        }; // struct qubit_value_to_rank_index< ::ket::mpi::utility::policy::simple_mpi >
      } // namespace dispatch

      namespace dispatch
      {
# ifdef KET_USE_DIAGONAL_LOOP
        template <typename MpiPolicy>
        struct diagonal_loop
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1, typename... ControlQubits>
          static void call(
            MpiPolicy const&, ParallelPolicy const, LocalState& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::communicator const& communicator,
            yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            ControlQubits... control_qubits);
        }; // struct diagonal_loop<MpiPolicy>

        template <>
        struct diagonal_loop< ::ket::mpi::utility::policy::simple_mpi >
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1, typename... ControlQubits>
          static void call(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            ParallelPolicy const parallel_policy, LocalState& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::communicator const& communicator,
            yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            ControlQubits... control_qubits)
          {
            using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
            auto local_permutated_control_qubits = std::array<permutated_control_qubit_type, 0u>{};

            auto const present_rank = communicator.rank(environment);
            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const least_global_permutated_qubit
              = permutated_qubit_type{::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment)};

            call_impl(
              mpi_policy, parallel_policy, local_state, permutation, present_rank,
              least_global_permutated_qubit, target_qubit,
              std::forward<Function0>(function0),
              std::forward<Function1>(function1),
              local_permutated_control_qubits, control_qubits...);
          }

         private:
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1,
            std::size_t num_local_control_qubits, typename... ControlQubits>
          static void call_impl(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            ParallelPolicy const parallel_policy, LocalState& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const present_rank,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            std::array<
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >,
              num_local_control_qubits> const& local_permutated_control_qubits,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ControlQubits... control_qubits)
          {
            auto const permutated_control_qubit = permutation[control_qubit];

            if (permutated_control_qubit < least_global_permutated_qubit)
            {
              using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
              auto new_local_permutated_control_qubits
                = std::array<permutated_control_qubit_type, num_local_control_qubits + 1u>{};
              std::copy(
                std::begin(local_permutated_control_qubits),
                std::end(local_permutated_control_qubits),
                std::begin(new_local_permutated_control_qubits));
              new_local_permutated_control_qubits.back() = permutated_control_qubit;

              call_impl(
                mpi_policy, parallel_policy, local_state, permutation, present_rank,
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

              if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand mask) != zero_state_integer)
                call_impl(
                  mpi_policy, parallel_policy, local_state, permutation, present_rank,
                  least_global_permutated_qubit, target_qubit,
                  std::forward<Function0>(function0),
                  std::forward<Function1>(function1),
                  local_permutated_control_qubits, control_qubits...);
            }
          }

          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1,
            std::size_t num_local_control_qubits>
          static void call_impl(
            ::ket::mpi::utility::policy::simple_mpi const mpi_policy,
            ParallelPolicy const parallel_policy, LocalState& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const present_rank,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            std::array<
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >,
              num_local_control_qubits> const& local_permutated_control_qubits)
          {
            auto const permutated_target_qubit = permutation[target_qubit];

            static constexpr auto one_state_integer = StateInteger{1u};

            auto const last_local_qubit_value = one_state_integer << least_global_permutated_qubit;

            if (permutated_target_qubit < least_global_permutated_qubit)
            {
              auto const mask = one_state_integer << permutated_target_qubit;

#   ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
              ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                parallel_policy, local_state, StateInteger{0u}, static_cast<StateInteger>(boost::size(local_state)), last_local_qubit_value, local_permutated_control_qubits,
                [&function0, &function1, mask](auto const iter, StateInteger const state_integer)
                {
                  static constexpr auto zero_state_integer = StateInteger{0u};

                  if ((state_integer bitand mask) == zero_state_integer)
                    function0(iter, state_integer);
                  else
                    function1(iter, state_integer);
                });
#   else // BOOST_NO_CXX14_GENERIC_LAMBDAS
              ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                parallel_policy, local_state, StateInteger{0u}, static_cast<StateInteger>(boost::size(local_state)), last_local_qubit_value, local_permutated_control_qubits,
                make_call_function_if_local(std::forward<Function0>(function0), std::forward<Function1>(function1), mask));
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
            }
            else
            {
              auto const mask
                = one_state_integer << (permutated_target_qubit - least_global_permutated_qubit);

              static constexpr auto zero_state_integer = StateInteger{0u};

              if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand mask) == zero_state_integer)
                ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                  parallel_policy, local_state, StateInteger{0u}, static_cast<StateInteger>(boost::size(local_state)), last_local_qubit_value, local_permutated_control_qubits,
                  std::forward<Function0>(function0));
              else
                ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                  parallel_policy, local_state, StateInteger{0u}, static_cast<StateInteger>(boost::size(local_state)), last_local_qubit_value, local_permutated_control_qubits,
                  std::forward<Function1>(function1));
            }
          }

#   ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Function0, typename Function1, typename StateInteger>
          struct call_function_if_local
          {
            Function0 function0_;
            Function1 function1_;
            StateInteger mask_;

            template <typename Iterator>
            void operator()(Iterator const iter, StateInteger const state_integer)
            {
              static constexpr auto zero_state_integer = StateInteger{0u};

              if ((state_integer bitand mask_) == zero_state_integer)
                function0_(iter, state_integer);
              else
                function1_(iter, state_integer);
            }
          }; // struct call_function_if_local<Function0, Function1, StateInteger>

          template <typename Function0, typename Function1, typename StateInteger>
          static call_function_if_local<Function0, Function1, StateInteger>
          make_call_function_if_local(Function0&& function0, Function1&& function1, StateInteger const mask)
          { return call_function_if_local<Function0, Function1, StateInteger>{std::forward<Function0>(function0), std::forward<Function1>(function1), mask}; }
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }; // struct diagonal_loop< ::ket::mpi::utility::policy::simple_mpi >
# endif // KET_USE_DIAGONAL_LOOP
      } // namespace dispatch

      template <
        typename MpiPolicy, typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator>
      void maybe_interchange_qubits(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
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
          mpi_policy, parallel_policy,
          local_state, qubits, unswappable_qubits, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      void maybe_interchange_qubits(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
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
          mpi_policy, parallel_policy,
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
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubits, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      void maybe_interchange_qubits(
        ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubits, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator>
      void maybe_interchange_qubits(
        LocalState& local_state,
        std::array< ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
        ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubits, permutation, buffer, communicator, environment);
      }

      template <
        typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      void maybe_interchange_qubits(
        LocalState& local_state,
        std::array< ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
        ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubits, permutation, buffer, datatype, communicator, environment);
      }

      template <typename MpiPolicy, typename LocalState, typename StateInteger>
      inline StateInteger rank_index_to_qubit_value(
        MpiPolicy const& mpi_policy, LocalState const& local_state, yampi::rank const rank, StateInteger const index)
      { return ::ket::mpi::utility::dispatch::rank_index_to_qubit_value<MpiPolicy>::call(mpi_policy, local_state, rank, index); }

      template <typename MpiPolicy, typename LocalState, typename StateInteger>
      inline std::pair<yampi::rank, StateInteger> qubit_value_to_rank_index(
        MpiPolicy const& mpi_policy,
        LocalState const& local_state, StateInteger const qubit_value,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::utility::dispatch::qubit_value_to_rank_index<MpiPolicy>::call(
          mpi_policy, local_state, qubit_value, communicator, environment);
      }

# ifdef KET_USE_DIAGONAL_LOOP
      template <
        typename MpiPolicy, typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, typename Allocator,
        typename Function0, typename Function1, typename... ControlQubits>
      inline void diagonal_loop(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        Function0&& function0, Function1&& function1,
        ControlQubits... control_qubits)
      {
        assert(not ::ket::mpi::page::is_on_page(permutation[target_qubit], local_state));
        assert(::ket::mpi::page::none_on_page(local_state, permutation[control_qubits]...));

        return ::ket::mpi::utility::dispatch::diagonal_loop<MpiPolicy>::call(
          mpi_policy, parallel_policy, local_state, permutation, communicator, environment,
          target_qubit,
          std::forward<Function0>(function0), std::forward<Function1>(function1),
          control_qubits...);
      }
# endif // KET_USE_DIAGONAL_LOOP
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_SIMPLE_MPI_HPP
