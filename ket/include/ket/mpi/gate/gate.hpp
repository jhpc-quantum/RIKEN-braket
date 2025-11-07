#ifndef KET_MPI_GATE_GATE_HPP
# define KET_MPI_GATE_GATE_HPP

# include <algorithm>
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   include <array>
# endif // KET_USE_BIT_MASKS_EXPLICITLY
# include <vector>
# include <iterator>
# include <utility>
# include <type_traits>

# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/qubit.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
#   include <ket/control_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/gate.hpp>
# include <ket/gate/meta/num_control_qubits.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/all_in_state_vector.hpp>
# include <ket/utility/none_in_state_vector.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/gate/page/gate.hpp>
# include <ket/mpi/page/page_size.hpp>
# include <ket/mpi/page/none_on_page.hpp>
# ifndef NDEBUG
#   include <ket/mpi/page/any_on_page.hpp>
# endif // NDEBUG
# include <ket/mpi/utility/apply_local_gate.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/buffer_range.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace local
      {
# ifndef KET_ENABLE_CACHE_AWARE_GATE_FUNCTION
        namespace nopage
        {
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename BufferAllocator, typename StateInteger, typename Function, typename... Qubits>
          inline auto gate(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask,
            Function&& function, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
          -> RandomAccessRange&
          {
            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
              [parallel_policy, &function, permutated_qubits...](auto const first, auto const last)
              { ::ket::gate::nocache::gate(parallel_policy, first, last, function, permutated_qubits.qubit()...); });
          }
        } // namespace nopage

        namespace page
        {
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename BufferAllocator, typename StateInteger, typename Function, typename... Qubits>
          inline auto gate(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask,
            Function&& function, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
          -> RandomAccessRange&
          {
            auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
            auto const data_block_size = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, rank_in_unit));
            auto const num_data_blocks = static_cast<StateInteger>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));

            using std::begin;
            auto const first = begin(local_state);
            for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
            {
              if ((static_cast<StateInteger>(::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit)) bitand unit_control_qubit_mask) != unit_control_qubit_mask)
                continue;

              ::ket::gate::nocache::gate(
                parallel_policy,
                first + data_block_index * data_block_size,
                first + (data_block_index + 1u) * data_block_size,
                function, permutated_qubits.qubit()...);
            }

            return local_state;
          }
        } // namespace page

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename BufferAllocator, typename StateInteger, typename Function, typename Qubit, typename... Qubits>
        inline auto gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          StateInteger const unit_control_qubit_mask,
          Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
        -> RandomAccessRange&
        {
          // Case 1) None of operated qubits is page qubit
          if (::ket::mpi::page::none_on_page(local_state, permutated_qubit, permutated_qubits...))
            return ::ket::mpi::gate::local::nopage::gate(
              mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask,
              std::forward<Function>(function), permutated_qubit, permutated_qubits...);

          // Case 2) Some operated qubits are page qubits
          return ::ket::mpi::gate::local::page::gate(
            mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask,
            std::forward<Function>(function), permutated_qubit, permutated_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename BufferAllocator, typename StateInteger, typename Function>
        inline auto gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          StateInteger const unit_control_qubit_mask,
          Function&& function)
        -> RandomAccessRange&
        {
          // Case 1) None of operated qubits is page qubit
          // ALWAYS SATISFIED
          return ::ket::mpi::gate::local::nopage::gate(
            mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask,
            std::forward<Function>(function));
        }
# else // KET_ENABLE_CACHE_AWARE_GATE_FUNCTION
        namespace nopage
        {
          namespace all_on_cache
          {
            // Case 1-1-1) page size <= on-cache state size
            //   ex1: pppp|ppzzzzzzzz
            //               ^  ^  ^  <- operated qubits
            //   ex2: ....|..ppzzzzzz (num. local qubits <= num. on-cache qubits)
            //                  ^  ^  <- operated qubits
            namespace small
            {
              // First argument of Function: iterator_t<RandomAccessRange> (without page) or vector::iterator (with page)
              template <
                typename MpiPolicy, typename ParallelPolicy,
                typename RandomAccessRange, typename StateInteger, typename Function, typename Qubit, typename... Qubits>
              inline auto gate(
                MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
                RandomAccessRange& local_state,
                yampi::communicator const& communicator, yampi::environment const& environment,
                StateInteger const unit_control_qubit_mask,
                Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
              -> RandomAccessRange&
              {
                using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;

                constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
                using qubit_type = ::ket::qubit<StateInteger, bit_integer_type>;
                std::array<qubit_type, num_operated_qubits> unsorted_qubits{
                  ::ket::remove_control(permutated_qubit.qubit()), ::ket::remove_control(permutated_qubits.qubit())...};

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
                constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};

                std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_qubits_with_sentinel{
                  ::ket::remove_control(permutated_qubit.qubit()), ::ket::remove_control(permutated_qubits.qubit())...,
                  ::ket::make_qubit<StateInteger>(num_on_cache_qubits)};
                using std::begin;
                using std::end;
                std::sort(begin(sorted_qubits_with_sentinel), std::prev(end(sorted_qubits_with_sentinel)));

                return ::ket::mpi::utility::for_each_local_range(
                  mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                  [parallel_policy, &unsorted_qubits, &sorted_qubits_with_sentinel, &function](auto const first, auto const last)
                  { ::ket::gate::gate_detail::gate(parallel_policy, first, last, unsorted_qubits, sorted_qubits_with_sentinel, function); });
#   else // KET_USE_BIT_MASKS_EXPLICITLY
                std::array<StateInteger, num_operated_qubits> qubit_masks{};
                ::ket::gate::gate_detail::make_qubit_masks(qubit_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);
                std::array<StateInteger, num_operated_qubits + 1u> index_masks{};
                ::ket::gate::gate_detail::make_index_masks(index_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);

                return ::ket::mpi::utility::for_each_local_range(
                  mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                  [parallel_policy, &qubit_masks, &index_masks, &function](auto const first, auto const last)
                  { ::ket::gate::gate_detail::gate(parallel_policy, first, last, qubit_masks, index_masks, function); });
#   endif // KET_USE_BIT_MASKS_EXPLICITLY
              }

              template <
                typename MpiPolicy, typename ParallelPolicy,
                typename RandomAccessRange, typename StateInteger, typename Function>
              inline auto gate(
                MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
                RandomAccessRange& local_state,
                yampi::communicator const& communicator, yampi::environment const& environment,
                StateInteger const unit_control_qubit_mask,
                Function&& function)
              -> RandomAccessRange&
              {
                using qubit_type = ::ket::qubit<StateInteger>;
                using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
                std::array<qubit_type, bit_integer_type{0u}> unsorted_qubits{};

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
                constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};

                std::array<qubit_type, bit_integer_type{1u}> sorted_qubits_with_sentinel{::ket::make_qubit<StateInteger>(num_on_cache_qubits)};

                return ::ket::mpi::utility::for_each_local_range(
                  mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                  [parallel_policy, &unsorted_qubits, &sorted_qubits_with_sentinel, &function](auto const first, auto const last)
                  { ::ket::gate::gate_detail::gate(parallel_policy, first, last, unsorted_qubits, sorted_qubits_with_sentinel, function); });
#   else // KET_USE_BIT_MASKS_EXPLICITLY
                std::array<StateInteger, bit_integer_type{0u}> qubit_masks{};
                std::array<StateInteger, bit_integer_type{1u}> index_masks{compl StateInteger{0u}};

                return ::ket::mpi::utility::for_each_local_range(
                  mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                  [parallel_policy, &qubit_masks, &index_masks, &function](auto const first, auto const last)
                  { ::ket::gate::gate_detail::gate(parallel_policy, first, last, qubit_masks, index_masks, function); });
#   endif // KET_USE_BIT_MASKS_EXPLICITLY
              }
            } // namespace small

            // Case 1-1-2) page size > on-cache state size
            //   ex: ppxx|zzzzzzzzzz
            //             ^   ^ ^   <- operated qubits
            // First argument of Function: iterator_t<RandomAccessRange> (without page) or vector::iterator (with page)
            template <
              typename MpiPolicy, typename ParallelPolicy,
              typename RandomAccessRange, typename StateInteger, typename Function, typename Qubit, typename... Qubits>
            inline auto gate(
              MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment,
              StateInteger const unit_control_qubit_mask,
              Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
            -> RandomAccessRange&
            {
              static_assert(std::is_same<StateInteger, ::ket::meta::state_integer_t<Qubit>>::value, "The state_integer_type of Qubit should be the same as StateInteger");
              using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;

              constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
              using qubit_type = ::ket::qubit<StateInteger, bit_integer_type>;
              std::array<qubit_type, num_operated_qubits> unsorted_qubits{
                ::ket::remove_control(permutated_qubit.qubit()), ::ket::remove_control(permutated_qubits.qubit())...};

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
              constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};

              std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_qubits_with_sentinel{
                ::ket::remove_control(permutated_qubit.qubit()), ::ket::remove_control(permutated_qubits.qubit())...,
                ::ket::make_qubit<StateInteger>(num_on_cache_qubits)};
              using std::begin;
              using std::end;
              std::sort(begin(sorted_qubits_with_sentinel), std::prev(end(sorted_qubits_with_sentinel)));

              return ::ket::mpi::utility::for_each_local_range(
                mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                [parallel_policy, &unsorted_qubits, &sorted_qubits_with_sentinel, &function](auto const first, auto const last)
                {
                  constexpr auto cache_size = ::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits);

                  for (auto iter = first; iter < last; iter += cache_size)
                    ::ket::gate::gate_detail::gate_n(parallel_policy, iter, cache_size, unsorted_qubits, sorted_qubits_with_sentinel, function);
                });
#   else // KET_USE_BIT_MASKS_EXPLICITLY
              std::array<StateInteger, num_operated_qubits> qubit_masks{};
              ::ket::gate::gate_detail::make_qubit_masks(qubit_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);
              std::array<StateInteger, num_operated_qubits + 1u> index_masks{};
              ::ket::gate::gate_detail::make_index_masks(index_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);

              return ::ket::mpi::utility::for_each_local_range(
                mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                [parallel_policy, &qubit_masks, &index_masks, &function](auto const first, auto const last)
                {
#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
                  constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
                  constexpr auto cache_size = ::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits);

                  for (auto iter = first; iter < last; iter += cache_size)
                    ::ket::gate::gate_detail::gate_n(parallel_policy, iter, cache_size, qubit_masks, index_masks, function);
                });
#   endif // KET_USE_BIT_MASKS_EXPLICITLY
            }

            template <
              typename MpiPolicy, typename ParallelPolicy,
              typename RandomAccessRange, typename StateInteger, typename Function>
            inline auto gate(
              MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment,
              StateInteger const unit_control_qubit_mask,
              Function&& function)
            -> RandomAccessRange&
            {
              using qubit_type = ::ket::qubit<StateInteger>;
              using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
              std::array<qubit_type, bit_integer_type{0u}> unsorted_qubits{};

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
              constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};

              std::array<qubit_type, bit_integer_type{1u}> sorted_qubits_with_sentinel{
                ::ket::make_qubit<StateInteger>(num_on_cache_qubits)};

              return ::ket::mpi::utility::for_each_local_range(
                mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                [parallel_policy, &unsorted_qubits, &sorted_qubits_with_sentinel, &function](auto const first, auto const last)
                {
                  constexpr auto cache_size = ::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits);

                  for (auto iter = first; iter < last; iter += cache_size)
                    ::ket::gate::gate_detail::gate_n(parallel_policy, iter, cache_size, unsorted_qubits, sorted_qubits_with_sentinel, function);
                });
#   else // KET_USE_BIT_MASKS_EXPLICITLY
              std::array<StateInteger, bit_integer_type{0u}> qubit_masks{};
              std::array<StateInteger, bit_integer_type{1u}> index_masks{compl StateInteger{0u}};

              return ::ket::mpi::utility::for_each_local_range(
                mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                [parallel_policy, &qubit_masks, &index_masks, &function](auto const first, auto const last)
                {
#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
                  constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
                  constexpr auto cache_size = ::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits);

                  for (auto iter = first; iter < last; iter += cache_size)
                    ::ket::gate::gate_detail::gate_n(parallel_policy, iter, cache_size, qubit_masks, index_masks, function);
                });
#   endif // KET_USE_BIT_MASKS_EXPLICITLY
            }
          } // namespace all_on_cache

#   ifndef KET_USE_ON_CACHE_STATE_VECTOR
          // Case 1-2) Some of the operated qubits are off-cache qubits (but not page qubits)
          //   ex: ppxx|yy|zzzzzzzz
          //         ^^             <- operated qubits
          namespace none_on_cache
          {
            // First argument of Function: ::ket::gate::utility::cache_aware_iterator<iterator_t<RandomAccessRange>> (without page) or ::ket::gate::utility::cache_aware_iterator<vector::iterator> (with page)
            template <
              typename MpiPolicy, typename ParallelPolicy,
              typename RandomAccessRange, typename StateInteger, typename Function, typename... Qubits>
            inline auto gate(
              MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment,
              StateInteger const unit_control_qubit_mask,
              Function&& function, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
            -> RandomAccessRange&
            {
              return ::ket::mpi::utility::for_each_local_range(
                mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                [parallel_policy, &function, permutated_qubits...](auto const first, auto const last)
                {
                  ::ket::gate::cache::none_on_cache::gate(
                    parallel_policy,
                    first, last, function, permutated_qubits.qubit()...);
                });
            }
          } // namespace none_on_cache

          // Case 1-3) Some of the operated qubits are off-cache qubits (but not page qubits)
          //   ex: ppxx|yyy|zzzzzzz
          //          ^ ^^     ^    <- operated qubits
          namespace some_on_cache
          {
            // First argument of Function: ::ket::gate::utility::runtime::cache_aware_iterator<iterator_t<RandomAccessRange>> (without page) or ::ket::gate::utility::runtime::cache_aware_iterator<vector::iterator> (with page)
            template <
              typename MpiPolicy, typename ParallelPolicy,
              typename RandomAccessRange, typename StateInteger, typename Function, typename... Qubits>
            inline auto gate(
              MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment,
              StateInteger const unit_control_qubit_mask,
              Function&& function, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
            -> RandomAccessRange&
            {
              return ::ket::mpi::utility::for_each_local_range(
                mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                [parallel_policy, &function, permutated_qubits...](auto const first, auto const last)
                {
                  ::ket::gate::cache::some_on_cache::gate(
                    parallel_policy,
                    first, last, function, permutated_qubits.qubit()...);
                });
            }
          } // namespace some_on_cache
#   else // KET_USE_ON_CACHE_STATE_VECTOR
          // Case 1-2) Some of the operated qubits are off-cache qubits (but not page qubits)
          //   ex: ppxx|yy|zzzzzzzz
          //         ^^             <- operated qubits
          namespace none_on_cache
          {
            // First argument of Function: vector::iterator
            template <
              typename MpiPolicy, typename ParallelPolicy,
              typename RandomAccessRange, typename BufferAllocator, typename StateInteger, typename Function, typename... Qubits>
            inline auto gate(
              MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
              yampi::communicator const& communicator, yampi::environment const& environment,
              StateInteger const unit_control_qubit_mask,
              Function&& function, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
            -> RandomAccessRange&
            {
              using qubit_type = ket::qubit<StateInteger>;
              using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
              constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
              constexpr auto cache_size = ::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits);
              assert(not ::ket::utility::all_in_state_vector(num_on_cache_qubits, permutated_qubits.qubit()...));
              assert(::ket::utility::none_in_state_vector(num_on_cache_qubits, permutated_qubits.qubit()...));

              // Case 1-2-1) Buffer size is large enough
              auto const present_buffer_size = static_cast<StateInteger>(::ket::mpi::utility::buffer_end(local_state, buffer) - ::ket::mpi::utility::buffer_begin(local_state, buffer));
              if (present_buffer_size >= cache_size)
              {
                auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);

                return ::ket::mpi::utility::for_each_local_range(
                  mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                  [parallel_policy, &function, permutated_qubits..., buffer_first](auto const first, auto const last)
                  {
                    ::ket::gate::cache::none_on_cache::gate(
                      parallel_policy,
                      first, last, buffer_first, buffer_first + cache_size,
                      function, permutated_qubits.qubit()...);
                  });
              }

              // Case 1-2-2) Buffer size is small
              if (cache_size > buffer.capacity())
                std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >{}.swap(buffer);
              buffer.resize(cache_size);

              return ::ket::mpi::utility::for_each_local_range(
                mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                [parallel_policy, &buffer, &function, permutated_qubits...](auto const first, auto const last)
                {
                  using std::begin;
                  using std::end;
                  ::ket::gate::cache::none_on_cache::gate(
                    parallel_policy,
                    first, last, begin(buffer), end(buffer),
                    function, permutated_qubits.qubit()...);
                });
            }
          } // namespace none_on_cache

          // Case 1-3) Some of the operated qubits are off-cache qubits (but not page qubits)
          //   ex: ppxx|yyy|zzzzzzz
          //          ^ ^^     ^    <- operated qubits
          namespace some_on_cache
          {
            // First argument of Function: vector::iterator
            template <
              typename MpiPolicy, typename ParallelPolicy,
              typename RandomAccessRange, typename BufferAllocator, typename StateInteger, typename Function, typename... Qubits>
            inline auto gate(
              MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
              yampi::communicator const& communicator, yampi::environment const& environment,
              StateInteger const unit_control_qubit_mask,
              Function&& function, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
            -> RandomAccessRange&
            {
              using qubit_type = ::ket::qubit<StateInteger>;
              using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
              constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
              constexpr auto cache_size = ::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits);
              assert(not ::ket::utility::all_in_state_vector(num_on_cache_qubits, permutated_qubits.qubit()...));
              assert(not ::ket::utility::none_in_state_vector(num_on_cache_qubits, permutated_qubits.qubit()...));

              // Case 1-3-1) Buffer size is large enough
              auto const present_buffer_size = static_cast<StateInteger>(::ket::mpi::utility::buffer_end(local_state, buffer) - ::ket::mpi::utility::buffer_begin(local_state, buffer));
              if (present_buffer_size >= cache_size)
              {
                auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);

                return ::ket::mpi::utility::for_each_local_range(
                  mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                  [parallel_policy, &function, permutated_qubits..., buffer_first](auto const first, auto const last)
                  {
                    ::ket::gate::cache::some_on_cache::gate(
                      parallel_policy,
                      first, last, buffer_first, buffer_first + cache_size,
                      function, permutated_qubits.qubit()...);
                  });
              }

              // Case 1-3-2) Buffer size is small
              if (cache_size > buffer.capacity())
                std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >{}.swap(buffer);
              buffer.resize(cache_size);

              return ::ket::mpi::utility::for_each_local_range(
                mpi_policy, local_state, communicator, environment, unit_control_qubit_mask,
                [parallel_policy, &buffer, &function, permutated_qubits...](auto const first, auto const last)
                {
                  using std::begin;
                  using std::end;
                  ::ket::gate::cache::some_on_cache::gate(
                    parallel_policy,
                    first, last, begin(buffer), end(buffer),
                    function, permutated_qubits.qubit()...);
                });
            }
          } // namespace some_on_cache

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename BufferAllocator, typename StateInteger, typename Function, typename Qubit, typename... Qubits>
          inline auto gate(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask,
            Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
          -> RandomAccessRange&
          {
            static_assert(std::is_same<StateInteger, ::ket::meta::state_integer_t<Qubit>>::value, "The state_integer_type of Qubit should be the same as StateInteger");
            using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
            constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
            constexpr auto cache_size = ::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits);

            // Case 1-1) All operated qubits are on-cache qubits
            //   ex1: ppxx|zzzzzzzzzz
            //              ^   ^ ^   <- operated qubits
            //   ex2: pppp|ppzzzzzzzz
            //               ^  ^  ^  <- operated qubits
            if (::ket::utility::all_in_state_vector(num_on_cache_qubits, permutated_qubit.qubit(), permutated_qubits.qubit()...))
            {
              // Case 1-1-1) page size <= on-cache state size
              //   ex1: pppp|ppzzzzzzzz
              //               ^  ^  ^  <- operated qubits
              //   ex2: ....|..ppzzzzzz (num. local qubits <= num. on-cache qubits)
              //                  ^  ^  <- operated qubits
              if (::ket::mpi::page::page_size(mpi_policy, local_state, communicator, environment) <= cache_size)
                return ::ket::mpi::gate::local::nopage::all_on_cache::small::gate(
                  mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
                  std::forward<Function>(function), permutated_qubit, permutated_qubits...);

              // Case 1-1-2) page size > on-cache state size
              //   ex: ppxx|zzzzzzzzzz
              //             ^   ^ ^   <- operated qubits
              return ::ket::mpi::gate::local::nopage::all_on_cache::gate(
                mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
                std::forward<Function>(function), permutated_qubit, permutated_qubits...);
            }

            // Case 1-2) None of the operated qubits are on-cache qubits (but not page qubits)
            //   ex: ppxx|yy|zzzzzzzz
            //         ^^             <- operated qubits
            if (::ket::utility::none_in_state_vector(num_on_cache_qubits, permutated_qubit.qubit(), permutated_qubits.qubit()...))
              return ::ket::mpi::gate::local::nopage::none_on_cache::gate(
                mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask,
                std::forward<Function>(function), permutated_qubit, permutated_qubits...);

            // Case 1-3) Some of the operated qubits are off-cache qubits (but not page qubits)
            //   ex: ppxx|yyy|zzzzzzz
            //          ^ ^^     ^    <- operated qubits
            return ::ket::mpi::gate::local::nopage::some_on_cache::gate(
              mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask,
              std::forward<Function>(function), permutated_qubit, permutated_qubits...);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename BufferAllocator, typename StateInteger, typename Function>
          inline auto gate(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask,
            Function&& function)
          -> RandomAccessRange&
          {
            using qubit_type = ::ket::qubit<StateInteger>;
            using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
            constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
            constexpr auto cache_size = ::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits);

            // Case 1-1) All operated qubits are on-cache qubits
            //   ex1: ppxx|zzzzzzzzzz
            //              ^   ^ ^   <- operated qubits
            //   ex2: pppp|ppzzzzzzzz
            //               ^  ^  ^  <- operated qubits
            // ALWAYS SATISFIED
            // CACHE-AWARENESS IS NOT NEEDED IF WE HAVE NO QUBITS

            // Case 1-1-1) page size <= on-cache state size
            //   ex1: pppp|ppzzzzzzzz
            //               ^  ^  ^  <- operated qubits
            //   ex2: ....|..ppzzzzzz (num. local qubits <= num. on-cache qubits)
            //                  ^  ^  <- operated qubits
            if (::ket::mpi::page::page_size(mpi_policy, local_state, communicator, environment) <= cache_size)
              return ::ket::mpi::gate::local::nopage::all_on_cache::small::gate(
                mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
                std::forward<Function>(function));

            // Case 1-1-2) page size > on-cache state size
            //   ex: ppxx|zzzzzzzzzz
            //             ^   ^ ^   <- operated qubits
            return ::ket::mpi::gate::local::nopage::all_on_cache::gate(
              mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
              std::forward<Function>(function));
          }
#   endif // KET_USE_ON_CACHE_STATE_VECTOR
        } // namespace nopage

        // Case 2) Some operated qubits are page qubits
        //   ex1: pppp|ppzzzzzzzz
        //             ^    ^ ^   <- operated qubits
        //   ex2: ppxx|zzzzzzzzzz
        //         ^    ^   ^ ^   <- operated qubits
        //   ex3: ppxx|zzzzzzzzzz
        //         ^^   ^     ^   <- operated qubits
        namespace page
        {
#   ifndef KET_USE_ON_CACHE_STATE_VECTOR
          namespace all_on_cache
          {
            // ....|..ppzzzzzz (num. qubits <= num. on-cache qubits)
            //         ^   ^   <- operated qubits
            namespace small
            {
              // First argument of Function: iterator_t<RandomAccessRange> a.k.a. page_iterator
              template <
                typename MpiPolicy, typename ParallelPolicy,
                typename RandomAccessRange, typename StateInteger, typename Function, typename Qubit, typename... Qubits>
              inline auto gate(
                MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
                RandomAccessRange& local_state,
                yampi::communicator const& communicator, yampi::environment const& environment,
                StateInteger const unit_control_qubit_mask,
                Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
              -> RandomAccessRange&
              {
                auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
                auto const data_block_size = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, rank_in_unit));
                auto const num_data_blocks = static_cast<StateInteger>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));

                using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;
                constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};
#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
                using qubit_type = ::ket::qubit<StateInteger, bit_integer_type>;
                std::array<qubit_type, num_operated_qubits> unsorted_qubits{
                  ::ket::remove_control(permutated_qubit.qubit()), ::ket::remove_control(permutated_qubits.qubit())...};

                auto const num_qubits = ::ket::mpi::utility::policy::num_qubits(mpi_policy, local_state, communicator, environment);

                std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_qubits_with_sentinel{
                  ::ket::remove_control(permutated_qubit.qubit()), ::ket::remove_control(permutated_qubits.qubit())...,
                  ::ket::make_qubit<StateInteger>(num_qubits)};
                using std::begin;
                using std::end;
                std::sort(begin(sorted_qubits_with_sentinel), std::prev(end(sorted_qubits_with_sentinel)));

                auto const first = begin(local_state);
                for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
                {
                  if ((static_cast<StateInteger>(::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit)) bitand unit_control_qubit_mask) != unit_control_qubit_mask)
                    continue;

                  ::ket::gate::gate_detail::gate(
                    parallel_policy,
                    first + data_block_index * data_block_size,
                    first + (data_block_index + 1u) * data_block_size,
                    unsorted_qubits, sorted_qubits_with_sentinel, std::forward<Function>(function));
                }
#   else // KET_USE_BIT_MASKS_EXPLICITLY
                std::array<StateInteger, num_operated_qubits> qubit_masks{};
                ::ket::gate::gate_detail::make_qubit_masks(qubit_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);
                std::array<StateInteger, num_operated_qubits + 1u> index_masks{};
                ::ket::gate::gate_detail::make_index_masks(index_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);

                auto const first = begin(local_state);
                for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
                {
                  if ((static_cast<StateInteger>(::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit)) bitand unit_control_qubit_mask) != unit_control_qubit_mask)
                    continue;

                  ::ket::gate::gate_detail::gate(
                    parallel_policy,
                    first + data_block_index * data_block_size,
                    first + (data_block_index + 1u) * data_block_size,
                    qubit_masks, index_masks, std::forward<Function>(function));
                }
#   endif // KET_USE_BIT_MASKS_EXPLICITLY

                return local_state;
              }

              template <
                typename MpiPolicy, typename ParallelPolicy,
                typename RandomAccessRange, typename StateInteger, typename Function>
              inline auto gate(
                MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
                RandomAccessRange& local_state,
                yampi::communicator const& communicator, yampi::environment const& environment,
                StateInteger const unit_control_qubit_mask,
                Function&& function)
              -> RandomAccessRange&
              {
                auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
                auto const data_block_size = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, rank_in_unit));
                auto const num_data_blocks = static_cast<StateInteger>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));

                using qubit_type = ::ket::qubit<StateInteger>;
                using bit_integer_type = ::ket::meta::bit_integer_t<qubit_type>;

#   ifndef KET_USE_BIT_MASKS_EXPLICITLY
                std::array<qubit_type, bit_integer_type{0u}> unsorted_qubits{};

                auto const num_qubits = ::ket::mpi::utility::policy::num_qubits(mpi_policy, local_state, communicator, environment);

                std::array<qubit_type, bit_integer_type{1u}> sorted_qubits_with_sentinel{::ket::make_qubit<StateInteger>(num_qubits)};

                auto const first = begin(local_state);
                for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
                {
                  if ((static_cast<StateInteger>(::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit)) bitand unit_control_qubit_mask) != unit_control_qubit_mask)
                    continue;

                  ::ket::gate::gate_detail::gate(
                    parallel_policy,
                    first + data_block_index * data_block_size,
                    first + (data_block_index + 1u) * data_block_size,
                    unsorted_qubits, sorted_qubits_with_sentinel, std::forward<Function>(function));
                }
#   else // KET_USE_BIT_MASKS_EXPLICITLY
                std::array<StateInteger, bit_integer_type{0u}> qubit_masks{};
                std::array<StateInteger, bit_integer_type{1u}> index_masks{compl StateInteger{0u}};

                auto const first = begin(local_state);
                for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
                {
                  if ((static_cast<StateInteger>(::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit)) bitand unit_control_qubit_mask) != unit_control_qubit_mask)
                    continue;

                  ::ket::gate::gate_detail::gate(
                    parallel_policy,
                    first + data_block_index * data_block_size,
                    first + (data_block_index + 1u) * data_block_size,
                    qubit_masks, index_masks, std::forward<Function>(function));
                }
#   endif // KET_USE_BIT_MASKS_EXPLICITLY

                return local_state;
              }
            } // namespace small

            // First argument of Function: iterator_t<RandomAccessRange> a.k.a. page_iterator
            template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename Function, typename... Qubits>
            inline auto gate(
              MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment,
              StateInteger const unit_control_qubit_mask,
              Function&& function, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
            -> RandomAccessRange&
            {
              auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
              auto const data_block_size = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, rank_in_unit));
              auto const num_data_blocks = static_cast<StateInteger>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));

              using std::begin;
              auto const first = begin(local_state);
              for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
              {
                if ((static_cast<StateInteger>(::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit)) bitand unit_control_qubit_mask) != unit_control_qubit_mask)
                  continue;

                ::ket::gate::cache::all_on_cache::gate(
                  parallel_policy,
                  first + data_block_index * data_block_size,
                  first + (data_block_index + 1u) * data_block_size,
                  std::forward<Function>(function), permutated_qubits.qubit()...);
              }

              return local_state;
            }
          } // namespace all_on_cache

          namespace none_on_cache
          {
            // First argument of Function: ::ket::gate::utility::cache_aware_iterator<iterator_t<RandomAccessRange>> a.k.a. ::ket::gate::utility::cache_aware_iterator<page_iterator>
            template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename Function, typename... Qubits>
            inline auto gate(
              MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment,
              StateInteger const unit_control_qubit_mask,
              Function&& function, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
            -> RandomAccessRange&
            {
              auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
              auto const data_block_size = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, rank_in_unit));
              auto const num_data_blocks = static_cast<StateInteger>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));

              using std::begin;
              auto const first = begin(local_state);
              for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
              {
                if ((static_cast<StateInteger>(::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit)) bitand unit_control_qubit_mask) != unit_control_qubit_mask)
                  continue;

                ::ket::gate::cache::none_on_cache::gate(
                  parallel_policy,
                  first + data_block_index * data_block_size,
                  first + (data_block_index + 1u) * data_block_size,
                  std::forward<Function>(function), permutated_qubits.qubit()...);
              }

              return local_state;
            }
          } // namespace none_on_cache

          namespace some_on_cache
          {
            // First argument of Function: ::ket::gate::utility::cache_aware_iterator<iterator_t<RandomAccessRange>> a.k.a. ::ket::gate::utility::cache_aware_iterator<page_iterator>
            template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename Function, typename... Qubits>
            inline auto gate(
              MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment,
              StateInteger const unit_control_qubit_mask,
              Function&& function, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
            -> RandomAccessRange&
            {
              auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
              auto const data_block_size = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, rank_in_unit));
              auto const num_data_blocks = static_cast<StateInteger>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));

              using std::begin;
              auto const first = begin(local_state);
              for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
              {
                if ((static_cast<StateInteger>(::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit)) bitand unit_control_qubit_mask) != unit_control_qubit_mask)
                  continue;

                ::ket::gate::cache::some_on_cache::gate(
                  parallel_policy,
                  first + data_block_index * data_block_size,
                  first + (data_block_index + 1u) * data_block_size,
                  std::forward<Function>(function), permutated_qubits.qubit()...);
              }

              return local_state;
            }
          } // namespace some_on_cache
#   else // KET_USE_ON_CACHE_STATE_VECTOR
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename BufferAllocator, typename StateInteger, typename Function, typename Qubit, typename... Qubits>
          inline auto gate(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask,
            Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
          -> RandomAccessRange&
          {
            static_assert(std::is_same<StateInteger, ::ket::meta::state_integer_t<Qubit>>::value, "The state_integer_type of Qubit should be the same as StateInteger");
            using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
            constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
            constexpr auto on_cache_state_size = ::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits);

            // xxxx|yyyy|zzzzzz: local qubits
            // * xxxx: off-cache qubits
            // * yyyy|zzzzzz: on-cache qubits
            //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)

            //   ex1: pppp|ppzzzzzzzz
            //             ^    ^ ^   <- operated qubits
            //   ex2: ppxx|zzzzzzzzzz
            //         ^    ^   ^ ^   <- operated qubits
            //   ex3: ppxx|zzzzzzzzzz
            //         ^^   ^     ^   <- operated qubits
            assert(::ket::mpi::page::any_on_page(local_state, permutated_qubit, permutated_qubits...));
            // Redefine on-cache state as its size is std::min(on_cache_state_size, page_size), then num. page qubits <= num. off-cache qubits becomes always to hold
            auto const modified_on_cache_state_size = std::min(on_cache_state_size, ::ket::mpi::page::page_size(mpi_policy, local_state, communicator, environment));

            auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
            auto const num_data_blocks = static_cast<StateInteger>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));

            // Case 2-1) Buffer size is large enough
            auto const present_buffer_size = static_cast<StateInteger>(::ket::mpi::utility::buffer_end(local_state, buffer) - ::ket::mpi::utility::buffer_begin(local_state, buffer));
            if (present_buffer_size >= modified_on_cache_state_size)
            {
              auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
              for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
              {
                if ((static_cast<StateInteger>(::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit)) bitand unit_control_qubit_mask) != unit_control_qubit_mask)
                  continue;

                ::ket::mpi::gate::page::gate(
                  parallel_policy,
                  local_state, buffer_first, buffer_first + modified_on_cache_state_size, data_block_index,
                  std::forward<Function>(function), permutated_qubit, permutated_qubits...);
              }

              return local_state;
            }

            // Case 2-2) Buffer size is small
            if (modified_on_cache_state_size > buffer.capacity())
              std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >{}.swap(buffer);
            buffer.resize(modified_on_cache_state_size);

            for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
              ::ket::mpi::gate::page::gate(
                parallel_policy,
                local_state, begin(buffer), end(buffer), data_block_index,
                std::forward<Function>(function), permutated_qubit, permutated_qubits...);

            return local_state;
          }

          // should not be called
          // TODO: throw not just an integer value but any error class
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename BufferAllocator, typename StateInteger, typename Function>
          inline auto gate(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask,
            Function&& function)
          -> RandomAccessRange&
          { throw 1; }
#   endif // KET_USE_ON_CACHE_STATE_VECTOR
        } // namespace page

#   ifndef KET_USE_ON_CACHE_STATE_VECTOR
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename BufferAllocator, typename StateInteger,
          typename Function, typename Qubit, typename... Qubits>
        inline auto gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          StateInteger const unit_control_qubit_mask,
          Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
        -> RandomAccessRange&
        {
          static_assert(std::is_same<StateInteger, ::ket::meta::state_integer_t<Qubit>>::value, "The state_integer_type of Qubit should be the same as StateInteger");
          using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
          constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
          constexpr auto cache_size = ::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits);

          // xxxx|yyyy|zzzzzz: local qubits
          // * xxxx: off-cache qubits
          // * yyyy|zzzzzz: on-cache qubits
          //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)

          // Case 1) None of operated qubits is page qubit
          if (::ket::mpi::page::none_on_page(local_state, permutated_qubit, permutated_qubits...))
          {
            // Case 1-1) All operated qubits are on-cache qubits
            //   ex1: ppxx|zzzzzzzzzz
            //              ^   ^ ^   <- operated qubits
            //   ex2: pppp|ppzzzzzzzz
            //               ^  ^  ^  <- operated qubits
            if (::ket::utility::all_in_state_vector(num_on_cache_qubits, permutated_qubit.qubit(), permutated_qubits.qubit()...))
            {
              // Case 1-1-1) page size <= on-cache state size
              //   ex1: pppp|ppzzzzzzzz
              //               ^  ^  ^  <- operated qubits
              //   ex2: ....|..ppzzzzzz (num. local qubits <= num. on-cache qubits)
              //                  ^  ^  <- operated qubits
              if (::ket::mpi::page::page_size(mpi_policy, local_state, communicator, environment) <= cache_size)
                return ::ket::mpi::gate::local::nopage::all_on_cache::small::gate(
                  mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
                  std::forward<Function>(function), permutated_qubit, permutated_qubits...);

              // Case 1-1-2) page size > on-cache state size
              //   ex: ppxx|zzzzzzzzzz
              //             ^   ^ ^   <- operated qubits
              return ::ket::mpi::gate::local::nopage::all_on_cache::gate(
                mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
                std::forward<Function>(function), permutated_qubit, permutated_qubits...);
            }

            // Case 1-2) None of the operated qubits are on-cache qubits (but not page qubits)
            //   ex: ppxx|yy|zzzzzzzz
            //         ^^             <- operated qubits
            if (::ket::utility::none_in_state_vector(num_on_cache_qubits, permutated_qubit.qubit(), permutated_qubits.qubit()...))
              return ::ket::mpi::gate::local::nopage::none_on_cache::gate(
                mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
                std::forward<Function>(function), permutated_qubit, permutated_qubits...);

            // Case 1-3) Some of the operated qubits are off-cache qubits (but not page qubits)
            //   ex: ppxx|yyy|zzzzzzz
            //          ^ ^^     ^    <- operated qubits
            return ::ket::mpi::gate::local::nopage::some_on_cache::gate(
              mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
              std::forward<Function>(function), permutated_qubit, permutated_qubits...);
          }

          // Case 2) Some operated qubits are page qubits
          //   ex1: pppp|ppzzzzzzzz
          //             ^    ^ ^   <- operated qubits
          //   ex2: ppxx|zzzzzzzzzz
          //         ^    ^   ^ ^   <- operated qubits
          //   ex3: ppxx|zzzzzzzzzz
          //         ^^   ^     ^   <- operated qubits
          if (::ket::utility::all_in_state_vector(num_on_cache_qubits, permutated_qubit.qubit(), permutated_qubits.qubit()...))
          {
            // ....|..ppzzzzzz (num. qubits <= num. on-cache qubits)
            //         ^   ^   <- operated qubits
            if (::ket::mpi::utility::policy::num_qubits(mpi_policy, local_state, communicator, environment) <= num_on_cache_qubits)
              return ::ket::mpi::gate::local::page::all_on_cache::small::gate(
                mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
                std::forward<Function>(function), permutated_qubit, permutated_qubits...);

            return ::ket::mpi::gate::local::page::all_on_cache::gate(
              mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
              std::forward<Function>(function), permutated_qubit, permutated_qubits...);
          }

          if (::ket::utility::none_in_state_vector(num_on_cache_qubits, permutated_qubit.qubit(), permutated_qubits.qubit()...))
            return ::ket::mpi::gate::local::page::none_on_cache::gate(
              mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
              std::forward<Function>(function), permutated_qubit, permutated_qubits...);

          return ::ket::mpi::gate::local::page::some_on_cache::gate(
            mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
            std::forward<Function>(function), permutated_qubit, permutated_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename BufferAllocator, typename StateInteger,
          typename Function>
        inline auto gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          StateInteger const unit_control_qubit_mask,
          Function&& function)
        -> RandomAccessRange&
        {
          using bit_integer_type = ::ket::meta::bit_integer_t< ::ket::qubit<StateInteger> >;

#     ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#       define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#     endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
          constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
          constexpr auto cache_size = ::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits);

          // xxxx|yyyy|zzzzzz: local qubits
          // * xxxx: off-cache qubits
          // * yyyy|zzzzzz: on-cache qubits
          //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)

          // Case 1) None of operated qubits is page qubit
          // ALWAYS SATISFIED

          // Case 1-1) All operated qubits are on-cache qubits
          //   ex1: ppxx|zzzzzzzzzz
          //              ^   ^ ^   <- operated qubits
          //   ex2: pppp|ppzzzzzzzz
          //               ^  ^  ^  <- operated qubits
          // ALWAYS SATISFIED
          // CACHE-AWARENESS IS NOT NEEDED IF WE HAVE NO QUBITS

          // Case 1-1-1) page size <= on-cache state size
          //   ex1: pppp|ppzzzzzzzz
          //               ^  ^  ^  <- operated qubits
          //   ex2: ....|..ppzzzzzz (num. local qubits <= num. on-cache qubits)
          //                  ^  ^  <- operated qubits
          if (::ket::mpi::page::page_size(mpi_policy, local_state, communicator, environment) <= cache_size)
            return ::ket::mpi::gate::local::nopage::all_on_cache::small::gate(
              mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
              std::forward<Function>(function));

          // Case 1-1-2) page size > on-cache state size
          //   ex: ppxx|zzzzzzzzzz
          //             ^   ^ ^   <- operated qubits
          return ::ket::mpi::gate::local::nopage::all_on_cache::gate(
            mpi_policy, parallel_policy, local_state, communicator, environment, unit_control_qubit_mask,
            std::forward<Function>(function));
        }
#   else // KET_USE_ON_CACHE_STATE_VECTOR
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename BufferAllocator, typename StateInteger,
          typename Function, typename Qubit, typename... Qubits>
        inline auto gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          StateInteger const unit_control_qubit_mask,
          Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
        -> RandomAccessRange&
        {
          // xxxx|yyyy|zzzzzz: local qubits
          // * xxxx: off-cache qubits
          // * yyyy|zzzzzz: on-cache qubits
          //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)

          // Case 1) None of operated qubits is page qubit
          if (::ket::mpi::page::none_on_page(local_state, permutated_qubit, permutated_qubits...))
            return ::ket::mpi::gate::local::nopage::gate(
              mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask,
              std::forward<Function>(function), permutated_qubit, permutated_qubits...);

          // Case 2) Some operated qubits are page qubits
          //   ex1: pppp|ppzzzzzzzz
          //             ^    ^ ^   <- operated qubits
          //   ex2: ppxx|zzzzzzzzzz
          //         ^    ^   ^ ^   <- operated qubits
          //   ex3: ppxx|zzzzzzzzzz
          //         ^^   ^     ^   <- operated qubits
          return ::ket::mpi::gate::local::page::gate(
            mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask,
            std::forward<Function>(function), permutated_qubit, permutated_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename BufferAllocator, typename StateInteger,
          typename Function>
        inline auto gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          StateInteger const unit_control_qubit_mask,
          Function&& function)
        -> RandomAccessRange&
        {
          // xxxx|yyyy|zzzzzz: local qubits
          // * xxxx: off-cache qubits
          // * yyyy|zzzzzz: on-cache qubits
          //   - yyyy: chunk qubits (chunk qubits are determined dynamically, and sometimes there is no chunk qubit)

          // Case 1) None of operated qubits is page qubit
          // ALWAYS SATISFIED
          return ::ket::mpi::gate::local::nopage::gate(
            mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask,
            std::forward<Function>(function));
        }
#   endif // KET_USE_ON_CACHE_STATE_VECTOR
# endif // KET_ENABLE_CACHE_AWARE_GATE_FUNCTION
      } // namespace local

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename Function, typename... Qubits>
      inline auto gate(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Function&& function, Qubits&&... qubits)
      -> RandomAccessRange&
      {
        constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, std::remove_reference_t<Qubits>...>::value;
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(num_control_qubits, 'C').append("Gate"), qubits...),
          environment};

        return ::ket::mpi::utility::apply_local_gate(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          [&function](
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, auto&&... permutated_qubits)
          {
            return ::ket::mpi::gate::local::gate(
              mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask,
              function, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);
          }, std::forward<Qubits>(qubits)...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype,
        typename Function, typename... Qubits>
      inline auto gate(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Function&& function, Qubits&&... qubits)
      -> RandomAccessRange&
      {
        constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, std::remove_reference_t<Qubits>...>::value;
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(num_control_qubits, 'C').append("Gate"), qubits...),
          environment};

        return ::ket::mpi::utility::apply_local_gate(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          [&function](
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, auto&&... permutated_qubits)
          {
            return ::ket::mpi::gate::local::gate(
              mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask,
              function, std::forward<decltype(permutated_qubits)>(permutated_qubits)...);
          }, std::forward<Qubits>(qubits)...);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_GATE_HPP
