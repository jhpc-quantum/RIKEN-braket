#ifndef KET_MPI_GATE_GATE_HPP
# define KET_MPI_GATE_GATE_HPP

# include <algorithm>
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
#   include <array>
# endif // KET_USE_BIT_MASKS_EXPLICITLY
# include <vector>
# include <iterator>
# include <utility>

# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/all_in_state_vector.hpp>
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
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/buffer_range.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace local
      {
# ifndef KET_USE_ON_CACHE_STATE_VECTOR
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename BufferAllocator, typename Function, typename Qubit, typename... Qubits>
        inline auto gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
        -> RandomAccessRange&
        {
          if (::ket::mpi::page::none_on_page(local_state, permutated_qubit, permutated_qubits...))
            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, &function, permutated_qubit, permutated_qubits...](auto const first, auto const last)
              { ::ket::gate::nocache::gate(parallel_policy, first, last, function, permutated_qubit.qubit(), permutated_qubits.qubit()...); });

          auto const data_block_size
            = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
          auto const num_data_blocks
            = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);

          using std::begin;
          auto const first = begin(local_state);
          for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
            ::ket::gate::nocache::gate(
              parallel_policy,
              first + data_block_index * data_block_size,
              first + (data_block_index + 1u) * data_block_size,
              function, permutated_qubit.qubit(), permutated_qubits.qubit()...);

          return local_state;
        }
# else // KET_USE_ON_CACHE_STATE_VECTOR
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename BufferAllocator, typename Function, typename Qubit, typename... Qubits>
        inline auto gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
        -> RandomAccessRange&
        {
          using state_integer_type = ::ket::meta::state_integer_t<Qubit>;
          using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;

#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
          constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
          constexpr auto on_cache_state_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);

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
              constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
              using qubit_type = ::ket::qubit<state_integer_type, bit_integer_type>;
              std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_qubits_with_sentinel{
                ::ket::remove_control(permutated_qubit.qubit()), ::ket::remove_control(permutated_qubits.qubit())...,
                ::ket::make_qubit<state_integer_type>(num_on_cache_qubits)};
              using std::begin;
              using std::end;
              std::sort(begin(sorted_qubits_with_sentinel), std::prev(end(sorted_qubits_with_sentinel)));

              std::array<qubit_type, num_operated_qubits> unsorted_qubits{
                ::ket::remove_control(permutated_qubit.qubit()), ::ket::remove_control(permutated_qubits.qubit())...};
# else // KET_USE_BIT_MASKS_EXPLICITLY
              std::array<state_integer_type, num_operated_qubits> qubit_masks{};
              ::ket::gate::gate_detail::make_qubit_masks(qubit_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);
              std::array<state_integer_type, num_operated_qubits + 1u> index_masks{};
              ::ket::gate::gate_detail::make_index_masks(index_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY

              // Case 1-1-1) page size <= on-cache state size
              //   ex1: pppp|ppzzzzzzzz
              //               ^  ^  ^  <- operated qubits
              //   ex2: ....|..ppzzzzzz (num. local qubits <= num. on-cache qubits)
              //                  ^  ^  <- operated qubits
              // all operated qubits are non-page qubits, so ket::gate::nocache::gate can be used
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
              if (::ket::mpi::page::page_size(mpi_policy, local_state, communicator, environment) <= on_cache_state_size)
                return ::ket::mpi::utility::for_each_local_range(
                  mpi_policy, local_state, communicator, environment,
                  [parallel_policy, &sorted_qubits_with_sentinel, &unsorted_qubits, &function](auto const first, auto const last)
                  { ::ket::gate::gate_detail::gate(parallel_policy, first, last, unsorted_qubits, sorted_qubits_with_sentinel, function); });
# else // KET_USE_BIT_MASKS_EXPLICITLY
              if (::ket::mpi::page::page_size(mpi_policy, local_state, communicator, environment) <= on_cache_state_size)
                return ::ket::mpi::utility::for_each_local_range(
                  mpi_policy, local_state, communicator, environment,
                  [parallel_policy, &qubit_masks, &index_masks, &function](auto const first, auto const last)
                  { ::ket::gate::gate_detail::gate(parallel_policy, first, last, qubit_masks, index_masks, function); });
# endif // KET_USE_BIT_MASKS_EXPLICITLY

              // Case 1-1-2) page size > on-cache state size
              //   ex: ppxx|zzzzzzzzzz
              //             ^   ^ ^   <- operated qubits
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
              return ::ket::mpi::utility::for_each_local_range(
                mpi_policy, local_state, communicator, environment,
                [parallel_policy, &sorted_qubits_with_sentinel, &unsorted_qubits, &function](auto const first, auto const last)
                {
                  for (auto iter = first; iter < last; iter += on_cache_state_size)
                    ::ket::gate::gate_detail::gate(parallel_policy, iter, iter + on_cache_state_size, unsorted_qubits, sorted_qubits_with_sentinel, function);
                });
# else // KET_USE_BIT_MASKS_EXPLICITLY
              return ::ket::mpi::utility::for_each_local_range(
                mpi_policy, local_state, communicator, environment,
                [parallel_policy, &qubit_masks, &index_masks, &function](auto const first, auto const last)
                {
                  for (auto iter = first; iter < last; iter += on_cache_state_size)
                    ::ket::gate::gate_detail::gate(parallel_policy, iter, iter + on_cache_state_size, qubit_masks, index_masks, function);
                });
# endif // KET_USE_BIT_MASKS_EXPLICITLY
            }

            // Case 1-2) Some of the operated qubits are off-cache qubits (but not page qubits)
            //   ex1: ppxx|yy|zzzzzzzz
            //          ^^             <- operated qubits
            //   ex2: ppxx|yyy|zzzzzzz
            //           ^ ^^     ^    <- operated qubits
            // Case 1-2-1) Buffer size is large enough
            auto const present_buffer_size = static_cast<state_integer_type>(::ket::mpi::utility::buffer_end(local_state, buffer) - ::ket::mpi::utility::buffer_begin(local_state, buffer));
            if (present_buffer_size >= on_cache_state_size)
            {
              auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
              return ::ket::mpi::utility::for_each_local_range(
                mpi_policy, local_state, communicator, environment,
                [parallel_policy, &function, permutated_qubit, permutated_qubits..., buffer_first](auto const first, auto const last)
                {
                  ::ket::gate::cache::unsafe::gate(
                    parallel_policy,
                    first, last, buffer_first, buffer_first + on_cache_state_size,
                    function, permutated_qubit.qubit(), permutated_qubits.qubit()...);
                });
            }

            // Case 1-2-2) Buffer size is small
            buffer.resize(on_cache_state_size);
            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, &buffer, &function, permutated_qubit, permutated_qubits...](auto const first, auto const last)
              {
                ::ket::gate::cache::unsafe::gate(
                  parallel_policy,
                  first, last, begin(buffer), end(buffer),
                  function, permutated_qubit.qubit(), permutated_qubits.qubit()...);
              });
          }

          // Case 2) Some operated qubits are page qubits
          //   ex1: pppp|ppzzzzzzzz
          //             ^    ^ ^   <- operated qubits
          //   ex2: ppxx|zzzzzzzzzz
          //         ^    ^   ^ ^   <- operated qubits
          //   ex3: ppxx|zzzzzzzzzz
          //         ^^   ^     ^   <- operated qubits
          assert(::ket::mpi::page::any_on_page(local_state, permutated_qubit, permutated_qubits...));
          // Redefine on-cache state as its size is std::min(on_cache_state_size, page_size), then num. page qubits <= num. off-cache qubits becomes always to hold
          auto const modified_on_cache_state_size = std::min(on_cache_state_size, ::ket::mpi::page::page_size(mpi_policy, local_state, communicator, environment));

          auto const num_data_blocks = static_cast<state_integer_type>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment));

          // Case 2-1) Buffer size is large enough
          auto const present_buffer_size = static_cast<state_integer_type>(::ket::mpi::utility::buffer_end(local_state, buffer) - ::ket::mpi::utility::buffer_begin(local_state, buffer));
          if (present_buffer_size >= modified_on_cache_state_size)
          {
            auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
            for (auto data_block_index = state_integer_type{0u}; data_block_index < num_data_blocks; ++data_block_index)
              ::ket::mpi::gate::page::gate(
                parallel_policy,
                local_state, buffer_first, buffer_first + modified_on_cache_state_size, data_block_index,
                std::forward<Function>(function), permutated_qubit, permutated_qubits...);

            return local_state;
          }

          // Case 2-2) Buffer size is small
          buffer.resize(modified_on_cache_state_size);
          for (auto data_block_index = state_integer_type{0u}; data_block_index < num_data_blocks; ++data_block_index)
            ::ket::mpi::gate::page::gate(
              parallel_policy,
              local_state, begin(buffer), end(buffer), data_block_index,
              std::forward<Function>(function), permutated_qubit, permutated_qubits...);

          return local_state;
        }
# endif // KET_USE_ON_CACHE_STATE_VECTOR
      } // namespace local

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename Function, typename Qubit, typename... Qubits>
      inline auto gate(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Function&& function, Qubit&& qubit, Qubits&&... qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);

        return ::ket::mpi::gate::local::gate(
          mpi_policy, parallel_policy, local_state, buffer, communicator, environment,
          std::forward<Function>(function), permutation[std::forward<Qubit>(qubit)], permutation[std::forward<Qubits>(qubits)]...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype,
        typename Function, typename Qubit, typename... Qubits>
      inline auto gate(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Function&& function, Qubit&& qubit, Qubits&&... qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);

        return ::ket::mpi::gate::local::gate(
          mpi_policy, parallel_policy, local_state, communicator, environment,
          std::forward<Function>(function), permutation[std::forward<Qubit>(qubit)], permutation[std::forward<Qubits>(qubits)]...);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_GATE_HPP
